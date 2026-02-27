#!/usr/bin/env bash
#
# CPQAI — Google Cloud infrastructure setup
#
# Creates:  Artifact Registry, Compute Engine VM (PostgreSQL + AGE + pgvector),
#           Secret Manager secrets, IAM service account, VPC networking,
#           and deploys the first Cloud Run revision.
#
# Prerequisites:
#   - gcloud CLI authenticated (`gcloud auth login`)
#   - A GCP project with billing enabled
#   - Owner or Editor role on the project
#
# Usage:
#   chmod +x deploy/gcp/setup.sh
#   ./deploy/gcp/setup.sh              # interactive prompts
#   ./deploy/gcp/setup.sh --defaults   # use all defaults (us-central1)

set -euo pipefail

# ── Colour helpers ──────────────────────────────────────────────────
info()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
ok()    { printf '\033[1;32m[OK]\033[0m    %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
err()   { printf '\033[1;31m[ERR]\033[0m   %s\n' "$*" >&2; }

# ── Configuration ───────────────────────────────────────────────────
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [[ -z "$PROJECT_ID" ]]; then
  err "No active GCP project. Run: gcloud config set project YOUR_PROJECT_ID"
  exit 1
fi

if [[ "${1:-}" == "--defaults" ]]; then
  REGION="us-central1"
  ZONE="${REGION}-a"
  SERVICE_NAME="cpqai"
  DB_VM_NAME="cpqai-db"
  DB_NAME="cpqai"
  DB_USER="cpqai"
  DB_VM_TYPE="e2-small"        # 2 vCPU shared, 2 GB RAM — ~$15/month
  DB_DISK_SIZE="50"            # GB SSD for PostgreSQL data
  REPO_NAME="cpqai"
  VPC_NETWORK="default"
  VPC_SUBNET="default"
else
  read -rp "Region [us-central1]: "              REGION;      REGION=${REGION:-us-central1}
  read -rp "Zone [${REGION:-us-central1}-a]: "   ZONE;        ZONE=${ZONE:-${REGION:-us-central1}-a}
  read -rp "Cloud Run service [cpqai]: "         SERVICE_NAME; SERVICE_NAME=${SERVICE_NAME:-cpqai}
  read -rp "DB VM name [cpqai-db]: "             DB_VM_NAME;  DB_VM_NAME=${DB_VM_NAME:-cpqai-db}
  read -rp "Database name [cpqai]: "             DB_NAME;     DB_NAME=${DB_NAME:-cpqai}
  read -rp "Database user [cpqai]: "             DB_USER;     DB_USER=${DB_USER:-cpqai}
  read -rp "DB VM type [e2-small]: "             DB_VM_TYPE;  DB_VM_TYPE=${DB_VM_TYPE:-e2-small}
  read -rp "DB disk size GB [50]: "              DB_DISK_SIZE; DB_DISK_SIZE=${DB_DISK_SIZE:-50}
  read -rp "Artifact Registry repo [cpqai]: "    REPO_NAME;   REPO_NAME=${REPO_NAME:-cpqai}
  read -rp "VPC network [default]: "             VPC_NETWORK; VPC_NETWORK=${VPC_NETWORK:-default}
  read -rp "VPC subnet [default]: "              VPC_SUBNET;  VPC_SUBNET=${VPC_SUBNET:-default}
fi

SA_NAME="${SERVICE_NAME}-sa"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
CLOUDBUILD_SA="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"

info "Project:    $PROJECT_ID (#$PROJECT_NUMBER)"
info "Region:     $REGION"
info "Zone:       $ZONE"
info "Service:    $SERVICE_NAME"
info "DB VM:      $DB_VM_NAME ($DB_VM_TYPE, ${DB_DISK_SIZE}GB SSD)"
info "VPC:        $VPC_NETWORK / $VPC_SUBNET"
info "SA:         $SA_EMAIL"
echo ""

# ── 1. Enable APIs ──────────────────────────────────────────────────
info "Enabling required GCP APIs..."
gcloud services enable \
  run.googleapis.com \
  compute.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  aiplatform.googleapis.com \
  cloudbuild.googleapis.com \
  --project="$PROJECT_ID"
ok "APIs enabled"

# ── 2. Artifact Registry ────────────────────────────────────────────
info "Creating Artifact Registry repository..."
if gcloud artifacts repositories describe "$REPO_NAME" \
    --location="$REGION" --project="$PROJECT_ID" &>/dev/null; then
  ok "Repository '$REPO_NAME' already exists"
else
  gcloud artifacts repositories create "$REPO_NAME" \
    --repository-format=docker \
    --location="$REGION" \
    --description="CPQAI container images" \
    --project="$PROJECT_ID"
  ok "Repository '$REPO_NAME' created"
fi

# ── 3. Service Account + IAM ────────────────────────────────────────
info "Creating service account..."
if gcloud iam service-accounts describe "$SA_EMAIL" --project="$PROJECT_ID" &>/dev/null; then
  ok "Service account '$SA_NAME' already exists"
else
  gcloud iam service-accounts create "$SA_NAME" \
    --display-name="CPQAI Cloud Run SA" \
    --project="$PROJECT_ID"
  ok "Service account created"
fi

# Cloud Run SA roles
ROLES=(
  "roles/aiplatform.user"             # Vertex AI inference
  "roles/secretmanager.secretAccessor" # Read secrets
  "roles/logging.logWriter"           # Cloud Logging
)
for role in "${ROLES[@]}"; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="$role" \
    --condition=None \
    --quiet &>/dev/null
done
ok "IAM roles bound for Cloud Run SA"

# Cloud Build SA needs permission to deploy to Cloud Run and impersonate the service account
info "Granting Cloud Build deploy permissions..."
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${CLOUDBUILD_SA}" \
  --role="roles/run.admin" \
  --condition=None \
  --quiet &>/dev/null
gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
  --member="serviceAccount:${CLOUDBUILD_SA}" \
  --role="roles/iam.serviceAccountUser" \
  --project="$PROJECT_ID" \
  --quiet &>/dev/null
ok "Cloud Build deploy permissions granted"

# ── 4. Generate DB password ─────────────────────────────────────────
DB_PASSWORD=$(openssl rand -base64 24 | tr -d '/+=' | head -c 32)

# ── 5. Firewall rule for PostgreSQL ─────────────────────────────────
FIREWALL_RULE="${DB_VM_NAME}-allow-pg"
info "Creating firewall rule for PostgreSQL..."
if gcloud compute firewall-rules describe "$FIREWALL_RULE" --project="$PROJECT_ID" &>/dev/null; then
  ok "Firewall rule '$FIREWALL_RULE' already exists"
else
  gcloud compute firewall-rules create "$FIREWALL_RULE" \
    --network="$VPC_NETWORK" \
    --direction=INGRESS \
    --action=ALLOW \
    --rules=tcp:5432 \
    --source-ranges=10.0.0.0/8 \
    --target-tags="${DB_VM_NAME}" \
    --description="Allow PostgreSQL from VPC-internal IPs (Cloud Run Direct VPC egress)" \
    --project="$PROJECT_ID"
  ok "Firewall rule created"
fi

# ── 6. PostgreSQL VM (Container-Optimized OS) ───────────────────────
# Note: gzdaniel/postgres-for-rag:16.6 does NOT use the standard postgres
# entrypoint — it ignores POSTGRES_DB/POSTGRES_USER/POSTGRES_PASSWORD env
# vars. We create the user and database after the container starts.
info "Creating PostgreSQL VM..."
DB_VM_CREATED=false
if gcloud compute instances describe "$DB_VM_NAME" --zone="$ZONE" --project="$PROJECT_ID" &>/dev/null; then
  ok "VM '$DB_VM_NAME' already exists"
else
  # VM needs a public IP to pull the container image from Docker Hub.
  # The firewall rule restricts PostgreSQL access to VPC-internal IPs only.
  gcloud compute instances create-with-container "$DB_VM_NAME" \
    --zone="$ZONE" \
    --machine-type="$DB_VM_TYPE" \
    --network="$VPC_NETWORK" \
    --subnet="$VPC_SUBNET" \
    --tags="${DB_VM_NAME}" \
    --image-family=cos-stable \
    --image-project=cos-cloud \
    --boot-disk-size=10GB \
    --create-disk="name=${DB_VM_NAME}-data,size=${DB_DISK_SIZE}GB,type=pd-ssd,auto-delete=no,mode=rw" \
    --container-image=gzdaniel/postgres-for-rag:16.6 \
    --container-mount-disk="mount-path=/var/lib/postgresql/data,name=${DB_VM_NAME}-data,mode=rw" \
    --container-restart-policy=always \
    --project="$PROJECT_ID"
  DB_VM_CREATED=true
  ok "VM created"
fi

# Get the VM's internal IP
DB_INTERNAL_IP=$(gcloud compute instances describe "$DB_VM_NAME" \
  --zone="$ZONE" --project="$PROJECT_ID" \
  --format='value(networkInterfaces[0].networkIP)')
ok "DB VM internal IP: $DB_INTERNAL_IP"

# Wait for PostgreSQL to be ready and create the user/database
if [[ "$DB_VM_CREATED" == "true" ]]; then
  info "Waiting for PostgreSQL to start (pulling image + initializing)..."
  for i in $(seq 1 30); do
    if gcloud compute ssh "$DB_VM_NAME" --zone="$ZONE" --project="$PROJECT_ID" \
      --command="docker exec \$(docker ps -q 2>/dev/null) pg_isready 2>/dev/null" \
      --ssh-flag="-o" --ssh-flag="StrictHostKeyChecking=no" \
      --ssh-flag="-o" --ssh-flag="ConnectTimeout=5" &>/dev/null; then
      ok "PostgreSQL is ready"
      break
    fi
    if [[ $i -eq 30 ]]; then
      err "PostgreSQL did not start within 150s. Check: gcloud compute ssh $DB_VM_NAME --zone=$ZONE"
      exit 1
    fi
    sleep 5
  done

  info "Creating database user '$DB_USER' and database '$DB_NAME'..."
  gcloud compute ssh "$DB_VM_NAME" --zone="$ZONE" --project="$PROJECT_ID" \
    --ssh-flag="-o" --ssh-flag="StrictHostKeyChecking=no" \
    --command="docker exec \$(docker ps -q) su - postgres -c \"psql -c \\\"CREATE USER ${DB_USER} WITH SUPERUSER CREATEDB CREATEROLE PASSWORD '${DB_PASSWORD}';\\\"\" && \
              docker exec \$(docker ps -q) su - postgres -c \"psql -c \\\"CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};\\\"\"" \
    &>/dev/null
  ok "Database user and database created"
fi

# ── 7. Secret Manager ───────────────────────────────────────────────
store_secret() {
  local name="$1" value="$2"
  if gcloud secrets describe "$name" --project="$PROJECT_ID" &>/dev/null; then
    printf '%s' "$value" | gcloud secrets versions add "$name" \
      --data-file=- --project="$PROJECT_ID" &>/dev/null
    ok "Secret '$name' updated"
  else
    printf '%s' "$value" | gcloud secrets create "$name" \
      --data-file=- --replication-policy=automatic \
      --project="$PROJECT_ID" &>/dev/null
    ok "Secret '$name' created"
  fi
}

info "Storing secrets..."
store_secret "cpqai-db-password" "$DB_PASSWORD"

API_KEY=$(openssl rand -base64 32 | tr -d '/+=' | head -c 40)
store_secret "cpqai-api-key" "$API_KEY"

# ── 8. Build and deploy ─────────────────────────────────────────────
BUILD_TAG=$(date +%Y%m%d-%H%M%S)
info "Building container image with Cloud Build (tag: $BUILD_TAG)..."
gcloud builds submit \
  --config=deploy/gcp/cloudbuild.yaml \
  --substitutions="_REGION=${REGION},_SERVICE_NAME=${SERVICE_NAME},_REPO_NAME=${REPO_NAME},_TAG=${BUILD_TAG},_DB_HOST=${DB_INTERNAL_IP},_DB_NAME=${DB_NAME},_DB_USER=${DB_USER},_VPC_NETWORK=${VPC_NETWORK},_VPC_SUBNET=${VPC_SUBNET}" \
  --project="$PROJECT_ID" \
  .

ok "Build and deploy complete"

# ── 9. Print summary ────────────────────────────────────────────────
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
  --region="$REGION" --project="$PROJECT_ID" \
  --format='value(status.url)' 2>/dev/null || echo "(pending)")

echo ""
echo "=============================================="
echo "  CPQAI deployed on Google Cloud"
echo "=============================================="
echo ""
echo "  Service URL:    $SERVICE_URL"
echo "  DB VM:          $DB_VM_NAME ($DB_INTERNAL_IP:5432)"
echo "  Database:       $DB_NAME"
echo "  DB User:        $DB_USER"
echo "  API Key:        $API_KEY"
echo ""
echo "  Service Account: $SA_EMAIL"
echo "  Image:          ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}:latest"
echo ""
echo "  Test:"
echo "    curl -H 'X-API-Key: ${API_KEY}' ${SERVICE_URL}/health"
echo ""
echo "  Secrets stored in Secret Manager:"
echo "    - cpqai-db-password"
echo "    - cpqai-api-key"
echo ""
echo "  DB VM management:"
echo "    gcloud compute instances start  $DB_VM_NAME --zone=$ZONE"
echo "    gcloud compute instances stop   $DB_VM_NAME --zone=$ZONE"
echo "    gcloud compute ssh $DB_VM_NAME --zone=$ZONE  # then: docker logs \$(docker ps -q)"
echo ""
warn "Save the API key above — it is also in Secret Manager (cpqai-api-key)"
echo ""
