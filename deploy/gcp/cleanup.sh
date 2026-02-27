#!/usr/bin/env bash
#
# CPQAI — Google Cloud resource cleanup
#
# Deletes all resources created by setup.sh:
#   Cloud Run service, Compute Engine VM + data disk,
#   Artifact Registry images, Secret Manager secrets,
#   firewall rule, service account, and IAM bindings.
#
# Usage:
#   ./deploy/gcp/cleanup.sh              # interactive (confirms each step)
#   ./deploy/gcp/cleanup.sh --yes        # skip confirmations
#   ./deploy/gcp/cleanup.sh --defaults   # use default names + skip confirmations

set -euo pipefail

# ── Colour helpers ──────────────────────────────────────────────────
info()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
ok()    { printf '\033[1;32m[OK]\033[0m    %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
err()   { printf '\033[1;31m[ERR]\033[0m   %s\n' "$*" >&2; }
skip()  { printf '\033[1;90m[SKIP]\033[0m  %s\n' "$*"; }

# ── Configuration ───────────────────────────────────────────────────
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [[ -z "$PROJECT_ID" ]]; then
  err "No active GCP project. Run: gcloud config set project YOUR_PROJECT_ID"
  exit 1
fi

AUTO_YES=false

if [[ "${1:-}" == "--defaults" ]]; then
  AUTO_YES=true
  REGION="us-central1"
  ZONE="${REGION}-a"
  SERVICE_NAME="cpqai"
  DB_VM_NAME="cpqai-db"
  REPO_NAME="cpqai"
elif [[ "${1:-}" == "--yes" ]]; then
  AUTO_YES=true
  read -rp "Region [us-central1]: "              REGION;       REGION=${REGION:-us-central1}
  read -rp "Zone [${REGION:-us-central1}-a]: "   ZONE;         ZONE=${ZONE:-${REGION:-us-central1}-a}
  read -rp "Cloud Run service [cpqai]: "         SERVICE_NAME; SERVICE_NAME=${SERVICE_NAME:-cpqai}
  read -rp "DB VM name [cpqai-db]: "             DB_VM_NAME;   DB_VM_NAME=${DB_VM_NAME:-cpqai-db}
  read -rp "Artifact Registry repo [cpqai]: "    REPO_NAME;    REPO_NAME=${REPO_NAME:-cpqai}
else
  read -rp "Region [us-central1]: "              REGION;       REGION=${REGION:-us-central1}
  read -rp "Zone [${REGION:-us-central1}-a]: "   ZONE;         ZONE=${ZONE:-${REGION:-us-central1}-a}
  read -rp "Cloud Run service [cpqai]: "         SERVICE_NAME; SERVICE_NAME=${SERVICE_NAME:-cpqai}
  read -rp "DB VM name [cpqai-db]: "             DB_VM_NAME;   DB_VM_NAME=${DB_VM_NAME:-cpqai-db}
  read -rp "Artifact Registry repo [cpqai]: "    REPO_NAME;    REPO_NAME=${REPO_NAME:-cpqai}
fi

SA_NAME="${SERVICE_NAME}-sa"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
FIREWALL_RULE="${DB_VM_NAME}-allow-pg"

echo ""
warn "This will permanently delete the following resources in project '$PROJECT_ID':"
echo ""
echo "  - Cloud Run service:     $SERVICE_NAME ($REGION)"
echo "  - Compute Engine VM:     $DB_VM_NAME ($ZONE)"
echo "  - Persistent disk:       ${DB_VM_NAME}-data ($ZONE)"
echo "  - Artifact Registry:     $REPO_NAME ($REGION)"
echo "  - Secrets:               cpqai-db-password, cpqai-api-key"
echo "  - Firewall rule:         $FIREWALL_RULE"
echo "  - Service account:       $SA_EMAIL"
echo ""

if [[ "$AUTO_YES" != "true" ]]; then
  read -rp "Are you sure? Type 'delete' to confirm: " CONFIRM
  if [[ "$CONFIRM" != "delete" ]]; then
    err "Aborted."
    exit 1
  fi
fi

# ── Helper to delete a resource safely ──────────────────────────────
try_delete() {
  local description="$1"
  shift
  info "Deleting $description..."
  if "$@" 2>/dev/null; then
    ok "$description deleted"
  else
    skip "$description not found or already deleted"
  fi
}

# ── 1. Cloud Run service ────────────────────────────────────────────
try_delete "Cloud Run service '$SERVICE_NAME'" \
  gcloud run services delete "$SERVICE_NAME" \
    --region="$REGION" --project="$PROJECT_ID" --quiet

# ── 2. Compute Engine VM ────────────────────────────────────────────
try_delete "VM '$DB_VM_NAME'" \
  gcloud compute instances delete "$DB_VM_NAME" \
    --zone="$ZONE" --project="$PROJECT_ID" --quiet

# ── 3. Persistent data disk ─────────────────────────────────────────
# The disk was created with auto-delete=no, so it survives VM deletion
try_delete "persistent disk '${DB_VM_NAME}-data'" \
  gcloud compute disks delete "${DB_VM_NAME}-data" \
    --zone="$ZONE" --project="$PROJECT_ID" --quiet

# ── 4. Firewall rule ────────────────────────────────────────────────
try_delete "firewall rule '$FIREWALL_RULE'" \
  gcloud compute firewall-rules delete "$FIREWALL_RULE" \
    --project="$PROJECT_ID" --quiet

# ── 5. Secret Manager secrets ───────────────────────────────────────
try_delete "secret 'cpqai-db-password'" \
  gcloud secrets delete cpqai-db-password \
    --project="$PROJECT_ID" --quiet

try_delete "secret 'cpqai-api-key'" \
  gcloud secrets delete cpqai-api-key \
    --project="$PROJECT_ID" --quiet

# ── 6. Artifact Registry repository ─────────────────────────────────
try_delete "Artifact Registry repo '$REPO_NAME'" \
  gcloud artifacts repositories delete "$REPO_NAME" \
    --location="$REGION" --project="$PROJECT_ID" --quiet

# ── 7. Service account ──────────────────────────────────────────────
# Remove IAM bindings first, then delete the SA
info "Removing IAM bindings..."
for role in "roles/aiplatform.user" "roles/secretmanager.secretAccessor" "roles/logging.logWriter"; do
  gcloud projects remove-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="$role" \
    --quiet &>/dev/null || true
done

# Remove Cloud Build IAM bindings
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)' 2>/dev/null || echo "")
if [[ -n "$PROJECT_NUMBER" ]]; then
  CLOUDBUILD_SA="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"
  gcloud projects remove-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${CLOUDBUILD_SA}" \
    --role="roles/run.admin" \
    --quiet &>/dev/null || true
  gcloud iam service-accounts remove-iam-policy-binding "$SA_EMAIL" \
    --member="serviceAccount:${CLOUDBUILD_SA}" \
    --role="roles/iam.serviceAccountUser" \
    --project="$PROJECT_ID" \
    --quiet &>/dev/null || true
fi
ok "IAM bindings removed"

try_delete "service account '$SA_NAME'" \
  gcloud iam service-accounts delete "$SA_EMAIL" \
    --project="$PROJECT_ID" --quiet

# ── 8. Cloud Build source bucket (optional) ─────────────────────────
info "Note: Cloud Build source bucket (gs://${PROJECT_ID}_cloudbuild/) was not deleted."
info "Delete manually if desired: gsutil rm -r gs://${PROJECT_ID}_cloudbuild/"

# ── Done ─────────────────────────────────────────────────────────────
echo ""
ok "Cleanup complete. All CPQAI resources have been removed from project '$PROJECT_ID'."
echo ""
