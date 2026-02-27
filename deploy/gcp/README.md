# CPQAI on Google Cloud Platform

Deploy CPQAI as a serverless Cloud Run service backed by a self-managed PostgreSQL VM (with Apache AGE + pgvector) and Vertex AI (Gemini).

## Architecture

```
                    Cloud Armor (optional WAF)
                            |
Internet  ──>  Cloud Run (CPQAI container)
                   |              |
                   |              └── Vertex AI (Gemini LLM + Embeddings)
                   |
              Direct VPC egress (no connector needed)
                   |
              Compute Engine VM (internal IP only)
              gzdaniel/postgres-for-rag:16.6
              PostgreSQL 16 + AGE + pgvector
                   ├── KV Storage
                   ├── Vector Storage (pgvector HNSW)
                   ├── Graph Storage  (Apache AGE)
                   └── Doc Status Storage
              [Persistent SSD attached]

Artifact Registry ── container images
Secret Manager    ── DB password, API key
```

## Why a VM instead of Cloud SQL?

LightRAG's `PGGraphStorage` uses the **Apache AGE** extension for graph operations (Cypher queries, vertex/edge storage). Cloud SQL does not support AGE — it only allows a curated set of extensions.

A self-managed PostgreSQL container (`gzdaniel/postgres-for-rag:16.6`) ships with AGE, pgvector, and other RAG-relevant extensions pre-installed, at a fraction of the cost:

| Option | Monthly cost |
|---|---|
| Cloud SQL `db-custom-2-8192` (HA) | ~$130 |
| Compute Engine `e2-medium` + 50GB SSD | **~$25** |

## Quick start

### Automated setup

```bash
# Interactive (prompts for region, names, VM size, etc.)
./deploy/gcp/setup.sh

# Or accept all defaults (us-central1, e2-medium, 50GB SSD)
./deploy/gcp/setup.sh --defaults
```

This creates:
- Artifact Registry repository
- Service account with least-privilege IAM roles
- Firewall rule allowing PostgreSQL traffic within VPC
- Compute Engine VM running `gzdaniel/postgres-for-rag:16.6` (no public IP)
- Persistent SSD disk for PostgreSQL data (survives VM deletion)
- Database password and API key in Secret Manager
- Builds and deploys CPQAI to Cloud Run with Direct VPC egress

### Manual / step-by-step

#### 1. Enable APIs

```bash
gcloud services enable \
  run.googleapis.com \
  compute.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  aiplatform.googleapis.com \
  cloudbuild.googleapis.com
```

#### 2. Create Artifact Registry

```bash
gcloud artifacts repositories create cpqai \
  --repository-format=docker \
  --location=us-central1
```

#### 3. Create the PostgreSQL VM

```bash
# Generate a password
DB_PASSWORD=$(openssl rand -base64 24 | tr -d '/+=' | head -c 32)

# Create firewall rule
gcloud compute firewall-rules create cpqai-db-allow-pg \
  --network=default \
  --direction=INGRESS \
  --action=ALLOW \
  --rules=tcp:5432 \
  --source-ranges=10.0.0.0/8 \
  --target-tags=cpqai-db

# Create VM with container and persistent SSD
gcloud compute instances create-with-container cpqai-db \
  --zone=us-central1-a \
  --machine-type=e2-medium \
  --network=default \
  --subnet=default \
  --no-address \
  --tags=cpqai-db \
  --image-family=cos-stable \
  --image-project=cos-cloud \
  --boot-disk-size=10GB \
  --create-disk=name=cpqai-db-data,size=50GB,type=pd-ssd,auto-delete=no,mode=rw \
  --container-image=gzdaniel/postgres-for-rag:16.6 \
  --container-mount-disk=mount-path=/var/lib/postgresql/data,name=cpqai-db-data,mode=rw \
  --container-env="POSTGRES_DB=cpqai,POSTGRES_USER=cpqai,POSTGRES_PASSWORD=${DB_PASSWORD}" \
  --container-restart-policy=always

# Get the VM's internal IP
DB_IP=$(gcloud compute instances describe cpqai-db \
  --zone=us-central1-a \
  --format='value(networkInterfaces[0].networkIP)')
echo "DB IP: $DB_IP"
```

#### 4. Store secrets

```bash
echo -n "$DB_PASSWORD" | gcloud secrets create cpqai-db-password --data-file=-
echo -n "YOUR_API_KEY"  | gcloud secrets create cpqai-api-key --data-file=-
```

#### 5. Create service account

```bash
gcloud iam service-accounts create cpqai-sa \
  --display-name="CPQAI Cloud Run SA"

for role in roles/aiplatform.user \
            roles/secretmanager.secretAccessor roles/logging.logWriter; do
  gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:cpqai-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="$role" --condition=None --quiet
done
```

#### 6. Build and deploy

```bash
# Build with Cloud Build
gcloud builds submit --config deploy/gcp/cloudbuild.yaml \
  --substitutions="_DB_HOST=${DB_IP}" .

# Or build locally and push
docker build -f Dockerfile.cloudrun \
  -t us-central1-docker.pkg.dev/$PROJECT_ID/cpqai/cpqai:latest .
docker push us-central1-docker.pkg.dev/$PROJECT_ID/cpqai/cpqai:latest

# Deploy with the service YAML (edit placeholders first)
gcloud run services replace deploy/gcp/cloudrun-service.yaml --region=us-central1
```

## Files

| File | Purpose |
|------|---------|
| `Dockerfile.cloudrun` | Multi-stage build optimized for Cloud Run (port 8080, PG storage defaults, Vertex AI) |
| `deploy/gcp/cloudbuild.yaml` | Cloud Build pipeline: build, push to Artifact Registry, deploy to Cloud Run with Direct VPC egress |
| `deploy/gcp/cloudrun-service.yaml` | Declarative Cloud Run service definition with health probes, secrets, and VPC networking |
| `deploy/gcp/setup.sh` | One-command infrastructure provisioning script |
| `deploy/gcp/.env.gcp.example` | Reference environment variable configuration |

## Configuration

### Vertex AI authentication

Cloud Run uses the service account identity — no API keys needed. The service account gets `roles/aiplatform.user` which grants access to Vertex AI models.

Key environment variables:
- `GOOGLE_GENAI_USE_VERTEXAI=true` — switches from Gemini API to Vertex AI
- `GOOGLE_CLOUD_PROJECT` — your GCP project ID
- `GOOGLE_CLOUD_LOCATION` — region (e.g., `us-central1`)
- `LLM_BINDING_HOST=DEFAULT_GEMINI_ENDPOINT` — auto-selects the regional endpoint

### PostgreSQL connection

Cloud Run connects to the PostgreSQL VM via **Direct VPC egress** — no VPC connector or Cloud SQL proxy needed. The VM has no public IP; it is only reachable from within the VPC.

```
POSTGRES_HOST=10.128.0.2    # VM's internal IP
POSTGRES_PORT=5432
```

A firewall rule restricts port 5432 to VPC-internal source ranges (`10.0.0.0/8`).

### Secrets

Passwords and API keys are stored in **Secret Manager** and injected as environment variables at runtime.

| Secret name | Injected as |
|---|---|
| `cpqai-db-password` | `POSTGRES_PASSWORD` |
| `cpqai-api-key` | `LIGHTRAG_API_KEY` |

### Scaling

Default configuration in `cloudrun-service.yaml`:

| Setting | Value | Notes |
|---|---|---|
| Min instances | 0 | Scale to zero when idle |
| Max instances | 10 | Adjust based on load |
| CPU | 2 | Minimum for concurrent LLM calls |
| Memory | 4 Gi | Handles graph operations and large contexts |
| Concurrency | 80 | Requests per instance |
| Timeout | 300s | Allows for slow LLM responses during indexing |
| Startup CPU boost | enabled | Faster cold starts |
| CPU throttling | disabled | Keeps DB connections alive between requests |

### Cloud Armor (optional)

To add WAF protection, create a Cloud Armor security policy and attach it:

```bash
# Create policy
gcloud compute security-policies create cpqai-policy \
  --description="CPQAI WAF"

# Add rate limiting rule
gcloud compute security-policies rules create 1000 \
  --security-policy=cpqai-policy \
  --action=throttle \
  --rate-limit-threshold-count=100 \
  --rate-limit-threshold-interval-sec=60 \
  --conform-action=allow \
  --exceed-action=deny-429 \
  --enforce-on-key=IP

# Attach to Cloud Run via a backend service (requires a load balancer)
# See: https://cloud.google.com/run/docs/securing/cloud-armor
```

## DB VM management

```bash
# SSH into the VM (Container-Optimized OS)
gcloud compute ssh cpqai-db --zone=us-central1-a

# View PostgreSQL logs
docker logs $(docker ps -q)

# Connect to PostgreSQL from inside the VM
docker exec -it $(docker ps -q) psql -U cpqai -d cpqai

# Stop / start the VM (data persists on the SSD)
gcloud compute instances stop  cpqai-db --zone=us-central1-a
gcloud compute instances start cpqai-db --zone=us-central1-a

# Resize the VM (stop first)
gcloud compute instances set-machine-type cpqai-db \
  --zone=us-central1-a --machine-type=e2-standard-2
```

### Backups

The PostgreSQL data lives on a separate persistent SSD (`cpqai-db-data`) with `auto-delete=no` — it survives VM deletion. For scheduled backups:

```bash
# Create a snapshot
gcloud compute disks snapshot cpqai-db-data \
  --zone=us-central1-a \
  --snapshot-names=cpqai-db-backup-$(date +%Y%m%d)

# Or use a snapshot schedule for automated daily backups
gcloud compute resource-policies create snapshot-schedule cpqai-db-daily \
  --region=us-central1 \
  --max-retention-days=14 \
  --daily-schedule \
  --start-time=03:00

gcloud compute disks add-resource-policies cpqai-db-data \
  --zone=us-central1-a \
  --resource-policies=cpqai-db-daily
```

## Monitoring

```bash
# Cloud Run logs
gcloud run services logs read cpqai --region=us-central1 --limit=50
gcloud run services logs tail cpqai --region=us-central1

# DB VM serial port output (boot logs)
gcloud compute instances get-serial-port-output cpqai-db --zone=us-central1-a
```

## Cost estimates

| Component | Approximate monthly cost |
|---|---|
| Compute Engine `e2-medium` + 50GB SSD | ~$25 |
| Cloud Run (scale-to-zero, moderate traffic) | ~$5–50 |
| Vertex AI (Gemini Flash, moderate usage) | ~$5–30 |
| Artifact Registry | < $1 |
| Secret Manager | < $1 |
| **Total** | **~$35–105** |

Costs vary with document ingestion volume and query frequency. The DB VM runs 24/7 — stop it when not in use to save costs, or use a [VM schedule](https://cloud.google.com/compute/docs/instances/schedule-instance-start-stop) to auto-stop during off-hours.
