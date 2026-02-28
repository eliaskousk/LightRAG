# CPQAI on Amazon Web Services

Deploy CPQAI as a managed App Runner service backed by a self-managed PostgreSQL EC2 instance (with Apache AGE + pgvector) and Amazon Bedrock (Claude Sonnet 4.6 + Titan Embed V2).

## Architecture

```
Internet  ──>  App Runner (CPQAI container, HTTPS)
                   |
              VPC Connector (App Runner → default VPC)
                   |
                   ├── EC2 instance
                   │   gzdaniel/postgres-for-rag:16.6
                   │   PostgreSQL 16 + AGE + pgvector
                   │        ├── KV Storage
                   │        ├── Vector Storage (pgvector HNSW)
                   │        ├── Graph Storage  (Apache AGE)
                   │        └── Doc Status Storage
                   │   [EBS gp3 volume attached]
                   │
                   └── VPC Endpoint (bedrock-runtime)
                        └── Bedrock (Claude Sonnet 4.6 + Titan Embed V2)

ECR             ── container images
Secrets Manager ── DB password, API key
CodeBuild       ── CI/CD pipeline
```

> **Note:** App Runner with a VPC Connector loses direct internet access.
> A VPC endpoint for `bedrock-runtime` is required so the container can
> reach Bedrock for LLM and embedding calls.

## Why EC2 instead of RDS?

LightRAG's `PGGraphStorage` uses the **Apache AGE** extension for graph operations (Cypher queries, vertex/edge storage). RDS does not support AGE — it only allows a curated set of extensions.

A self-managed PostgreSQL container (`gzdaniel/postgres-for-rag:16.6`) ships with AGE, pgvector, and other RAG-relevant extensions pre-installed, at a fraction of the cost:

| Option | Monthly cost |
|---|---|
| RDS `db.t3.medium` (Multi-AZ) | ~$130 |
| EC2 `t3.small` + 50GB gp3 | **~$20** |

## Prerequisites

1. **AWS CLI v2** configured (`aws configure` or SSO)
2. **IAM permissions**: Admin or equivalent, including `aws-marketplace:ViewSubscriptions` and `aws-marketplace:Subscribe`
3. **Bedrock model access**: Enable the following models in the [Bedrock console](https://console.aws.amazon.com/bedrock/home#/modelaccess):
   - **Anthropic → Claude Sonnet 4.6** (requires filling out the Anthropic use case form)
   - **Amazon → Titan Text Embeddings V2**

## Quick start

### Automated setup

```bash
# Interactive (prompts for region, names, instance type, etc.)
./deploy/aws/setup.sh

# Or accept all defaults (us-east-1, t3.small, 50GB gp3)
./deploy/aws/setup.sh --defaults
```

This creates:
- ECR repository
- Security group allowing PostgreSQL traffic within VPC
- EC2 instance running `gzdaniel/postgres-for-rag:16.6` with EBS gp3 volume
- Database password and API key in Secrets Manager
- IAM roles for App Runner (Bedrock, Secrets, ECR access)
- Docker image built and pushed to ECR (via CodeBuild)
- Bedrock VPC endpoint (private link for LLM + embedding calls)
- App Runner VPC Connector for private DB access
- App Runner service with Bedrock + PostgreSQL configuration
- CodeBuild project for CI/CD

### Manual / step-by-step

#### 1. Create ECR repository

```bash
aws ecr create-repository \
  --repository-name cpqai \
  --region us-east-1 \
  --image-scanning-configuration scanOnPush=true
```

#### 2. Create the PostgreSQL EC2 instance

```bash
# Generate a password
DB_PASSWORD=$(openssl rand -base64 24 | tr -d '/+=' | head -c 32)

# Get default VPC ID
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" \
  --query 'Vpcs[0].VpcId' --output text)

# Create security group
SG_ID=$(aws ec2 create-security-group \
  --group-name cpqai-db-sg \
  --description "CPQAI PostgreSQL" \
  --vpc-id "$VPC_ID" \
  --query 'GroupId' --output text)

aws ec2 authorize-security-group-ingress \
  --group-id "$SG_ID" --protocol tcp --port 5432 --cidr 172.31.0.0/16

# Launch instance with user-data (installs Docker + runs postgres-for-rag)
aws ec2 run-instances \
  --image-id $(aws ec2 describe-images --owners amazon \
    --filters "Name=name,Values=al2023-ami-2023*-x86_64" "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' --output text) \
  --instance-type t3.small \
  --security-group-ids "$SG_ID" \
  --associate-public-ip-address \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=cpqai-db}]' \
  --user-data file://deploy/aws/user-data.sh

# Get private IP
DB_IP=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=cpqai-db" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].PrivateIpAddress' --output text)
echo "DB IP: $DB_IP"
```

#### 3. Store secrets

```bash
aws secretsmanager create-secret --name cpqai-db-password --secret-string "$DB_PASSWORD"
aws secretsmanager create-secret --name cpqai-api-key --secret-string "YOUR_API_KEY"
```

#### 4. Create IAM roles

See `setup.sh` for the full IAM role definitions. The key roles are:
- **App Runner instance role**: Bedrock invoke, Secrets Manager read, CloudWatch logs
- **App Runner ECR role**: Pull images from ECR
- **CodeBuild role**: ECR push, App Runner deployment trigger

#### 5. Build and deploy

```bash
# Build locally
docker build -f Dockerfile.apprunner -t cpqai .

# Push to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
docker tag cpqai:latest $ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/cpqai:latest
docker push $ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/cpqai:latest

# Create App Runner service (via CLI or console)
aws apprunner create-service --cli-input-json file://apprunner-config.json
```

## Files

| File | Purpose |
|------|---------|
| `Dockerfile.apprunner` | Multi-stage build optimized for App Runner (port 8080, PG storage defaults, Bedrock) |
| `deploy/aws/buildspec.yml` | CodeBuild pipeline: build, push to ECR, trigger App Runner deployment |
| `deploy/aws/setup.sh` | One-command infrastructure provisioning script |
| `deploy/aws/cleanup.sh` | One-command resource teardown script |
| `deploy/aws/.env.aws.example` | Reference environment variable configuration |

## Configuration

### Bedrock authentication

App Runner uses the instance role — no API keys needed. The instance role gets `bedrock:InvokeModel*` permission for both foundation models and cross-region inference profiles (the `us.` model prefix routes requests across US regions for higher availability).

A **VPC endpoint** for `bedrock-runtime` provides private connectivity from the VPC to Bedrock, since App Runner with a VPC Connector has no direct internet access.

Key environment variables:
- `LLM_BINDING=aws_bedrock` — selects the Bedrock LLM provider
- `LLM_MODEL=us.anthropic.claude-sonnet-4-6` — Claude Sonnet 4.6 via cross-region inference
- `EMBEDDING_BINDING=aws_bedrock` — selects the Bedrock embedding provider
- `EMBEDDING_MODEL=amazon.titan-embed-text-v2:0` — Titan Embed V2 (1024 dimensions)
- `AWS_REGION=us-east-1` — Bedrock region

### PostgreSQL connection

App Runner connects to the PostgreSQL EC2 instance via a **VPC Connector** — the App Runner service routes traffic through the VPC to reach the instance's private IP.

```
POSTGRES_HOST=10.0.0.100    # Instance's private IP
POSTGRES_PORT=5432
```

A security group restricts port 5432 to VPC-internal source ranges.

### Secrets

Passwords and API keys are stored in **Secrets Manager** and injected as environment variables at runtime via App Runner's native secrets integration.

| Secret name | Injected as |
|---|---|
| `cpqai-db-password` | `POSTGRES_PASSWORD` |
| `cpqai-api-key` | `LIGHTRAG_API_KEY` |

### Scaling

Default App Runner configuration:

| Setting | Value | Notes |
|---|---|---|
| Min instances | 1 | App Runner requires at least 1 |
| Max instances | 10 | Adjust based on load |
| CPU | 2 vCPU | Minimum for concurrent LLM calls |
| Memory | 4 GB | Handles graph operations and large contexts |
| Health check | `/health` HTTP | 10s interval, 5s timeout |

## DB instance management

```bash
# Connect via EC2 Instance Connect (no pre-configured SSH key needed)
aws ec2-instance-connect ssh --instance-id i-0123456789abcdef0 --region us-east-1

# View PostgreSQL logs
sudo docker logs cpqai-postgres

# Connect to PostgreSQL from inside the instance
sudo docker exec -it cpqai-postgres psql -U cpqai -d cpqai

# Stop / start the instance (data persists on EBS)
aws ec2 stop-instances  --instance-ids i-0123456789abcdef0 --region us-east-1
aws ec2 start-instances --instance-ids i-0123456789abcdef0 --region us-east-1

# Resize the instance (stop first)
aws ec2 modify-instance-attribute \
  --instance-id i-0123456789abcdef0 \
  --instance-type '{"Value": "t3.medium"}'
```

### Backups

The PostgreSQL data lives on a separate EBS gp3 volume (`cpqai-db-data`). For backups:

```bash
# Create a snapshot
VOLUME_ID=$(aws ec2 describe-volumes \
  --filters "Name=tag:Name,Values=cpqai-db-data" \
  --query 'Volumes[0].VolumeId' --output text)

aws ec2 create-snapshot \
  --volume-id "$VOLUME_ID" \
  --description "cpqai-db-backup-$(date +%Y%m%d)" \
  --tag-specifications "ResourceType=snapshot,Tags=[{Key=Name,Value=cpqai-db-backup-$(date +%Y%m%d)}]"

# For automated backups, use AWS Backup or Data Lifecycle Manager
aws dlm create-lifecycle-policy \
  --description "CPQAI DB daily snapshots" \
  --state ENABLED \
  --execution-role-arn arn:aws:iam::$ACCOUNT_ID:role/AWSDataLifecycleManagerDefaultRole \
  --policy-details '{
    "PolicyType": "EBS_SNAPSHOT_MANAGEMENT",
    "ResourceTypes": ["VOLUME"],
    "TargetTags": [{"Key": "Name", "Value": "cpqai-db-data"}],
    "Schedules": [{
      "Name": "Daily",
      "CreateRule": {"Interval": 24, "IntervalUnit": "HOURS", "Times": ["03:00"]},
      "RetainRule": {"Count": 14}
    }]
  }'
```

## Monitoring

```bash
# App Runner logs (via CloudWatch)
aws logs tail /aws/apprunner/cpqai --follow --region us-east-1

# App Runner service status
aws apprunner describe-service --service-arn $SERVICE_ARN --region us-east-1

# EC2 instance status
aws ec2 describe-instance-status --instance-ids $INSTANCE_ID --region us-east-1
```

## Cost estimates

| Component | Approximate monthly cost |
|---|---|
| EC2 `t3.small` + 50GB gp3 | ~$20 |
| App Runner (min 1 instance, moderate traffic) | ~$15–60 |
| Bedrock (Claude Sonnet 4.6 + Titan Embed V2, moderate) | ~$10–50 |
| ECR | < $1 |
| Secrets Manager | < $1 |
| **Total** | **~$45–130** |

Costs vary with document ingestion volume and query frequency. The EC2 instance runs 24/7 — stop it when not in use to save costs.

## Cleanup

```bash
# Interactive (confirms each resource)
./deploy/aws/cleanup.sh

# Skip confirmations
./deploy/aws/cleanup.sh --yes

# Use defaults + skip confirmations
./deploy/aws/cleanup.sh --defaults
```
