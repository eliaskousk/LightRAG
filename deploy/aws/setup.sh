#!/usr/bin/env bash
#
# CPQAI — AWS infrastructure setup
#
# Creates:  ECR repository, EC2 instance (PostgreSQL + AGE + pgvector),
#           Secrets Manager secrets, IAM roles, VPC networking,
#           and deploys the first App Runner service.
#
# Prerequisites:
#   - AWS CLI v2 configured (`aws configure` or SSO)
#   - Sufficient IAM permissions (Admin or equivalent)
#   - Docker installed locally (for image build + push)
#
# Usage:
#   chmod +x deploy/aws/setup.sh
#   ./deploy/aws/setup.sh              # interactive prompts
#   ./deploy/aws/setup.sh --defaults   # use all defaults (us-east-1)

set -euo pipefail

# ── Colour helpers ──────────────────────────────────────────────────
info()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
ok()    { printf '\033[1;32m[OK]\033[0m    %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
err()   { printf '\033[1;31m[ERR]\033[0m   %s\n' "$*" >&2; }

# ── Prerequisites check ────────────────────────────────────────────
if ! command -v aws &>/dev/null; then
  err "AWS CLI not found. Install: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
  exit 1
fi

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null) || {
  err "AWS CLI not configured. Run: aws configure"
  exit 1
}

# ── Configuration ───────────────────────────────────────────────────
if [[ "${1:-}" == "--defaults" ]]; then
  REGION="us-east-1"
  SERVICE_NAME="cpqai"
  DB_INSTANCE_NAME="cpqai-db"
  DB_NAME="cpqai"
  DB_USER="cpqai"
  DB_INSTANCE_TYPE="t3.small"       # 2 vCPU, 2 GB RAM — ~$15/month
  DB_DISK_SIZE="50"                 # GB gp3 for PostgreSQL data
  ECR_REPO="cpqai"
else
  read -rp "Region [us-east-1]: "                    REGION;           REGION=${REGION:-us-east-1}
  read -rp "App Runner service [cpqai]: "            SERVICE_NAME;     SERVICE_NAME=${SERVICE_NAME:-cpqai}
  read -rp "EC2 instance name [cpqai-db]: "          DB_INSTANCE_NAME; DB_INSTANCE_NAME=${DB_INSTANCE_NAME:-cpqai-db}
  read -rp "Database name [cpqai]: "                 DB_NAME;          DB_NAME=${DB_NAME:-cpqai}
  read -rp "Database user [cpqai]: "                 DB_USER;          DB_USER=${DB_USER:-cpqai}
  read -rp "EC2 instance type [t3.small]: "          DB_INSTANCE_TYPE; DB_INSTANCE_TYPE=${DB_INSTANCE_TYPE:-t3.small}
  read -rp "EBS volume size GB [50]: "               DB_DISK_SIZE;     DB_DISK_SIZE=${DB_DISK_SIZE:-50}
  read -rp "ECR repository [cpqai]: "                ECR_REPO;         ECR_REPO=${ECR_REPO:-cpqai}
fi

ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}"
APPRUNNER_INSTANCE_ROLE="${SERVICE_NAME}-apprunner-instance-role"
APPRUNNER_ECR_ROLE="${SERVICE_NAME}-apprunner-ecr-role"
CODEBUILD_ROLE="${SERVICE_NAME}-codebuild-role"
SG_NAME="${DB_INSTANCE_NAME}-sg"

info "Account:    $AWS_ACCOUNT_ID"
info "Region:     $REGION"
info "Service:    $SERVICE_NAME"
info "DB Instance: $DB_INSTANCE_NAME ($DB_INSTANCE_TYPE, ${DB_DISK_SIZE}GB gp3)"
info "ECR:        $ECR_URI"
echo ""

# ── 1. ECR Repository ────────────────────────────────────────────────
info "Creating ECR repository..."
if aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$REGION" &>/dev/null; then
  ok "Repository '$ECR_REPO' already exists"
else
  aws ecr create-repository \
    --repository-name "$ECR_REPO" \
    --region "$REGION" \
    --image-scanning-configuration scanOnPush=true \
    --output text &>/dev/null
  ok "Repository '$ECR_REPO' created"
fi

# ── 2. Generate DB password ──────────────────────────────────────────
DB_PASSWORD=$(openssl rand -base64 24 | tr -d '/+=' | head -c 32)

# ── 3. Default VPC & subnets ─────────────────────────────────────────
info "Looking up default VPC..."
VPC_ID=$(aws ec2 describe-vpcs \
  --filters "Name=isDefault,Values=true" \
  --region "$REGION" \
  --query 'Vpcs[0].VpcId' --output text)

if [[ -z "$VPC_ID" || "$VPC_ID" == "None" ]]; then
  err "No default VPC found in $REGION. Create one: aws ec2 create-default-vpc --region $REGION"
  exit 1
fi

VPC_CIDR=$(aws ec2 describe-vpcs --vpc-ids "$VPC_ID" --region "$REGION" \
  --query 'Vpcs[0].CidrBlock' --output text)
ok "Default VPC: $VPC_ID ($VPC_CIDR)"

# Find AZs that support the requested instance type (not all AZs support all types)
info "Finding AZs that support $DB_INSTANCE_TYPE..."
SUPPORTED_AZS=$(aws ec2 describe-instance-type-offerings \
  --location-type availability-zone \
  --filters "Name=instance-type,Values=$DB_INSTANCE_TYPE" \
  --region "$REGION" \
  --query 'InstanceTypeOfferings[*].Location' --output text)

if [[ -z "$SUPPORTED_AZS" ]]; then
  err "Instance type $DB_INSTANCE_TYPE is not available in any AZ in $REGION"
  exit 1
fi

# Get all default subnets, then filter to supported AZs in bash
# (avoids fragile dynamic JMESPath query construction)
ALL_SUBNETS=$(aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=$VPC_ID" "Name=default-for-az,Values=true" \
  --region "$REGION" \
  --query 'Subnets[*].[SubnetId,AvailabilityZone]' --output text)

SUBNET_ARRAY=()
while IFS=$'\t' read -r subnet_id az; do
  for supported_az in $SUPPORTED_AZS; do
    if [[ "$az" == "$supported_az" ]]; then
      SUBNET_ARRAY+=("$subnet_id")
      break
    fi
  done
done <<< "$ALL_SUBNETS"

if [[ ${#SUBNET_ARRAY[@]} -lt 2 ]]; then
  err "Need at least 2 subnets in AZs that support $DB_INSTANCE_TYPE. Found: ${#SUBNET_ARRAY[@]}"
  err "Supported AZs: $SUPPORTED_AZS"
  exit 1
fi

# Use first subnet for EC2, first two for VPC Connector
EC2_SUBNET="${SUBNET_ARRAY[0]}"
VPC_CONNECTOR_SUBNETS="${SUBNET_ARRAY[0]},${SUBNET_ARRAY[1]}"
ok "Using subnets: ${SUBNET_ARRAY[0]}, ${SUBNET_ARRAY[1]}"

# Get AZ for the first subnet (for EC2 placement)
EC2_AZ=$(aws ec2 describe-subnets --subnet-ids "$EC2_SUBNET" --region "$REGION" \
  --query 'Subnets[0].AvailabilityZone' --output text)
ok "EC2 will launch in $EC2_AZ"

# ── 4. Security Group ────────────────────────────────────────────────
info "Creating security group..."
SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
  --region "$REGION" \
  --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null)

if [[ -n "$SG_ID" && "$SG_ID" != "None" ]]; then
  ok "Security group '$SG_NAME' already exists ($SG_ID)"
else
  SG_ID=$(aws ec2 create-security-group \
    --group-name "$SG_NAME" \
    --description "CPQAI PostgreSQL - allow port 5432 from VPC" \
    --vpc-id "$VPC_ID" \
    --region "$REGION" \
    --query 'GroupId' --output text)

  aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" \
    --protocol tcp \
    --port 5432 \
    --cidr "$VPC_CIDR" \
    --region "$REGION" &>/dev/null

  # Allow SSH for initial setup (from VPC only)
  aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" \
    --protocol tcp \
    --port 22 \
    --cidr "$VPC_CIDR" \
    --region "$REGION" &>/dev/null

  ok "Security group created ($SG_ID)"
fi

# ── 5. EC2 Instance (PostgreSQL) ─────────────────────────────────────
info "Creating PostgreSQL EC2 instance..."
DB_INSTANCE_CREATED=false

EXISTING_INSTANCE=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=$DB_INSTANCE_NAME" "Name=instance-state-name,Values=running,stopped,pending" \
  --region "$REGION" \
  --query 'Reservations[0].Instances[0].InstanceId' --output text 2>/dev/null)

if [[ -n "$EXISTING_INSTANCE" && "$EXISTING_INSTANCE" != "None" ]]; then
  ok "Instance '$DB_INSTANCE_NAME' already exists ($EXISTING_INSTANCE)"
  DB_INSTANCE_ID="$EXISTING_INSTANCE"
else
  # Get latest Amazon Linux 2023 AMI
  AMI_ID=$(aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=al2023-ami-2023*-x86_64" "Name=state,Values=available" \
    --region "$REGION" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' --output text)

  # User-data script to install Docker and run PostgreSQL
  USER_DATA=$(cat <<'USERDATA'
#!/bin/bash
set -e

# Install Docker
dnf install -y docker
systemctl enable docker
systemctl start docker

# Create data directory on EBS volume
# Wait for the EBS volume to be attached
for i in $(seq 1 30); do
  if lsblk /dev/xvdf &>/dev/null || lsblk /dev/nvme1n1 &>/dev/null; then
    break
  fi
  sleep 2
done

# Determine the device name (varies by instance type)
if lsblk /dev/nvme1n1 &>/dev/null; then
  DATA_DEV="/dev/nvme1n1"
elif lsblk /dev/xvdf &>/dev/null; then
  DATA_DEV="/dev/xvdf"
else
  echo "EBS volume not found, using root volume"
  mkdir -p /pgdata
  DATA_DEV=""
fi

if [ -n "$DATA_DEV" ]; then
  # Format only if not already formatted
  if ! blkid "$DATA_DEV" &>/dev/null; then
    mkfs.ext4 "$DATA_DEV"
  fi
  mkdir -p /pgdata
  mount "$DATA_DEV" /pgdata
  echo "$DATA_DEV /pgdata ext4 defaults,nofail 0 2" >> /etc/fstab
fi

chmod 777 /pgdata

# Add ec2-user to docker group for non-root access
usermod -aG docker ec2-user

# Pull and run PostgreSQL with AGE + pgvector
docker pull gzdaniel/postgres-for-rag:16.6
docker run -d \
  --name cpqai-postgres \
  --restart always \
  -p 5432:5432 \
  -v /pgdata:/var/lib/postgresql/data \
  gzdaniel/postgres-for-rag:16.6
USERDATA
)

  DB_INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$DB_INSTANCE_TYPE" \
    --subnet-id "$EC2_SUBNET" \
    --security-group-ids "$SG_ID" \
    --associate-public-ip-address \
    --block-device-mappings "DeviceName=/dev/xvda,Ebs={VolumeSize=10,VolumeType=gp3}" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$DB_INSTANCE_NAME}]" \
    --user-data "$USER_DATA" \
    --region "$REGION" \
    --query 'Instances[0].InstanceId' --output text)

  ok "EC2 instance launched ($DB_INSTANCE_ID)"

  # Create and attach EBS data volume
  info "Creating ${DB_DISK_SIZE}GB gp3 EBS volume..."
  VOLUME_ID=$(aws ec2 create-volume \
    --availability-zone "$EC2_AZ" \
    --size "$DB_DISK_SIZE" \
    --volume-type gp3 \
    --tag-specifications "ResourceType=volume,Tags=[{Key=Name,Value=${DB_INSTANCE_NAME}-data}]" \
    --region "$REGION" \
    --query 'VolumeId' --output text)

  info "Waiting for volume to be available..."
  aws ec2 wait volume-available --volume-ids "$VOLUME_ID" --region "$REGION"

  info "Waiting for instance to be running..."
  aws ec2 wait instance-running --instance-ids "$DB_INSTANCE_ID" --region "$REGION"

  aws ec2 attach-volume \
    --volume-id "$VOLUME_ID" \
    --instance-id "$DB_INSTANCE_ID" \
    --device /dev/xvdf \
    --region "$REGION" &>/dev/null
  ok "EBS volume attached ($VOLUME_ID)"

  DB_INSTANCE_CREATED=true
fi

# Get the instance's private IP
DB_PRIVATE_IP=$(aws ec2 describe-instances \
  --instance-ids "$DB_INSTANCE_ID" \
  --region "$REGION" \
  --query 'Reservations[0].Instances[0].PrivateIpAddress' --output text)
ok "DB instance private IP: $DB_PRIVATE_IP"

# Wait for PostgreSQL to be ready and create user/database
if [[ "$DB_INSTANCE_CREATED" == "true" ]]; then
  info "Waiting for PostgreSQL to start (pulling image + initializing)..."

  # Get the public IP for SSH access
  DB_PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$DB_INSTANCE_ID" \
    --region "$REGION" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

  # Wait for instance status checks to pass
  info "Waiting for instance status checks..."
  aws ec2 wait instance-status-ok --instance-ids "$DB_INSTANCE_ID" --region "$REGION"

  # Generate a temporary SSH key for EC2 Instance Connect
  TMPKEY=$(mktemp -d)
  ssh-keygen -t rsa -b 2048 -f "$TMPKEY/key" -N "" -q

  # Helper: push temporary key and run a command via SSH
  ssh_run() {
    aws ec2-instance-connect send-ssh-public-key \
      --instance-id "$DB_INSTANCE_ID" \
      --instance-os-user ec2-user \
      --ssh-public-key "file://${TMPKEY}/key.pub" \
      --region "$REGION" &>/dev/null
    ssh -i "$TMPKEY/key" \
      -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -o ConnectTimeout=10 \
      -o LogLevel=ERROR \
      "ec2-user@${DB_PUBLIC_IP}" "$@"
  }

  # Temporarily allow SSH from our IP for initial setup
  MY_IP=$(curl -s https://checkip.amazonaws.com)/32
  aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" \
    --protocol tcp \
    --port 22 \
    --cidr "$MY_IP" \
    --region "$REGION" &>/dev/null || true
  SSH_RULE_ADDED=true

  # Wait for Docker + PostgreSQL to be ready
  info "Waiting for PostgreSQL container to start..."
  for i in $(seq 1 40); do
    OUTPUT=$(ssh_run "sudo docker exec cpqai-postgres pg_isready 2>/dev/null && echo READY || echo NOTREADY" 2>/dev/null) || true

    if [[ "$OUTPUT" == *"READY"* ]]; then
      ok "PostgreSQL is ready"
      break
    fi

    if [[ $i -eq 40 ]]; then
      warn "PostgreSQL readiness check timed out. The container may still be pulling."
      warn "Check manually: ssh ec2-user@$DB_PUBLIC_IP 'sudo docker exec cpqai-postgres pg_isready'"
      # Don't exit — continue setup, DB will be ready by the time App Runner starts
    fi
    sleep 5
  done

  info "Creating database user '$DB_USER' and database '$DB_NAME'..."
  if ssh_run "sudo docker exec -e PGPASSWORD=postgres cpqai-postgres psql -h 127.0.0.1 -U postgres -c \"CREATE USER ${DB_USER} WITH SUPERUSER CREATEDB CREATEROLE PASSWORD '${DB_PASSWORD}';\" && sudo docker exec -e PGPASSWORD=postgres cpqai-postgres psql -h 127.0.0.1 -U postgres -c \"CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};\" && sudo docker exec -e PGPASSWORD=postgres cpqai-postgres psql -h 127.0.0.1 -U postgres -d ${DB_NAME} -c \"CREATE EXTENSION IF NOT EXISTS age; CREATE EXTENSION IF NOT EXISTS vector;\"" 2>/dev/null; then
    ok "Database user, database, and extensions created"
  else
    warn "Could not create DB user/database via SSH. Create manually:"
    warn "  ssh ec2-user@$DB_PUBLIC_IP"
    warn "  sudo docker exec -e PGPASSWORD=postgres cpqai-postgres psql -h 127.0.0.1 -U postgres -c \"CREATE USER ${DB_USER} WITH SUPERUSER PASSWORD '${DB_PASSWORD}';\""
    warn "  sudo docker exec -e PGPASSWORD=postgres cpqai-postgres psql -h 127.0.0.1 -U postgres -c \"CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};\""
  fi

  # Remove temporary SSH rule
  if [[ "${SSH_RULE_ADDED:-}" == "true" ]]; then
    aws ec2 revoke-security-group-ingress \
      --group-id "$SG_ID" \
      --protocol tcp \
      --port 22 \
      --cidr "$MY_IP" \
      --region "$REGION" &>/dev/null || true
  fi

  # Clean up temporary SSH key
  rm -rf "$TMPKEY"
fi

# ── 6. Secrets Manager ───────────────────────────────────────────────
store_secret() {
  local name="$1" value="$2"
  if aws secretsmanager describe-secret --secret-id "$name" --region "$REGION" &>/dev/null; then
    aws secretsmanager put-secret-value \
      --secret-id "$name" \
      --secret-string "$value" \
      --region "$REGION" &>/dev/null
    ok "Secret '$name' updated"
  else
    aws secretsmanager create-secret \
      --name "$name" \
      --secret-string "$value" \
      --region "$REGION" &>/dev/null
    ok "Secret '$name' created"
  fi
}

info "Storing secrets..."
store_secret "cpqai-db-password" "$DB_PASSWORD"

API_KEY=$(openssl rand -base64 32 | tr -d '/+=' | head -c 40)
store_secret "cpqai-api-key" "$API_KEY"

DB_PASSWORD_ARN=$(aws secretsmanager describe-secret --secret-id "cpqai-db-password" --region "$REGION" --query 'ARN' --output text)
API_KEY_ARN=$(aws secretsmanager describe-secret --secret-id "cpqai-api-key" --region "$REGION" --query 'ARN' --output text)

# ── 7. IAM Roles ─────────────────────────────────────────────────────
create_role() {
  local role_name="$1" trust_policy="$2" description="$3"
  if aws iam get-role --role-name "$role_name" &>/dev/null; then
    ok "IAM role '$role_name' already exists"
  else
    aws iam create-role \
      --role-name "$role_name" \
      --assume-role-policy-document "$trust_policy" \
      --description "$description" \
      --output text &>/dev/null
    ok "IAM role '$role_name' created"
  fi
}

# App Runner instance role (Bedrock, Secrets Manager, CloudWatch)
info "Creating IAM roles..."

APPRUNNER_TRUST='{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "tasks.apprunner.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}'
create_role "$APPRUNNER_INSTANCE_ROLE" "$APPRUNNER_TRUST" "CPQAI App Runner instance role"

# Inline policy for Bedrock + Secrets Manager + CloudWatch
INSTANCE_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BedrockInvoke",
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": [
        "arn:aws:bedrock:*::foundation-model/*",
        "arn:aws:bedrock:*:${AWS_ACCOUNT_ID}:inference-profile/*"
      ]
    },
    {
      "Sid": "SecretsManagerRead",
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue"
      ],
      "Resource": [
        "${DB_PASSWORD_ARN}",
        "${API_KEY_ARN}"
      ]
    },
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:${REGION}:${AWS_ACCOUNT_ID}:*"
    }
  ]
}
EOF
)
aws iam put-role-policy \
  --role-name "$APPRUNNER_INSTANCE_ROLE" \
  --policy-name "${SERVICE_NAME}-instance-policy" \
  --policy-document "$INSTANCE_POLICY" &>/dev/null
ok "Instance role policy attached"

# App Runner ECR access role
ECR_TRUST='{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "build.apprunner.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}'
create_role "$APPRUNNER_ECR_ROLE" "$ECR_TRUST" "CPQAI App Runner ECR access role"

aws iam attach-role-policy \
  --role-name "$APPRUNNER_ECR_ROLE" \
  --policy-arn "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess" &>/dev/null
ok "ECR access role policy attached"

# CodeBuild service role
CODEBUILD_TRUST='{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "codebuild.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}'
create_role "$CODEBUILD_ROLE" "$CODEBUILD_TRUST" "CPQAI CodeBuild service role"

CODEBUILD_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ECRAuth",
      "Effect": "Allow",
      "Action": "ecr:GetAuthorizationToken",
      "Resource": "*"
    },
    {
      "Sid": "ECRPush",
      "Effect": "Allow",
      "Action": [
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:PutImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload"
      ],
      "Resource": "arn:aws:ecr:${REGION}:${AWS_ACCOUNT_ID}:repository/${ECR_REPO}"
    },
    {
      "Sid": "AppRunnerDeploy",
      "Effect": "Allow",
      "Action": "apprunner:StartDeployment",
      "Resource": "arn:aws:apprunner:${REGION}:${AWS_ACCOUNT_ID}:service/${SERVICE_NAME}/*"
    },
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:${REGION}:${AWS_ACCOUNT_ID}:*"
    },
    {
      "Sid": "S3Source",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:GetObjectVersion",
        "s3:GetBucketAcl",
        "s3:GetBucketLocation",
        "s3:PutObject"
      ],
      "Resource": [
        "arn:aws:s3:::${SERVICE_NAME}-codebuild-${AWS_ACCOUNT_ID}-${REGION}",
        "arn:aws:s3:::${SERVICE_NAME}-codebuild-${AWS_ACCOUNT_ID}-${REGION}/*"
      ]
    }
  ]
}
EOF
)
aws iam put-role-policy \
  --role-name "$CODEBUILD_ROLE" \
  --policy-name "${SERVICE_NAME}-codebuild-policy" \
  --policy-document "$CODEBUILD_POLICY" &>/dev/null
ok "CodeBuild role policy attached"

# ── 8. CodeBuild project ──────────────────────────────────────────────
info "Creating CodeBuild project..."
CODEBUILD_ROLE_ARN=$(aws iam get-role --role-name "$CODEBUILD_ROLE" --query 'Role.Arn' --output text)
SOURCE_BUCKET="${SERVICE_NAME}-codebuild-${AWS_ACCOUNT_ID}-${REGION}"

# Create S3 bucket for source uploads (like gcloud builds submit uploads source)
if aws s3api head-bucket --bucket "$SOURCE_BUCKET" --region "$REGION" &>/dev/null; then
  ok "Source bucket '$SOURCE_BUCKET' already exists"
else
  if [[ "$REGION" == "us-east-1" ]]; then
    aws s3api create-bucket --bucket "$SOURCE_BUCKET" --region "$REGION" &>/dev/null
  else
    aws s3api create-bucket --bucket "$SOURCE_BUCKET" --region "$REGION" \
      --create-bucket-configuration LocationConstraint="$REGION" &>/dev/null
  fi
  ok "Source bucket '$SOURCE_BUCKET' created"
fi

if aws codebuild batch-get-projects --names "$SERVICE_NAME" --region "$REGION" \
   --query 'projects[0].name' --output text 2>/dev/null | grep -q "$SERVICE_NAME"; then
  ok "CodeBuild project '$SERVICE_NAME' already exists — updating source"
  aws codebuild update-project \
    --name "$SERVICE_NAME" \
    --source "{\"type\":\"S3\",\"location\":\"${SOURCE_BUCKET}/source.zip\",\"buildspec\":\"deploy/aws/buildspec.yml\"}" \
    --region "$REGION" \
    --output text &>/dev/null
else
  # IAM roles need a few seconds to propagate before CodeBuild can assume them
  info "Waiting for IAM role propagation..."
  sleep 10

  aws codebuild create-project \
    --name "$SERVICE_NAME" \
    --source "{\"type\":\"S3\",\"location\":\"${SOURCE_BUCKET}/source.zip\",\"buildspec\":\"deploy/aws/buildspec.yml\"}" \
    --artifacts '{"type":"NO_ARTIFACTS"}' \
    --environment "{
      \"type\": \"LINUX_CONTAINER\",
      \"computeType\": \"BUILD_GENERAL1_MEDIUM\",
      \"image\": \"aws/codebuild/amazonlinux-x86_64-standard:5.0\",
      \"privilegedMode\": true,
      \"environmentVariables\": [
        {\"name\": \"AWS_ACCOUNT_ID\", \"value\": \"${AWS_ACCOUNT_ID}\"},
        {\"name\": \"AWS_REGION\", \"value\": \"${REGION}\"},
        {\"name\": \"ECR_REPO\", \"value\": \"${ECR_REPO}\"},
        {\"name\": \"SERVICE_NAME\", \"value\": \"${SERVICE_NAME}\"}
      ]
    }" \
    --service-role "$CODEBUILD_ROLE_ARN" \
    --region "$REGION" \
    --output text &>/dev/null
  ok "CodeBuild project created"
fi

# ── 9. Build with CodeBuild (remote) ─────────────────────────────────
BUILD_TAG=$(date +%Y%m%d-%H%M%S)
info "Uploading source and building with CodeBuild (tag: $BUILD_TAG)..."

# Package source as ZIP and upload to S3 (CodeBuild S3 source requires ZIP)
(cd "$(pwd)" && zip -qr /tmp/cpqai-source.zip . \
  -x '.git/*' 'node_modules/*' '.venv/*' '__pycache__/*' '.env' '.env.*' 'deploy/gcp/.env.gcp')
aws s3 cp /tmp/cpqai-source.zip "s3://${SOURCE_BUCKET}/source.zip" --region "$REGION" &>/dev/null
rm -f /tmp/cpqai-source.zip
ok "Source uploaded to S3"

# Start the build
BUILD_ID=$(aws codebuild start-build \
  --project-name "$SERVICE_NAME" \
  --environment-variables-override "[{\"name\":\"BUILD_TAG\",\"value\":\"${BUILD_TAG}\",\"type\":\"PLAINTEXT\"}]" \
  --region "$REGION" \
  --query 'build.id' --output text)
info "CodeBuild started: $BUILD_ID"

# Wait for build to complete
info "Waiting for build to complete (this may take several minutes)..."
while true; do
  BUILD_STATUS=$(aws codebuild batch-get-builds --ids "$BUILD_ID" --region "$REGION" \
    --query 'builds[0].buildStatus' --output text)
  BUILD_PHASE=$(aws codebuild batch-get-builds --ids "$BUILD_ID" --region "$REGION" \
    --query 'builds[0].currentPhase' --output text)

  case "$BUILD_STATUS" in
    SUCCEEDED)
      ok "CodeBuild completed successfully"
      break
      ;;
    FAILED|FAULT|STOPPED|TIMED_OUT)
      err "CodeBuild failed with status: $BUILD_STATUS"
      err "View logs: aws codebuild batch-get-builds --ids $BUILD_ID --region $REGION --query 'builds[0].logs.deepLink' --output text"
      exit 1
      ;;
    IN_PROGRESS)
      info "  Build phase: $BUILD_PHASE..."
      sleep 15
      ;;
    *)
      sleep 10
      ;;
  esac
done

# ── 10. Bedrock VPC Endpoint ─────────────────────────────────────────
# App Runner with a VPC Connector loses direct internet access, so it needs
# a VPC endpoint to reach Bedrock (LLM + embeddings) via private link.
VPCE_SG_NAME="${SERVICE_NAME}-bedrock-vpce-sg"
info "Creating Bedrock VPC endpoint..."

EXISTING_VPCE=$(aws ec2 describe-vpc-endpoints \
  --filters "Name=service-name,Values=com.amazonaws.${REGION}.bedrock-runtime" "Name=vpc-id,Values=$VPC_ID" "Name=vpc-endpoint-state,Values=available,pending" \
  --region "$REGION" \
  --query 'VpcEndpoints[0].VpcEndpointId' --output text 2>/dev/null)

if [[ -n "$EXISTING_VPCE" && "$EXISTING_VPCE" != "None" ]]; then
  ok "Bedrock VPC endpoint already exists ($EXISTING_VPCE)"
else
  # Security group for the VPC endpoint (allow HTTPS from VPC)
  VPCE_SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=$VPCE_SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
    --region "$REGION" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null)

  if [[ -z "$VPCE_SG_ID" || "$VPCE_SG_ID" == "None" ]]; then
    VPCE_SG_ID=$(aws ec2 create-security-group \
      --group-name "$VPCE_SG_NAME" \
      --description "Allow HTTPS to Bedrock VPC endpoint" \
      --vpc-id "$VPC_ID" \
      --region "$REGION" \
      --query 'GroupId' --output text)
    aws ec2 authorize-security-group-ingress \
      --group-id "$VPCE_SG_ID" \
      --protocol tcp \
      --port 443 \
      --cidr "$VPC_CIDR" \
      --region "$REGION" &>/dev/null
  fi

  # Get default subnets (one per AZ)
  DEFAULT_SUBNETS=$(aws ec2 describe-subnets \
    --filters "Name=vpc-id,Values=$VPC_ID" "Name=default-for-az,Values=true" \
    --region "$REGION" \
    --query 'Subnets[*].SubnetId' --output text)

  VPCE_ID=$(aws ec2 create-vpc-endpoint \
    --vpc-id "$VPC_ID" \
    --vpc-endpoint-type Interface \
    --service-name "com.amazonaws.${REGION}.bedrock-runtime" \
    --subnet-ids $DEFAULT_SUBNETS \
    --security-group-ids "$VPCE_SG_ID" \
    --private-dns-enabled \
    --region "$REGION" \
    --query 'VpcEndpoint.VpcEndpointId' --output text)

  info "Waiting for VPC endpoint to become available..."
  aws ec2 wait vpc-endpoint-available --vpc-endpoint-ids "$VPCE_ID" --region "$REGION" 2>/dev/null || \
    for i in $(seq 1 12); do
      STATE=$(aws ec2 describe-vpc-endpoints --vpc-endpoint-ids "$VPCE_ID" --region "$REGION" --query 'VpcEndpoints[0].State' --output text 2>/dev/null)
      if [[ "$STATE" == "available" ]]; then break; fi
      sleep 10
    done
  ok "Bedrock VPC endpoint created ($VPCE_ID)"
fi

# ── 11. App Runner VPC Connector ─────────────────────────────────────
VPC_CONNECTOR_NAME="${SERVICE_NAME}-vpc-connector"
info "Creating App Runner VPC Connector..."

EXISTING_CONNECTOR=$(aws apprunner list-vpc-connectors \
  --region "$REGION" \
  --query "VpcConnectors[?VpcConnectorName=='${VPC_CONNECTOR_NAME}' && Status=='ACTIVE'].VpcConnectorArn | [0]" \
  --output text 2>/dev/null)

if [[ -n "$EXISTING_CONNECTOR" && "$EXISTING_CONNECTOR" != "None" ]]; then
  VPC_CONNECTOR_ARN="$EXISTING_CONNECTOR"
  ok "VPC Connector '$VPC_CONNECTOR_NAME' already exists"
else
  VPC_CONNECTOR_ARN=$(aws apprunner create-vpc-connector \
    --vpc-connector-name "$VPC_CONNECTOR_NAME" \
    --subnets ${SUBNET_ARRAY[0]} ${SUBNET_ARRAY[1]} \
    --security-groups "$SG_ID" \
    --region "$REGION" \
    --query 'VpcConnector.VpcConnectorArn' --output text)
  ok "VPC Connector created ($VPC_CONNECTOR_ARN)"
fi

# ── 11. App Runner Service ───────────────────────────────────────────
info "Creating App Runner service..."

INSTANCE_ROLE_ARN=$(aws iam get-role --role-name "$APPRUNNER_INSTANCE_ROLE" --query 'Role.Arn' --output text)
ECR_ROLE_ARN=$(aws iam get-role --role-name "$APPRUNNER_ECR_ROLE" --query 'Role.Arn' --output text)

# Check if service already exists
EXISTING_SERVICE=$(aws apprunner list-services \
  --region "$REGION" \
  --query "ServiceSummaryList[?ServiceName=='${SERVICE_NAME}'].ServiceArn | [0]" \
  --output text 2>/dev/null)

if [[ -n "$EXISTING_SERVICE" && "$EXISTING_SERVICE" != "None" ]]; then
  ok "App Runner service '$SERVICE_NAME' already exists — triggering new deployment"
  aws apprunner start-deployment --service-arn "$EXISTING_SERVICE" --region "$REGION" &>/dev/null
  SERVICE_ARN="$EXISTING_SERVICE"
else
  SERVICE_ARN=$(aws apprunner create-service \
    --service-name "$SERVICE_NAME" \
    --source-configuration "{
      \"AuthenticationConfiguration\": {
        \"AccessRoleArn\": \"${ECR_ROLE_ARN}\"
      },
      \"ImageRepository\": {
        \"ImageIdentifier\": \"${ECR_URI}:latest\",
        \"ImageRepositoryType\": \"ECR\",
        \"ImageConfiguration\": {
          \"Port\": \"8080\",
          \"RuntimeEnvironmentVariables\": {
            \"POSTGRES_HOST\": \"${DB_PRIVATE_IP}\",
            \"POSTGRES_PORT\": \"5432\",
            \"POSTGRES_USER\": \"${DB_USER}\",
            \"POSTGRES_DATABASE\": \"${DB_NAME}\",
            \"AWS_REGION\": \"${REGION}\"
          },
          \"RuntimeEnvironmentSecrets\": {
            \"POSTGRES_PASSWORD\": \"${DB_PASSWORD_ARN}\",
            \"LIGHTRAG_API_KEY\": \"${API_KEY_ARN}\"
          }
        }
      }
    }" \
    --instance-configuration "{
      \"Cpu\": \"2 vCPU\",
      \"Memory\": \"4 GB\",
      \"InstanceRoleArn\": \"${INSTANCE_ROLE_ARN}\"
    }" \
    --health-check-configuration "{
      \"Protocol\": \"HTTP\",
      \"Path\": \"/health\",
      \"Interval\": 10,
      \"Timeout\": 5,
      \"HealthyThreshold\": 1,
      \"UnhealthyThreshold\": 5
    }" \
    --network-configuration "{
      \"EgressConfiguration\": {
        \"EgressType\": \"VPC\",
        \"VpcConnectorArn\": \"${VPC_CONNECTOR_ARN}\"
      }
    }" \
    --region "$REGION" \
    --query 'Service.ServiceArn' --output text)

  ok "App Runner service created"
fi

# Wait for service to be running
info "Waiting for App Runner service to become active (this may take a few minutes)..."
for i in $(seq 1 60); do
  STATUS=$(aws apprunner describe-service \
    --service-arn "$SERVICE_ARN" \
    --region "$REGION" \
    --query 'Service.Status' --output text 2>/dev/null)

  if [[ "$STATUS" == "RUNNING" ]]; then
    ok "App Runner service is running"
    break
  elif [[ "$STATUS" == "CREATE_FAILED" || "$STATUS" == "DELETE_FAILED" ]]; then
    err "App Runner service failed: $STATUS"
    exit 1
  fi

  if [[ $i -eq 60 ]]; then
    warn "Timed out waiting for App Runner. Current status: $STATUS"
    warn "Check: aws apprunner describe-service --service-arn $SERVICE_ARN --region $REGION"
  fi
  sleep 10
done

SERVICE_URL=$(aws apprunner describe-service \
  --service-arn "$SERVICE_ARN" \
  --region "$REGION" \
  --query 'Service.ServiceUrl' --output text 2>/dev/null || echo "(pending)")

# Update CodeBuild project with SERVICE_ARN for future deployments
info "Updating CodeBuild project with App Runner service ARN..."
aws codebuild update-project \
  --name "$SERVICE_NAME" \
  --environment "{
    \"type\": \"LINUX_CONTAINER\",
    \"computeType\": \"BUILD_GENERAL1_MEDIUM\",
    \"image\": \"aws/codebuild/amazonlinux-x86_64-standard:5.0\",
    \"privilegedMode\": true,
    \"environmentVariables\": [
      {\"name\": \"AWS_ACCOUNT_ID\", \"value\": \"${AWS_ACCOUNT_ID}\"},
      {\"name\": \"AWS_REGION\", \"value\": \"${REGION}\"},
      {\"name\": \"ECR_REPO\", \"value\": \"${ECR_REPO}\"},
      {\"name\": \"SERVICE_NAME\", \"value\": \"${SERVICE_NAME}\"},
      {\"name\": \"SERVICE_ARN\", \"value\": \"${SERVICE_ARN}\"}
    ]
  }" \
  --region "$REGION" \
  --output text &>/dev/null
ok "CodeBuild project updated with SERVICE_ARN"

# ── 13. Print summary ────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "  CPQAI deployed on AWS"
echo "=============================================="
echo ""
echo "  Service URL:    https://${SERVICE_URL}"
echo "  DB Instance:    $DB_INSTANCE_NAME ($DB_PRIVATE_IP:5432)"
echo "  Database:       $DB_NAME"
echo "  DB User:        $DB_USER"
echo "  API Key:        $API_KEY"
echo ""
echo "  App Runner ARN: $SERVICE_ARN"
echo "  ECR Image:      ${ECR_URI}:latest"
echo ""
echo "  Test:"
echo "    curl -H 'X-API-Key: ${API_KEY}' https://${SERVICE_URL}/health"
echo ""
echo "  Secrets stored in Secrets Manager:"
echo "    - cpqai-db-password"
echo "    - cpqai-api-key"
echo ""
echo "  DB instance management:"
echo "    aws ec2 start-instances --instance-ids $DB_INSTANCE_ID --region $REGION"
echo "    aws ec2 stop-instances  --instance-ids $DB_INSTANCE_ID --region $REGION"
echo "    ssh ec2-user@$DB_PUBLIC_IP  # via EC2 Instance Connect"
echo ""
echo "  Redeploy:"
echo "    aws codebuild start-build --project-name $SERVICE_NAME --region $REGION"
echo ""
warn "Save the API key above — it is also in Secrets Manager (cpqai-api-key)"
echo ""
