#!/usr/bin/env bash
#
# CPQAI — AWS resource cleanup
#
# Deletes all resources created by setup.sh:
#   App Runner service, VPC Connector, EC2 instance + EBS volume,
#   Security Group, Secrets Manager secrets, ECR repository,
#   IAM roles + policies, CodeBuild project.
#
# Usage:
#   ./deploy/aws/cleanup.sh              # interactive (confirms each step)
#   ./deploy/aws/cleanup.sh --yes        # skip confirmations
#   ./deploy/aws/cleanup.sh --defaults   # use default names + skip confirmations

set -euo pipefail

# ── Colour helpers ──────────────────────────────────────────────────
info()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
ok()    { printf '\033[1;32m[OK]\033[0m    %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
err()   { printf '\033[1;31m[ERR]\033[0m   %s\n' "$*" >&2; }
skip()  { printf '\033[1;90m[SKIP]\033[0m  %s\n' "$*"; }

# ── Prerequisites check ────────────────────────────────────────────
if ! command -v aws &>/dev/null; then
  err "AWS CLI not found."
  exit 1
fi

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null) || {
  err "AWS CLI not configured. Run: aws configure"
  exit 1
}

# ── Configuration ───────────────────────────────────────────────────
AUTO_YES=false

if [[ "${1:-}" == "--defaults" ]]; then
  AUTO_YES=true
  REGION="us-east-1"
  SERVICE_NAME="cpqai"
  DB_INSTANCE_NAME="cpqai-db"
  ECR_REPO="cpqai"
elif [[ "${1:-}" == "--yes" ]]; then
  AUTO_YES=true
  read -rp "Region [us-east-1]: "                    REGION;           REGION=${REGION:-us-east-1}
  read -rp "App Runner service [cpqai]: "            SERVICE_NAME;     SERVICE_NAME=${SERVICE_NAME:-cpqai}
  read -rp "EC2 instance name [cpqai-db]: "          DB_INSTANCE_NAME; DB_INSTANCE_NAME=${DB_INSTANCE_NAME:-cpqai-db}
  read -rp "ECR repository [cpqai]: "                ECR_REPO;         ECR_REPO=${ECR_REPO:-cpqai}
else
  read -rp "Region [us-east-1]: "                    REGION;           REGION=${REGION:-us-east-1}
  read -rp "App Runner service [cpqai]: "            SERVICE_NAME;     SERVICE_NAME=${SERVICE_NAME:-cpqai}
  read -rp "EC2 instance name [cpqai-db]: "          DB_INSTANCE_NAME; DB_INSTANCE_NAME=${DB_INSTANCE_NAME:-cpqai-db}
  read -rp "ECR repository [cpqai]: "                ECR_REPO;         ECR_REPO=${ECR_REPO:-cpqai}
fi

APPRUNNER_INSTANCE_ROLE="${SERVICE_NAME}-apprunner-instance-role"
APPRUNNER_ECR_ROLE="${SERVICE_NAME}-apprunner-ecr-role"
CODEBUILD_ROLE="${SERVICE_NAME}-codebuild-role"
SG_NAME="${DB_INSTANCE_NAME}-sg"
VPCE_SG_NAME="${SERVICE_NAME}-bedrock-vpce-sg"
VPC_CONNECTOR_NAME="${SERVICE_NAME}-vpc-connector"
SOURCE_BUCKET="${SERVICE_NAME}-codebuild-${AWS_ACCOUNT_ID}-${REGION}"

echo ""
warn "This will permanently delete the following resources in account '$AWS_ACCOUNT_ID' ($REGION):"
echo ""
echo "  - App Runner service:    $SERVICE_NAME"
echo "  - VPC Connector:         $VPC_CONNECTOR_NAME"
echo "  - Bedrock VPC endpoint:  (bedrock-runtime)"
echo "  - EC2 instance:          $DB_INSTANCE_NAME"
echo "  - EBS volume:            ${DB_INSTANCE_NAME}-data"
echo "  - Security groups:       $SG_NAME, $VPCE_SG_NAME"
echo "  - Secrets:               cpqai-db-password, cpqai-api-key"
echo "  - ECR repository:        $ECR_REPO"
echo "  - S3 bucket:             $SOURCE_BUCKET"
echo "  - IAM roles:             $APPRUNNER_INSTANCE_ROLE, $APPRUNNER_ECR_ROLE, $CODEBUILD_ROLE"
echo "  - CodeBuild project:     $SERVICE_NAME"
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

# ── 1. App Runner service ────────────────────────────────────────────
info "Deleting App Runner service '$SERVICE_NAME'..."
SERVICE_ARN=$(aws apprunner list-services \
  --region "$REGION" \
  --query "ServiceSummaryList[?ServiceName=='${SERVICE_NAME}'].ServiceArn | [0]" \
  --output text 2>/dev/null)

if [[ -n "$SERVICE_ARN" && "$SERVICE_ARN" != "None" ]]; then
  aws apprunner delete-service --service-arn "$SERVICE_ARN" --region "$REGION" &>/dev/null
  ok "App Runner service deletion initiated"

  # Wait for deletion to complete before cleaning up VPC connector
  info "Waiting for App Runner service to be deleted..."
  for i in $(seq 1 30); do
    STATUS=$(aws apprunner describe-service \
      --service-arn "$SERVICE_ARN" \
      --region "$REGION" \
      --query 'Service.Status' --output text 2>/dev/null) || break
    if [[ "$STATUS" == "DELETED" || "$STATUS" == "DELETE_FAILED" ]]; then
      break
    fi
    sleep 10
  done
  ok "App Runner service deleted"
else
  skip "App Runner service '$SERVICE_NAME' not found"
fi

# ── 2. VPC Connector ─────────────────────────────────────────────────
info "Deleting VPC Connector '$VPC_CONNECTOR_NAME'..."
VPC_CONNECTOR_ARN=$(aws apprunner list-vpc-connectors \
  --region "$REGION" \
  --query "VpcConnectors[?VpcConnectorName=='${VPC_CONNECTOR_NAME}' && Status=='ACTIVE'].VpcConnectorArn | [0]" \
  --output text 2>/dev/null)

if [[ -n "$VPC_CONNECTOR_ARN" && "$VPC_CONNECTOR_ARN" != "None" ]]; then
  aws apprunner delete-vpc-connector --vpc-connector-arn "$VPC_CONNECTOR_ARN" --region "$REGION" &>/dev/null
  ok "VPC Connector deleted"
else
  skip "VPC Connector '$VPC_CONNECTOR_NAME' not found"
fi

# ── Look up default VPC (needed for endpoint + security group cleanup) ──
VPC_ID=$(aws ec2 describe-vpcs \
  --filters "Name=isDefault,Values=true" \
  --region "$REGION" \
  --query 'Vpcs[0].VpcId' --output text 2>/dev/null)

# ── 2b. Bedrock VPC Endpoint ───────────────────────────────────────────
info "Deleting Bedrock VPC endpoint..."
VPCE_ID=$(aws ec2 describe-vpc-endpoints \
  --filters "Name=service-name,Values=com.amazonaws.${REGION}.bedrock-runtime" "Name=vpc-id,Values=$VPC_ID" "Name=vpc-endpoint-state,Values=available,pending" \
  --region "$REGION" \
  --query 'VpcEndpoints[0].VpcEndpointId' --output text 2>/dev/null)

if [[ -n "$VPCE_ID" && "$VPCE_ID" != "None" ]]; then
  aws ec2 delete-vpc-endpoints --vpc-endpoint-ids "$VPCE_ID" --region "$REGION" &>/dev/null
  ok "Bedrock VPC endpoint deleted ($VPCE_ID)"
else
  skip "Bedrock VPC endpoint not found"
fi

# Delete VPC endpoint security group
VPCE_SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=$VPCE_SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
  --region "$REGION" \
  --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null)

if [[ -n "$VPCE_SG_ID" && "$VPCE_SG_ID" != "None" ]]; then
  # VPC endpoint ENIs take up to 30s to detach after endpoint deletion
  for i in $(seq 1 4); do
    if aws ec2 delete-security-group --group-id "$VPCE_SG_ID" --region "$REGION" 2>/dev/null; then
      ok "VPC endpoint security group deleted"
      break
    fi
    if [[ $i -eq 4 ]]; then
      warn "Could not delete security group '$VPCE_SG_NAME' — delete manually: aws ec2 delete-security-group --group-id $VPCE_SG_ID --region $REGION"
    fi
    sleep 10
  done
else
  skip "VPC endpoint security group '$VPCE_SG_NAME' not found"
fi

# ── 3. EC2 Instance ──────────────────────────────────────────────────
info "Terminating EC2 instance '$DB_INSTANCE_NAME'..."
DB_INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=$DB_INSTANCE_NAME" "Name=instance-state-name,Values=running,stopped,pending,stopping" \
  --region "$REGION" \
  --query 'Reservations[0].Instances[0].InstanceId' --output text 2>/dev/null)

if [[ -n "$DB_INSTANCE_ID" && "$DB_INSTANCE_ID" != "None" ]]; then
  aws ec2 terminate-instances --instance-ids "$DB_INSTANCE_ID" --region "$REGION" &>/dev/null
  info "Waiting for instance to terminate..."
  aws ec2 wait instance-terminated --instance-ids "$DB_INSTANCE_ID" --region "$REGION" 2>/dev/null || true
  ok "EC2 instance terminated"
else
  skip "EC2 instance '$DB_INSTANCE_NAME' not found"
fi

# ── 4. EBS Volume ────────────────────────────────────────────────────
info "Deleting EBS volume '${DB_INSTANCE_NAME}-data'..."
VOLUME_ID=$(aws ec2 describe-volumes \
  --filters "Name=tag:Name,Values=${DB_INSTANCE_NAME}-data" "Name=status,Values=available" \
  --region "$REGION" \
  --query 'Volumes[0].VolumeId' --output text 2>/dev/null)

if [[ -n "$VOLUME_ID" && "$VOLUME_ID" != "None" ]]; then
  aws ec2 delete-volume --volume-id "$VOLUME_ID" --region "$REGION" &>/dev/null
  ok "EBS volume deleted"
else
  # Volume may still be attached or already deleted
  VOLUME_ID=$(aws ec2 describe-volumes \
    --filters "Name=tag:Name,Values=${DB_INSTANCE_NAME}-data" \
    --region "$REGION" \
    --query 'Volumes[0].VolumeId' --output text 2>/dev/null)
  if [[ -n "$VOLUME_ID" && "$VOLUME_ID" != "None" ]]; then
    warn "EBS volume exists but is not in 'available' state. It may have been deleted with the instance."
    warn "Check: aws ec2 describe-volumes --volume-ids $VOLUME_ID --region $REGION"
  else
    skip "EBS volume '${DB_INSTANCE_NAME}-data' not found"
  fi
fi

# ── 5. Security Group ────────────────────────────────────────────────
info "Deleting security group '$SG_NAME'..."

SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
  --region "$REGION" \
  --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null)

if [[ -n "$SG_ID" && "$SG_ID" != "None" ]]; then
  # Security groups can take a moment to become deletable after instances are terminated
  for i in $(seq 1 6); do
    if aws ec2 delete-security-group --group-id "$SG_ID" --region "$REGION" 2>/dev/null; then
      ok "Security group deleted"
      break
    fi
    if [[ $i -eq 6 ]]; then
      warn "Could not delete security group '$SG_NAME' — may still be in use by VPC connector"
      warn "Delete manually: aws ec2 delete-security-group --group-id $SG_ID --region $REGION"
    fi
    sleep 10
  done
else
  skip "Security group '$SG_NAME' not found"
fi

# ── 6. Secrets Manager ───────────────────────────────────────────────
try_delete "secret 'cpqai-db-password'" \
  aws secretsmanager delete-secret --secret-id cpqai-db-password \
    --force-delete-without-recovery --region "$REGION"

try_delete "secret 'cpqai-api-key'" \
  aws secretsmanager delete-secret --secret-id cpqai-api-key \
    --force-delete-without-recovery --region "$REGION"

# ── 7. ECR Repository ────────────────────────────────────────────────
try_delete "ECR repository '$ECR_REPO'" \
  aws ecr delete-repository --repository-name "$ECR_REPO" \
    --force --region "$REGION"

# ── 8. IAM Roles + Policies ──────────────────────────────────────────
delete_role() {
  local role_name="$1"
  info "Deleting IAM role '$role_name'..."

  if ! aws iam get-role --role-name "$role_name" &>/dev/null; then
    skip "IAM role '$role_name' not found"
    return
  fi

  # Delete inline policies
  POLICIES=$(aws iam list-role-policies --role-name "$role_name" --query 'PolicyNames' --output text 2>/dev/null)
  for policy in $POLICIES; do
    aws iam delete-role-policy --role-name "$role_name" --policy-name "$policy" &>/dev/null || true
  done

  # Detach managed policies
  ATTACHED=$(aws iam list-attached-role-policies --role-name "$role_name" --query 'AttachedPolicies[*].PolicyArn' --output text 2>/dev/null)
  for arn in $ATTACHED; do
    aws iam detach-role-policy --role-name "$role_name" --policy-arn "$arn" &>/dev/null || true
  done

  # Delete the role
  aws iam delete-role --role-name "$role_name" &>/dev/null
  ok "IAM role '$role_name' deleted"
}

delete_role "$APPRUNNER_INSTANCE_ROLE"
delete_role "$APPRUNNER_ECR_ROLE"
delete_role "$CODEBUILD_ROLE"

# ── 9. S3 Source Bucket ───────────────────────────────────────────────
info "Deleting S3 source bucket '$SOURCE_BUCKET'..."
if aws s3api head-bucket --bucket "$SOURCE_BUCKET" --region "$REGION" &>/dev/null; then
  aws s3 rm "s3://${SOURCE_BUCKET}" --recursive --region "$REGION" &>/dev/null
  aws s3api delete-bucket --bucket "$SOURCE_BUCKET" --region "$REGION" &>/dev/null
  ok "S3 source bucket deleted"
else
  skip "S3 source bucket '$SOURCE_BUCKET' not found"
fi

# ── 10. CodeBuild Project ────────────────────────────────────────────
try_delete "CodeBuild project '$SERVICE_NAME'" \
  aws codebuild delete-project --name "$SERVICE_NAME" --region "$REGION"

# ── Done ─────────────────────────────────────────────────────────────
echo ""
ok "Cleanup complete. All CPQAI resources have been removed from account '$AWS_ACCOUNT_ID' ($REGION)."
echo ""
