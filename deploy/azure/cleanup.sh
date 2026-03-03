#!/usr/bin/env bash
#
# CPQAI — Azure resource cleanup
#
# By default, deletes the entire resource group in one operation (fastest,
# guaranteed to remove everything).
#
# Use --manual to delete individual resources instead, keeping the resource
# group intact.
#
# Usage:
#   ./deploy/azure/cleanup.sh                    # delete resource group (interactive)
#   ./deploy/azure/cleanup.sh --yes              # delete resource group (skip confirmation)
#   ./deploy/azure/cleanup.sh --defaults         # default names + skip confirmation
#   ./deploy/azure/cleanup.sh --manual           # delete resources individually (interactive)
#   ./deploy/azure/cleanup.sh --manual --yes     # delete resources individually (skip confirmation)
#   ./deploy/azure/cleanup.sh --manual --defaults # individual + default names + skip confirmation

set -euo pipefail

# ── Colour helpers ──────────────────────────────────────────────────
info()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
ok()    { printf '\033[1;32m[OK]\033[0m    %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
err()   { printf '\033[1;31m[ERR]\033[0m   %s\n' "$*" >&2; }
skip()  { printf '\033[1;90m[SKIP]\033[0m  %s\n' "$*"; }

# ── Prerequisites check ────────────────────────────────────────────
if ! command -v az &>/dev/null; then
  err "Azure CLI not found."
  exit 1
fi

SUBSCRIPTION_ID=$(az account show --query id --output tsv 2>/dev/null) || {
  err "Azure CLI not logged in. Run: az login"
  exit 1
}

# ── Parse flags ───────────────────────────────────────────────────
AUTO_YES=false
MANUAL=false

for arg in "$@"; do
  case "$arg" in
    --yes)      AUTO_YES=true ;;
    --defaults) AUTO_YES=true; MANUAL_DEFAULTS=true ;;
    --manual)   MANUAL=true ;;
    *)          err "Unknown flag: $arg"; exit 1 ;;
  esac
done

# ── Configuration ─────────────────────────────────────────────────
if [[ "${MANUAL_DEFAULTS:-}" == "true" ]]; then
  LOCATION="eastus"
  RESOURCE_GROUP="cpqai-rg"
  SERVICE_NAME="cpqai"
  DB_VM_NAME="cpqai-db"
  ACR_NAME="cpqaiacr"
  OPENAI_NAME="cpqai-openai"
  VNET_NAME="cpqai-vnet"
  KV_NAME="cpqai-kv"
  IDENTITY_NAME="cpqai-identity"
elif [[ "$MANUAL" == "true" ]]; then
  read -rp "Resource group [cpqai-rg]: "              RESOURCE_GROUP; RESOURCE_GROUP=${RESOURCE_GROUP:-cpqai-rg}
  read -rp "Container App name [cpqai]: "             SERVICE_NAME;   SERVICE_NAME=${SERVICE_NAME:-cpqai}
  read -rp "DB VM name [cpqai-db]: "                  DB_VM_NAME;     DB_VM_NAME=${DB_VM_NAME:-cpqai-db}
  read -rp "Container Registry name [cpqaiacr]: "     ACR_NAME;       ACR_NAME=${ACR_NAME:-cpqaiacr}
  read -rp "Azure OpenAI name [cpqai-openai]: "       OPENAI_NAME;    OPENAI_NAME=${OPENAI_NAME:-cpqai-openai}
  read -rp "VNet name [cpqai-vnet]: "                 VNET_NAME;      VNET_NAME=${VNET_NAME:-cpqai-vnet}
  read -rp "Key Vault name [cpqai-kv]: "              KV_NAME;        KV_NAME=${KV_NAME:-cpqai-kv}
  read -rp "Managed Identity name [cpqai-identity]: " IDENTITY_NAME;  IDENTITY_NAME=${IDENTITY_NAME:-cpqai-identity}
else
  # Resource group mode — only need the group name
  if [[ "$AUTO_YES" != "true" ]]; then
    read -rp "Resource group [cpqai-rg]: " RESOURCE_GROUP
  fi
  RESOURCE_GROUP=${RESOURCE_GROUP:-cpqai-rg}
fi

# ══════════════════════════════════════════════════════════════════
# Manual mode — delete individual resources
# ══════════════════════════════════════════════════════════════════
if [[ "$MANUAL" == "true" ]]; then
  ENV_NAME="${SERVICE_NAME}-env"
  NSG_NAME="${DB_VM_NAME}-nsg"
  APIM_NAME="${SERVICE_NAME}-apim"

  echo ""
  warn "This will permanently delete the following resources in resource group '$RESOURCE_GROUP':"
  echo ""
  echo "  - Container App:          $SERVICE_NAME"
  echo "  - Container Apps Env:     $ENV_NAME"
  echo "  - VM + disks:             $DB_VM_NAME"
  echo "  - Azure OpenAI:           $OPENAI_NAME"
  echo "  - Container Registry:     $ACR_NAME"
  echo "  - Key Vault:              $KV_NAME"
  echo "  - VNet:                   $VNET_NAME"
  echo "  - NSG:                    $NSG_NAME"
  echo "  - Managed Identity:       $IDENTITY_NAME"
  echo "  - APIM (if exists):       $APIM_NAME"
  echo ""
  echo "  NOTE: The resource group itself will NOT be deleted."
  echo ""

  if [[ "$AUTO_YES" != "true" ]]; then
    read -rp "Are you sure? Type 'delete' to confirm: " CONFIRM
    if [[ "$CONFIRM" != "delete" ]]; then
      err "Aborted."
      exit 1
    fi
  fi

  # ── Helper to delete a resource safely ────────────────────────
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

  # ── 1. API Management (if exists) ────────────────────────────
  if az apim show --name "$APIM_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
    try_delete "APIM '$APIM_NAME'" \
      az apim delete --name "$APIM_NAME" --resource-group "$RESOURCE_GROUP" --yes
  fi

  # ── 2. Container App ─────────────────────────────────────────
  try_delete "Container App '$SERVICE_NAME'" \
    az containerapp delete --name "$SERVICE_NAME" --resource-group "$RESOURCE_GROUP" --yes

  # ── 3. Container Apps Environment ─────────────────────────────
  try_delete "Container Apps Environment '$ENV_NAME'" \
    az containerapp env delete --name "$ENV_NAME" --resource-group "$RESOURCE_GROUP" --yes

  # ── 4. VM (includes OS disk, data disk is separate) ───────────
  info "Deleting VM '$DB_VM_NAME'..."
  if az vm show --name "$DB_VM_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
    # Get data disk IDs before deleting the VM
    DATA_DISKS=$(az vm show --name "$DB_VM_NAME" --resource-group "$RESOURCE_GROUP" \
      --query "storageProfile.dataDisks[*].managedDisk.id" --output tsv 2>/dev/null) || true
    OS_DISK=$(az vm show --name "$DB_VM_NAME" --resource-group "$RESOURCE_GROUP" \
      --query "storageProfile.osDisk.managedDisk.id" --output tsv 2>/dev/null) || true

    # Delete VM (does not automatically delete disks or NICs)
    az vm delete --name "$DB_VM_NAME" --resource-group "$RESOURCE_GROUP" \
      --yes --force-deletion true --output none 2>/dev/null
    ok "VM deleted"

    # Delete data disks
    for disk_id in $DATA_DISKS; do
      info "Deleting data disk..."
      az disk delete --ids "$disk_id" --yes --output none 2>/dev/null || true
      ok "Data disk deleted"
    done

    # Delete OS disk
    if [[ -n "$OS_DISK" ]]; then
      az disk delete --ids "$OS_DISK" --yes --output none 2>/dev/null || true
      ok "OS disk deleted"
    fi

    # Delete NIC
    NIC_NAME="${DB_VM_NAME}VMNic"
    az network nic delete --name "$NIC_NAME" --resource-group "$RESOURCE_GROUP" \
      --output none 2>/dev/null || true

    # Delete public IP
    try_delete "public IP '${DB_VM_NAME}-pip'" \
      az network public-ip delete --name "${DB_VM_NAME}-pip" --resource-group "$RESOURCE_GROUP"
  else
    skip "VM '$DB_VM_NAME' not found"
  fi

  # ── 5. Azure OpenAI ──────────────────────────────────────────
  try_delete "Azure OpenAI '$OPENAI_NAME'" \
    az cognitiveservices account delete --name "$OPENAI_NAME" --resource-group "$RESOURCE_GROUP"

  # Purge the soft-deleted resource (Azure OpenAI uses soft-delete)
  info "Purging soft-deleted Azure OpenAI resource..."
  az cognitiveservices account purge \
    --name "$OPENAI_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$(az group show --name "$RESOURCE_GROUP" --query location --output tsv 2>/dev/null || echo eastus)" \
    --output none 2>/dev/null || true

  # ── 6. Key Vault ─────────────────────────────────────────────
  info "Deleting Key Vault '$KV_NAME'..."
  if az keyvault show --name "$KV_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
    az keyvault delete --name "$KV_NAME" --resource-group "$RESOURCE_GROUP" --output none 2>/dev/null
    ok "Key Vault deleted"
    # Purge the soft-deleted vault
    info "Purging soft-deleted Key Vault..."
    az keyvault purge --name "$KV_NAME" --output none 2>/dev/null || true
    ok "Key Vault purged"
  else
    skip "Key Vault '$KV_NAME' not found"
  fi

  # ── 7. Container Registry ────────────────────────────────────
  try_delete "ACR '$ACR_NAME'" \
    az acr delete --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" --yes

  # ── 8. Managed Identity ──────────────────────────────────────
  try_delete "Managed Identity '$IDENTITY_NAME'" \
    az identity delete --name "$IDENTITY_NAME" --resource-group "$RESOURCE_GROUP"

  # ── 9. NSG ───────────────────────────────────────────────────
  # Must detach from subnet first
  az network vnet subnet update \
    --name "vm-subnet" \
    --vnet-name "$VNET_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --remove networkSecurityGroup \
    --output none 2>/dev/null || true

  try_delete "NSG '$NSG_NAME'" \
    az network nsg delete --name "$NSG_NAME" --resource-group "$RESOURCE_GROUP"

  # ── 10. VNet ─────────────────────────────────────────────────
  try_delete "VNet '$VNET_NAME'" \
    az network vnet delete --name "$VNET_NAME" --resource-group "$RESOURCE_GROUP"

  echo ""
  info "Resource group '$RESOURCE_GROUP' was NOT deleted."
  info "To delete it: az group delete --name $RESOURCE_GROUP --yes"
  echo ""
  ok "Cleanup complete. All CPQAI resources have been removed from resource group '$RESOURCE_GROUP'."
  echo ""
  exit 0
fi

# ══════════════════════════════════════════════════════════════════
# Default mode — delete the entire resource group
# ══════════════════════════════════════════════════════════════════
echo ""
warn "This will permanently delete the entire resource group '$RESOURCE_GROUP'"
warn "and ALL resources within it."
echo ""
echo "  TIP: To delete resources individually (keeping the resource group),"
echo "       run: ./deploy/azure/cleanup.sh --manual"
echo ""

if [[ "$AUTO_YES" != "true" ]]; then
  read -rp "Are you sure? Type 'delete' to confirm: " CONFIRM
  if [[ "$CONFIRM" != "delete" ]]; then
    err "Aborted."
    exit 1
  fi
fi

info "Deleting resource group '$RESOURCE_GROUP' (this may take a few minutes)..."
if az group delete --name "$RESOURCE_GROUP" --yes --output none 2>/dev/null; then
  ok "Resource group '$RESOURCE_GROUP' deleted"
else
  err "Failed to delete resource group '$RESOURCE_GROUP'. It may not exist or you lack permissions."
  exit 1
fi

echo ""
ok "Cleanup complete. Resource group '$RESOURCE_GROUP' and all its resources have been removed."
echo ""
