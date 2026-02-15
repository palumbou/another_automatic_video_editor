#!/usr/bin/env bash
set -euo pipefail

# Another Automatic Video Editor
# A single-script wrapper around CloudFormation + Step Functions to produce a YouTube-ready aftermovie.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -----------------------
# Utilities
# -----------------------
log()  { echo -e "[another-automatic-video-editor] $*"; }
warn() { echo -e "[another-automatic-video-editor][WARN] $*" >&2; }
die()  { echo -e "[another-automatic-video-editor][ERROR] $*" >&2; exit 1; }

has() { command -v "$1" >/dev/null 2>&1; }

# -----------------------
# NixOS auto-bootstrap
# -----------------------
maybe_enter_nix_shell() {
  local missing=()
  for bin in aws jq python3; do
    if ! has "$bin"; then
      missing+=("$bin")
    fi
  done

  if [ ${#missing[@]} -eq 0 ]; then
    return 0
  fi

  # If we're already inside nix-shell, don't recurse forever.
  if [ "${IN_NIX_SHELL:-}" = "1" ]; then
    die "Missing dependencies even inside nix-shell: ${missing[*]}"
  fi

  # Only do this if nix is available.
  if has nix-shell; then
    log "Missing dependencies: ${missing[*]}"
    log "Detected nix-shell. Re-executing inside a temporary nix-shell..."

    # We prefer awscli2.
    # Note: keep the package list minimal.
    exec nix-shell -p awscli2 jq python3 --run "IN_NIX_SHELL=1 bash \"$0\" ${*:-}"
  fi

  if has nix; then
    log "Missing dependencies: ${missing[*]}"
    log "Detected nix. Re-executing inside a temporary nix shell..."

    # nix shell (flakes) form. Works on many modern Nix installs.
    exec nix shell nixpkgs#awscli2 nixpkgs#jq nixpkgs#python3 -c bash "$0" ${*:-}
  fi

  die "Missing required tools: ${missing[*]}. Install them or run on NixOS with nix-shell available."
}

# -----------------------
# Defaults
# -----------------------
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-eu-west-1}}"
STACK_NAME=""
YES="false"

# Bedrock defaults
BEDROCK_MODEL_ID="us.amazon.nova-lite-v1:0"
BEDROCK_REGION="" # empty => stack region

# AI behavior
SKIP_AI_CHECK="false"
REQUIRE_AI="false"

# Worker sizing defaults (match template defaults)
CPU="2048"
MEMORY="4096"
EPHEMERAL_GIB="50"
ENABLE_TRANSCRIBE="true"

# Job
JOB_DIR=""
OUTPUT_DIR="./another_automatic_video_editor_output"

# -----------------------
# Help
# -----------------------
usage() {
  cat <<'EOF'
Another Automatic Video Editor

Usage:
  ./another_automatic_video_editor.sh <command> [options]

Commands:
  create        Deploy (or update) the CloudFormation stack
  status        Show stack status + important outputs
  run           Upload a job folder and run the Step Functions workflow
  delete        Delete the CloudFormation stack
  help          Show this help

Common Options:
  --region <region>                 AWS region (default: env or eu-west-1)
  --name <stack-name>               CloudFormation stack name (default: auto-generated)
  --yes                             Do not prompt for confirmation (delete)

Create Options:
  --bedrock-model-id <model-id>     Bedrock model ID (must be enabled in your account/region)
  --bedrock-region <region>         Override Bedrock region (default: stack region)
  --cpu <cpu-units>                 Fargate CPU units (1024,2048,4096,8192,16384)
  --memory <MiB>                    Fargate memory (MiB)
  --ephemeral-gib <GiB>             Fargate ephemeral storage (20..200)
  --enable-transcribe <true|false>  Use Amazon Transcribe (default: true)

Run Options:
  --job-dir <path>                  Folder containing manifest.json + media/
  --out-dir <path>                  Local output folder (default: ./another_automatic_video_editor_output)
  --skip-ai-check                   Skip AI connectivity test (use fallback if AI fails)
  --require-ai                      Fail if AI is not available (no fallback)

Examples:
  # Create infra
  ./another_automatic_video_editor.sh create \
    --region eu-west-1 \
    --bedrock-model-id amazon.nova-lite-v1:0

  # Run a job
  ./another_automatic_video_editor.sh run --job-dir ./my-job

  # Delete everything
  ./another_automatic_video_editor.sh delete --yes
EOF
}

# -----------------------
# Arg parsing (simple)
# -----------------------
# Preserve original args for nix-shell re-exec
ORIG_ARGS=("$@")
maybe_enter_nix_shell "${ORIG_ARGS[@]}"

COMMAND="${1:-help}"
shift || true

while [ $# -gt 0 ]; do
  case "$1" in
    --region) REGION="${2:-}"; shift 2;;
    --name) STACK_NAME="${2:-}"; shift 2;;
    --yes) YES="true"; shift 1;;

    --bedrock-model-id) BEDROCK_MODEL_ID="${2:-}"; shift 2;;
    --bedrock-region) BEDROCK_REGION="${2:-}"; shift 2;;
    --cpu) CPU="${2:-}"; shift 2;;
    --memory) MEMORY="${2:-}"; shift 2;;
    --ephemeral-gib) EPHEMERAL_GIB="${2:-}"; shift 2;;
    --enable-transcribe) ENABLE_TRANSCRIBE="${2:-}"; shift 2;;

    --job-dir) JOB_DIR="${2:-}"; shift 2;;
    --out-dir) OUTPUT_DIR="${2:-}"; shift 2;;
    --skip-ai-check) SKIP_AI_CHECK="true"; shift 1;;
    --require-ai) REQUIRE_AI="true"; shift 1;;

    -h|--help|help) COMMAND="help"; shift 1;;
    *)
      die "Unknown argument: $1 (try: ./another_automatic_video_editor.sh help)"
      ;;
  esac
done


# -----------------------
# AWS helpers
# -----------------------
aws_ok() {
  aws sts get-caller-identity --region "$REGION" >/dev/null 2>&1
}

require_aws() {
  aws_ok || die "AWS credentials not configured or not working for region $REGION. Try: aws configure"
}

gen_stack_name() {
  # Deterministic-ish: another-automatic-video-editor-YYYYMMDD-HHMMSS-rand
  python3 - <<'PY'
import datetime, secrets
ts = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
print(f"another-automatic-video-editor-{ts}-{secrets.token_hex(2)}")
PY
}

stack_exists() {
  local name="$1"
  aws cloudformation describe-stacks --stack-name "$name" --region "$REGION" >/dev/null 2>&1
}

get_stack_outputs_json() {
  local name="$1"
  aws cloudformation describe-stacks \
    --stack-name "$name" \
    --region "$REGION" \
    --query 'Stacks[0].Outputs' \
    --output json
}

get_output_value() {
  local outputs_json="$1"
  local key="$2"
  echo "$outputs_json" | jq -r ".[] | select(.OutputKey==\"$key\") | .OutputValue" | head -n1
}

sync_app_code() {
  local input_bucket="$1"
  log "Uploading src/ to s3://$input_bucket/app/ ..."
  aws s3 sync "$SCRIPT_DIR/src" "s3://$input_bucket/app/src" --region "$REGION" --delete >/dev/null
  aws s3 sync "$SCRIPT_DIR/app" "s3://$input_bucket/app" --region "$REGION" --delete >/dev/null
}

# -----------------------
# Commands
# -----------------------
cmd_create() {
  require_aws

  if [ -z "$STACK_NAME" ]; then
    STACK_NAME="$(gen_stack_name)"
    log "Generated stack name: $STACK_NAME"
  fi

  log "Deploying stack: $STACK_NAME (region: $REGION)"
  aws cloudformation deploy \
    --region "$REGION" \
    --stack-name "$STACK_NAME" \
    --template-file "$SCRIPT_DIR/infrastructure/template.yaml" \
    --capabilities CAPABILITY_NAMED_IAM \
    --parameter-overrides \
      BedrockModelId="$BEDROCK_MODEL_ID" \
      BedrockRegion="$BEDROCK_REGION" \
      FargateCpu="$CPU" \
      FargateMemory="$MEMORY" \
      EphemeralStorageGiB="$EPHEMERAL_GIB" \
      EnableTranscribe="$ENABLE_TRANSCRIBE" \
    >/dev/null

  log "✓ Stack deployed."
  cmd_status
}

cmd_status() {
  require_aws

  if [ -z "$STACK_NAME" ]; then
    # Try to find the most recent stack that starts with another-automatic-video-editor-
    local candidate
    candidate="$(aws cloudformation list-stacks --region "$REGION" \
      --query 'StackSummaries[?starts_with(StackName, `another-automatic-video-editor-`) && StackStatus!=`DELETE_COMPLETE`]|sort_by(@,&CreationTime)[-1].StackName' \
      --output text 2>/dev/null || true)"

    if [ -n "$candidate" ] && [ "$candidate" != "None" ]; then
      STACK_NAME="$candidate"
      log "Auto-selected stack: $STACK_NAME"
    else
      die "No --name provided and no existing stack found in region $REGION."
    fi
  fi

  if ! stack_exists "$STACK_NAME"; then
    die "Stack not found: $STACK_NAME (region: $REGION)"
  fi

  local outputs_json
  outputs_json="$(get_stack_outputs_json "$STACK_NAME")"

  local in_bucket out_bucket sfn_arn log_group
  in_bucket="$(get_output_value "$outputs_json" "InputBucketName")"
  out_bucket="$(get_output_value "$outputs_json" "OutputBucketName")"
  sfn_arn="$(get_output_value "$outputs_json" "StateMachineArn")"
  log_group="$(get_output_value "$outputs_json" "WorkerLogGroupName")"

  cat <<EOF
Stack:      $STACK_NAME
Region:     $REGION

InputBucket:  $in_bucket
OutputBucket: $out_bucket
StateMachine: $sfn_arn
LogGroup:     $log_group
EOF
}

cmd_delete() {
  require_aws

  if [ -z "$STACK_NAME" ]; then
    # Try auto-select like status
    local candidate
    candidate="$(aws cloudformation list-stacks --region "$REGION" \
      --query 'StackSummaries[?starts_with(StackName, `another-automatic-video-editor-`) && StackStatus!=`DELETE_COMPLETE`]|sort_by(@,&CreationTime)[-1].StackName' \
      --output text 2>/dev/null || true)"

    if [ -n "$candidate" ] && [ "$candidate" != "None" ]; then
      STACK_NAME="$candidate"
      log "Auto-selected stack: $STACK_NAME"
    else
      die "No --name provided and no existing stack found in region $REGION."
    fi
  fi

  if ! stack_exists "$STACK_NAME"; then
    die "Stack not found: $STACK_NAME (region: $REGION)"
  fi

  if [ "$YES" != "true" ]; then
    echo "About to delete stack '$STACK_NAME' in region '$REGION'. This will delete buckets and all resources."
    read -r -p "Type 'yes' to confirm: " confirm
    if [ "$confirm" != "yes" ]; then
      die "Aborted."
    fi
  fi

  # Get bucket names before deletion
  local outputs_json
  outputs_json="$(get_stack_outputs_json "$STACK_NAME")" || true
  local in_bucket out_bucket
  in_bucket="$(get_output_value "$outputs_json" "InputBucketName")" || true
  out_bucket="$(get_output_value "$outputs_json" "OutputBucketName")" || true

  # Empty buckets before stack deletion (CloudFormation can't delete non-empty buckets)
  if [ -n "$in_bucket" ]; then
    log "Emptying input bucket: $in_bucket ..."
    aws s3 rm "s3://$in_bucket" --recursive --region "$REGION" 2>/dev/null || true
  fi
  if [ -n "$out_bucket" ]; then
    log "Emptying output bucket: $out_bucket ..."
    aws s3 rm "s3://$out_bucket" --recursive --region "$REGION" 2>/dev/null || true
  fi

  log "Deleting stack: $STACK_NAME"
  aws cloudformation delete-stack --stack-name "$STACK_NAME" --region "$REGION"

  log "Waiting for deletion to complete..."
  aws cloudformation wait stack-delete-complete --stack-name "$STACK_NAME" --region "$REGION" || true

  log "✓ Stack deleted."
}

cmd_run() {
  require_aws

  if [ -z "$JOB_DIR" ]; then
    die "--job-dir <path> is required."
  fi
  if [ ! -d "$JOB_DIR" ]; then
    die "Job dir not found: $JOB_DIR"
  fi
  if [ ! -f "$JOB_DIR/manifest.json" ]; then
    die "Missing manifest.json in job dir: $JOB_DIR"
  fi
  if [ ! -d "$JOB_DIR/media" ]; then
    die "Missing media/ directory in job dir: $JOB_DIR"
  fi

  # Resolve stack
  if [ -z "$STACK_NAME" ]; then
    # Try auto-select like status
    cmd_status >/dev/null
  else
    cmd_status >/dev/null
  fi

  local outputs_json
  outputs_json="$(get_stack_outputs_json "$STACK_NAME")"
  local in_bucket out_bucket sfn_arn
  in_bucket="$(get_output_value "$outputs_json" "InputBucketName")"
  out_bucket="$(get_output_value "$outputs_json" "OutputBucketName")"
  sfn_arn="$(get_output_value "$outputs_json" "StateMachineArn")"

  # Upload app code
  sync_app_code "$in_bucket"

  # Create job id
  local job_id
  job_id="$(python3 - <<'PY'
import datetime, secrets
ts = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
print(f"job-{ts}-{secrets.token_hex(2)}")
PY
)"
  log "Job ID: $job_id"

  log "Uploading job folder to s3://$in_bucket/jobs/$job_id/ ..."
  aws s3 sync "$JOB_DIR" "s3://$in_bucket/jobs/$job_id" \
    --region "$REGION" \
    --exclude ".git/*" --exclude ".DS_Store" \
    >/dev/null

  # Prepare Step Functions input
  # Determine correct inference profile prefix based on region
  local bedrock_model="$BEDROCK_MODEL_ID"
  if [[ "$bedrock_model" == us.amazon.nova-* ]] && [[ ! "$REGION" == us-* ]]; then
    # Auto-switch to EU profile for non-US regions
    bedrock_model="${bedrock_model/us.amazon/eu.amazon}"
    log "Auto-switched Bedrock model to: $bedrock_model (for region $REGION)"
  elif [[ "$bedrock_model" == eu.amazon.nova-* ]] && [[ "$REGION" == us-* ]]; then
    # Auto-switch to US profile for US regions
    bedrock_model="${bedrock_model/eu.amazon/us.amazon}"
    log "Auto-switched Bedrock model to: $bedrock_model (for region $REGION)"
  fi

  mkdir -p "$OUTPUT_DIR"
  local input_json
  input_json="$(python3 - <<PY
import json
payload = {
  "jobId": "$job_id",
  "bedrockModelId": "$bedrock_model",
  "bedrockRegion": "${BEDROCK_REGION:-$REGION}",
  "enableTranscribe": "$ENABLE_TRANSCRIBE",
  "skipAiCheck": "$SKIP_AI_CHECK",
  "requireAi": "$REQUIRE_AI"
}
print(json.dumps(payload))
PY
)"

  log "Starting Step Functions execution..."
  local exec_arn
  exec_arn="$(aws stepfunctions start-execution \
    --region "$REGION" \
    --state-machine-arn "$sfn_arn" \
    --input "$input_json" \
    --query 'executionArn' \
    --output text)"

  log "Execution ARN: $exec_arn"
  log "Waiting for completion..."

  while true; do
    local status
    status="$(aws stepfunctions describe-execution \
      --region "$REGION" \
      --execution-arn "$exec_arn" \
      --query 'status' \
      --output text)"

    case "$status" in
      RUNNING)
        sleep 15
        ;;
      SUCCEEDED)
        log "✓ Job succeeded."
        break
        ;;
      FAILED|TIMED_OUT|ABORTED)
        warn "Job ended with status: $status"
        warn "Check CloudWatch Logs (log group from 'status') for details."
        break
        ;;
      *)
        warn "Unknown status: $status"
        sleep 10
        ;;
    esac
  done

  log "Downloading outputs from s3://$out_bucket/jobs/$job_id/output/ ..."
  local local_out="$OUTPUT_DIR/$job_id"
  mkdir -p "$local_out"
  aws s3 sync "s3://$out_bucket/jobs/$job_id/output" "$local_out" --region "$REGION" >/dev/null || true

  cat <<EOF

Local output folder:
  $local_out

Key files:
  - $local_out/final.mp4
  - $local_out/chapters.txt
  - $local_out/description.md

EOF
}

case "$COMMAND" in
  help|-h|--help) usage ;;
  create) cmd_create ;;
  status) cmd_status ;;
  run) cmd_run ;;
  delete) cmd_delete ;;
  *)
    die "Unknown command: $COMMAND (try: ./another_automatic_video_editor.sh help)"
    ;;
esac
