# Deployment Scripts

This directory contains scripts for deploying and managing the Another Automatic Video Editor infrastructure.

> **Available languages**: [English (current)](README.md) | [Italiano](README.it.md)

## deploy.sh

Main deployment script for CloudFormation stack management.

### Prerequisites

- AWS CLI installed and configured
- Valid AWS credentials with appropriate permissions
- jq (for JSON parsing)

### Usage

```bash
# Deploy with defaults
./scripts/deploy.sh

# Deploy to specific region
./scripts/deploy.sh --region eu-west-1

# Deploy with custom Bedrock model
./scripts/deploy.sh --bedrock-model-id anthropic.claude-3-sonnet-20240229-v1:0

# Check stack status
./scripts/deploy.sh --status

# Delete all resources
./scripts/deploy.sh --cleanup --yes
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-s, --stack-name` | CloudFormation stack name | `another-automatic-video-editor` |
| `-r, --region` | AWS region | `eu-west-1` |
| `--bedrock-model-id` | Bedrock model ID | `amazon.nova-lite-v1:0` |
| `--bedrock-region` | Override Bedrock region | Stack region |
| `--cpu` | Fargate CPU units | `2048` |
| `--memory` | Fargate memory (MiB) | `4096` |
| `--ephemeral-gib` | Ephemeral storage (GiB) | `50` |
| `--enable-transcribe` | Enable Transcribe | `true` |
| `--cleanup` | Delete all resources | - |
| `--status` | Show stack status | - |
| `--dry-run` | Preview changes | - |
| `-y, --yes` | Skip confirmations | - |

## Cost Considerations

All AWS resources are tagged with `CostCenter` for cost tracking. Monitor costs in AWS Cost Explorer by filtering on this tag.
