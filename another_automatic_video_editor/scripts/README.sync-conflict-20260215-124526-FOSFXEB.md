# Scripts

Automation scripts for Another Automatic Video Editor.

> **Available languages**: [English (current)](README.md) | [Italiano](README.it.md)

## Main Script

The main script `another_automatic_video_editor.sh` in the project root handles all operations:

```bash
# Deploy infrastructure
./another_automatic_video_editor.sh create --region eu-west-1

# Check status
./another_automatic_video_editor.sh status

# Run a job
./another_automatic_video_editor.sh run --job-dir ./examples/job

# Delete all resources
./another_automatic_video_editor.sh delete --yes
```

### NixOS Support

The script automatically detects if required tools (`aws`, `jq`, `python3`) are missing and, if running on NixOS, enters a temporary nix-shell with the required packages.

### Commands

| Command | Description |
|---------|-------------|
| `create` | Deploy or update the CloudFormation stack |
| `status` | Show stack status and outputs |
| `run` | Upload a job and run the rendering workflow |
| `delete` | Delete all AWS resources |
| `help` | Show help message |

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--region` | AWS region | `eu-west-1` |
| `--name` | CloudFormation stack name | auto-generated |
| `--bedrock-model-id` | Bedrock model ID | `amazon.nova-lite-v1:0` |
| `--bedrock-region` | Override Bedrock region | stack region |
| `--cpu` | Fargate CPU units (1024-16384) | `2048` |
| `--memory` | Fargate memory (MiB) | `4096` |
| `--ephemeral-gib` | Ephemeral storage (20-200 GiB) | `50` |
| `--enable-transcribe` | Enable speech-to-text | `true` |
| `--job-dir` | Job folder path (for `run`) | - |
| `--out-dir` | Local output folder | `./another_automatic_video_editor_output` |
| `--yes` | Skip confirmation prompts | - |
