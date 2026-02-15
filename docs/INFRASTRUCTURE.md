# Infrastructure Guide

Complete guide to AWS infrastructure for Another Automatic Video Editor.

> **Available languages**: [English (current)](INFRASTRUCTURE.md) | [Italiano](INFRASTRUCTURE.it.md)

## Architecture

The system uses a serverless architecture on AWS:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   S3 Input      │────▶│  Step Functions  │────▶│   ECS Fargate   │
│   Bucket        │     │  State Machine   │     │   (FFmpeg)      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                         │
                        ┌──────────────────┐             │
                        │   S3 Output      │◀────────────┘
                        │   Bucket         │
                        └──────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
┌───────────────┐    ┌───────────────────┐    ┌───────────────────┐
│  Rekognition  │    │     Bedrock       │    │    Transcribe     │
│  (Labels)     │    │  (AI Planning)    │    │  (Speech-to-Text) │
└───────────────┘    └───────────────────┘    └───────────────────┘
```

## AWS Components

### Networking

- **VPC**: Dedicated VPC with configurable CIDR (default: 10.20.0.0/16)
- **Subnets**: Two public subnets in different AZs for high availability
- **Internet Gateway**: For internet access from Fargate tasks
- **Security Group**: Outbound traffic only for AWS APIs

### Storage

- **Input Bucket**: Job uploads and application code
- **Output Bucket**: Rendered videos and metadata

### Compute

- **ECS Cluster**: Fargate cluster for rendering tasks
- **Task Definition**: Python container with FFmpeg and ImageMagick
- **Step Functions**: Rendering workflow orchestration

### AI Services

- **Amazon Rekognition**: Image and video frame analysis
- **Amazon Bedrock**: Video plan and SEO generation (with connectivity check)
- **Amazon Transcribe**: Speech-to-text (optional)

### Monitoring

- **CloudWatch Logs**: Centralized logs with configurable retention
- **Cost Tags**: All resources tagged with `CostCenter`

## Deployment

### Quick Start

```bash
# Deploy with defaults
./another_automatic_video_editor.sh create --region eu-west-1

# Check status
./another_automatic_video_editor.sh status

# Run a job
./another_automatic_video_editor.sh run --job-dir ./examples/job
```

### CloudFormation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VpcCidr` | 10.20.0.0/16 | CIDR block for VPC |
| `FargateCpu` | 2048 | CPU units (1024-16384) |
| `FargateMemory` | 4096 | Memory in MiB |
| `EphemeralStorageGiB` | 50 | Ephemeral storage (20-200) |
| `BedrockModelId` | us.amazon.nova-lite-v1:0 | Bedrock model or inference profile |
| `BedrockRegion` | (stack region) | Bedrock region override |
| `EnableTranscribe` | true | Enable speech-to-text |
| `LogRetentionDays` | 14 | CloudWatch log retention |

### Bedrock Model Configuration

The default model uses an inference profile (`us.amazon.nova-lite-v1:0`) for cross-region inference. Available profiles:

| Region | Inference Profile |
|--------|-------------------|
| US regions | `us.amazon.nova-lite-v1:0` |
| EU regions | `eu.amazon.nova-lite-v1:0` |

You can also use direct model IDs if you have on-demand access enabled:
- `anthropic.claude-3-haiku-20240307-v1:0`
- `anthropic.claude-3-sonnet-20240229-v1:0`

## AI Behavior

### Connectivity Check

Before processing media, the runner tests Bedrock connectivity:

1. **Test passes**: AI planning proceeds normally
2. **Test fails (interactive)**: User is prompted to continue with fallback or abort
3. **Test fails (non-interactive)**: Automatically uses fallback plan

### Run Options

| Option | Behavior |
|--------|----------|
| (default) | Test AI, prompt user if unavailable |
| `--skip-ai-check` | Skip test, silently fallback if AI fails |
| `--require-ai` | Fail immediately if AI unavailable |

### Fallback Plans

When AI is unavailable:
- **aftermovie mode**: Uses heuristic plan (chronological, scaled to target duration)
- **longform mode**: Uses deterministic plan (includes all media, chapter-based)

## Costs

### Estimate for typical job (3 min video, 50 media files)

| Service | Estimated Cost |
|---------|----------------|
| ECS Fargate (10 min) | ~$0.05 |
| S3 Storage | ~$0.01 |
| Rekognition | ~$0.10 |
| Bedrock | ~$0.02 |
| Transcribe | ~$0.05 |
| **Total** | **~$0.23/job** |

### Cost Optimization

- Disable Transcribe if not needed (`--enable-transcribe false`)
- Use cheaper Bedrock models
- Reduce CPU/memory for simple jobs
- Use `--skip-ai-check` to avoid AI test overhead

## Cleanup

The delete command automatically empties S3 buckets before deletion:

```bash
# Delete all resources (with confirmation)
./another_automatic_video_editor.sh delete

# Delete without confirmation
./another_automatic_video_editor.sh delete --yes
```

The cleanup process:
1. Retrieves bucket names from stack outputs
2. Empties input bucket (`aws s3 rm --recursive`)
3. Empties output bucket (`aws s3 rm --recursive`)
4. Deletes CloudFormation stack
5. Waits for deletion to complete

## Troubleshooting

### AI not available

If you see "Bedrock AI is NOT available":

1. **Check model access**: Ensure the model is enabled in your AWS account (Bedrock console → Model access)
2. **Use inference profile**: Change to `us.amazon.nova-lite-v1:0` or `eu.amazon.nova-lite-v1:0`
3. **Check IAM permissions**: Ensure `bedrock:Converse` is allowed
4. **Try different region**: Use `--bedrock-region` to specify a region with model access

### Fargate task fails

1. Check CloudWatch logs: `/aws/ecs/<stack-name>/worker`
2. Verify manifest.json is valid JSON
3. Check media files are in supported formats (jpg, png, mp4, mov, etc.)
4. Check `render.log` in output for detailed errors

### Out of memory

Increase Fargate memory:
```bash
./another_automatic_video_editor.sh create --memory 8192
```

### Stack deletion fails

If deletion hangs, buckets may not be empty. The script now handles this automatically, but if issues persist:

```bash
# Manually empty buckets
aws s3 rm s3://<input-bucket> --recursive
aws s3 rm s3://<output-bucket> --recursive

# Retry deletion
./another_automatic_video_editor.sh delete --yes
```

### Music volume too low

Increase `style.music.volume` in manifest.json (recommended: 0.40-0.50):

```json
"music": {
  "enabled": true,
  "duck": true,
  "volume": 0.45
}
```
