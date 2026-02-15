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
- **Amazon Bedrock**: Video plan and SEO generation
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
| `BedrockModelId` | amazon.nova-lite-v1:0 | Bedrock model |
| `BedrockRegion` | (stack region) | Bedrock region override |
| `EnableTranscribe` | true | Enable speech-to-text |
| `LogRetentionDays` | 14 | CloudWatch log retention |

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

## Cleanup

```bash
# Delete all resources
./another_automatic_video_editor.sh delete --yes
```

⚠️ This deletes all S3 buckets and their contents.

## Troubleshooting

### Fargate task fails

1. Check CloudWatch logs: `/aws/ecs/<stack-name>/worker`
2. Verify manifest.json is valid
3. Check media files are in supported formats

### Bedrock not responding

1. Verify model is enabled in the region
2. Use `--bedrock-region` to specify a different region
3. Check IAM permissions

### Out of memory

Increase Fargate memory:
```bash
./another_automatic_video_editor.sh create --memory 8192
```
