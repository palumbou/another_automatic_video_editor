# Another Automatic Video Editor

> **Available languages**: [English (current)](README.md) | [Italiano](README.it.md)

AWS mini-infrastructure (CloudFormation) + a single launcher script that turns a folder of **images + videos** into a **YouTube-ready MP4**, with:

- **AI-driven planning** (AWS Bedrock) for intelligent video sequencing
- **AI-driven analysis** (AWS Rekognition) of images / sampled video frames
- Optional **speech-to-text** (AWS Transcribe) to improve chapter titles / description
- Optional **"Best moments" extraction** (lightweight shot scoring) to build a highlights chapter
- **AI connectivity check** before processing with interactive fallback options
- Auto-generated:
  - `chapters.txt`
  - `description.md` (SEO-friendly, rigid template)
  - `title.txt`

You provide a `manifest.json` (outline / constraints / YouTube links). The system outputs a final `final.mp4` you can upload directly to YouTube.

> Default Bedrock model: **Amazon Nova Lite** (`us.amazon.nova-lite-v1:0` inference profile).

## Architecture

Serverless system on AWS with the following components:

### AWS Components
- **S3 Buckets**: Input (job uploads) and Output (rendered videos)
- **Step Functions**: Workflow orchestration
- **ECS Fargate**: Container-based video rendering with FFmpeg
- **Amazon Rekognition**: Image and video frame analysis
- **Amazon Bedrock**: AI-powered video planning and SEO generation
- **Amazon Transcribe**: Speech-to-text (optional)
- **CloudWatch**: Logging and monitoring

### Code Components
- `src/config.py`: Configuration management
- `src/logging_config.py`: Structured logging
- `src/aws_clients.py`: AWS client utilities
- `src/models.py`: Data models
- `src/utils.py`: Utility functions
- `app/runner.py`: Main rendering pipeline

## Project Structure

```
another_automatic_video_editor/
├── app/                    # Main application code
│   └── runner.py          # Video rendering pipeline
├── src/                    # Modular source code
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── logging_config.py  # Structured logging
│   ├── aws_clients.py     # AWS client utilities
│   ├── models.py          # Data models
│   └── utils.py           # Utility functions
├── infrastructure/         # CloudFormation templates
│   └── template.yaml
├── docs/                  # Documentation
│   └── INFRASTRUCTURE.md
├── examples/              # Example jobs
│   ├── job/
│   └── job_longform/
├── requirements.txt       # Python dependencies
└── another_automatic_video_editor.sh  # Main CLI
```

## Two Modes

Set `style.mode` in your `manifest.json`:

- `aftermovie` (default): short highlight-style video (e.g. 2–4 minutes)
- `longform`: **full-length** event video (e.g. 30–120+ minutes)

Both modes use AI (Bedrock) for planning when available, with automatic fallback to deterministic planning if AI is unavailable.

## Quick Start

### Prerequisites

- AWS CLI installed and configured
- Python 3.11+
- Valid AWS credentials with appropriate permissions
- Bedrock model access enabled (Amazon Nova Lite recommended)

### 1) Create the AWS stack

```bash
./another_automatic_video_editor.sh create \
  --region eu-west-1 \
  --bedrock-model-id us.amazon.nova-lite-v1:0
```

### 2) Run an example job

#### Example A — Aftermovie (short)

```bash
./another_automatic_video_editor.sh run \
  --job-dir ./examples/job
```

#### Example B — Longform (full video, intro first)

Put your media into:

- `./examples/job_longform/media/`
  - `intro.jpg` or `intro.mp4` (first segment)
  - your full talk recordings (e.g. `talk1.mp4`, `talk2.mp4`)
  - photos

Then run:

```bash
./another_automatic_video_editor.sh run \
  --job-dir ./examples/job_longform
```

### 3) Outputs

The `run` command waits for completion and downloads results locally.

By default outputs go to `./another_automatic_video_editor_output/<JOB_ID>/`.

## CLI Reference

```bash
./another_automatic_video_editor.sh <command> [options]
```

### Commands

| Command | Description |
|---------|-------------|
| `create` | Deploy (or update) the CloudFormation stack |
| `status` | Show stack status and important outputs |
| `run` | Upload a job folder and run the workflow |
| `delete` | Delete the CloudFormation stack (empties buckets first) |
| `help` | Show help |

### Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--region <region>` | AWS region | `eu-west-1` |
| `--name <stack-name>` | CloudFormation stack name | auto-generated |
| `--yes` | Skip confirmation prompts | `false` |

### Create Options

| Option | Description | Default |
|--------|-------------|---------|
| `--bedrock-model-id <id>` | Bedrock model or inference profile | `us.amazon.nova-lite-v1:0` |
| `--bedrock-region <region>` | Override Bedrock region | stack region |
| `--cpu <units>` | Fargate CPU (1024-16384) | `2048` |
| `--memory <MiB>` | Fargate memory | `4096` |
| `--ephemeral-gib <GiB>` | Fargate storage (20-200) | `50` |
| `--enable-transcribe <bool>` | Enable Amazon Transcribe | `true` |

### Run Options

| Option | Description | Default |
|--------|-------------|---------|
| `--job-dir <path>` | Job folder with manifest.json + media/ | required |
| `--out-dir <path>` | Local output folder | `./another_automatic_video_editor_output` |
| `--skip-ai-check` | Skip AI connectivity test | `false` |
| `--require-ai` | Fail if AI unavailable (no fallback) | `false` |

### AI Behavior Options

The runner performs an AI connectivity check before processing media:

- **Default**: Tests Bedrock, prompts user if unavailable (continue with fallback or abort)
- `--skip-ai-check`: Skip the test, silently use fallback if AI fails later
- `--require-ai`: Fail immediately if AI is not available

## Manifest Configuration

### Intro (first segment)

Supports both images (jpg, png, etc.) and videos:

```json
"intro": {
  "file": "intro.jpg",
  "duration_seconds": 5,
  "caption": "AWS User Group Salerno"
}
```

### Music

```json
"music": {
  "enabled": true,
  "duck": true,
  "duck_amount": 0.15,
  "loop": true,
  "volume": 0.45
}
```

- `enabled`: Enable/disable background music
- `volume`: Base music volume (0.0-1.0, recommended: 0.40-0.50)
- `duck`: Automatically lower music when video has speech
- `duck_amount`: How much to lower during ducking (0.0-1.0, lower = more reduction, default: 0.15)
- `loop`: Loop music if shorter than video (default: true)

### Audio Normalization

```json
"style": {
  "normalize_audio": true
}
```

Automatically normalizes all video clips to -14 LUFS (YouTube standard), so videos with different volumes will have consistent levels.

### Best Moments (shot scoring)

```json
"best_moments": {
  "enabled": true,
  "insert_in_timeline": true,
  "max_clips_total": 12,
  "max_moments_per_video": 2,
  "clip_duration_seconds": 8,
  "samples_per_video": 10,
  "min_gap_seconds": 25
}
```

### SEO Override

```json
"seo": {
  "enabled": true,
  "cta": "Like + subscribe!",
  "hashtags": ["#AWS", "#Serverless", "#Meetup"]
}
```

### AI Configuration

```json
"ai": {
  "enabled": true,
  "enable_transcribe": true,
  "system_prompt": "",
  "task_prompt": "",
  "extra_instructions": ""
}
```

- `enabled`: Enable AI-powered planning (Bedrock)
- `enable_transcribe`: Enable speech-to-text for video transcripts
- `system_prompt`: Custom system prompt for AI (leave empty for default)
- `task_prompt`: Custom task description (leave empty for default)
- `extra_instructions`: Additional instructions for AI planning

### Fade Out

The video automatically includes a fade to black at the end (audio + video). Default duration is 3 seconds.

## Outputs

The output folder contains:

| File | Description |
|------|-------------|
| `{title}_final.mp4` | Final rendered video |
| `{title}_no_music.mp4` | Video without music track |
| `chapters.txt` | YouTube chapters (timestamps + titles) |
| `description.md` | SEO-friendly YouTube description |
| `title.txt` | Suggested YouTube title |
| `catalog.json` | Media catalog with analysis |
| `plan.json` | Video plan (AI or fallback) |
| `render_meta.json` | Render metadata |
| `best_moments_top.json` | Top scored clips (if enabled) |
| `render.log` | Processing log |

Video filenames use the project title from manifest as prefix.

## Documentation

- [Infrastructure Guide](docs/INFRASTRUCTURE.md) - Complete infrastructure setup ([Italiano](docs/INFRASTRUCTURE.it.md))

## Clean Up

The delete command automatically empties S3 buckets before deletion:

```bash
./another_automatic_video_editor.sh delete --yes
```

Without `--yes`, you'll be prompted to confirm.

## Troubleshooting

### AI not available

If you see "Bedrock AI is NOT available", check:
1. Model access is enabled in your AWS account
2. You're using the correct inference profile (e.g., `us.amazon.nova-lite-v1:0` for US, `eu.amazon.nova-lite-v1:0` for EU)
3. IAM permissions include `bedrock:Converse`

### Music volume too low

Increase `style.music.volume` in manifest (recommended: 0.40-0.50).

### Not all media used

For `aftermovie` mode, media is scaled to fit `target_duration_seconds`. Increase this value or use `longform` mode.

## License

CC BY-NC 4.0 — see [LICENSE](LICENSE).
