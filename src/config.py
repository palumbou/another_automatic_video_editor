"""Configuration management for Another Automatic Video Editor."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AWSConfig:
    """Configuration for AWS services."""
    
    region: str = "eu-west-1"
    bedrock_model_id: str = "amazon.nova-lite-v1:0"
    bedrock_region: str = ""
    enable_transcribe: bool = True


@dataclass
class RenderConfig:
    """Configuration for video rendering."""
    
    fps: int = 30
    target_width: int = 1920
    target_height: int = 1080
    fade_seconds: float = 0.5
    captions_enabled: bool = True


@dataclass
class MusicConfig:
    """Configuration for music/audio."""
    
    enabled: bool = True
    duck: bool = True
    volume: float = 0.20


@dataclass
class StyleConfig:
    """Style configuration from manifest."""
    
    mode: str = "aftermovie"
    target_duration_seconds: int = 180
    render: RenderConfig = field(default_factory=RenderConfig)
    music: MusicConfig = field(default_factory=MusicConfig)


class Config:
    """Main configuration manager."""

    def __init__(self):
        """Initialize configuration from environment variables."""
        self.input_bucket = os.getenv("INPUT_BUCKET", "")
        self.output_bucket = os.getenv("OUTPUT_BUCKET", "")
        self.app_prefix = os.getenv("APP_PREFIX", "app")
        self.bedrock_model_id = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")
        self.bedrock_region = os.getenv("BEDROCK_REGION", os.getenv("AWS_REGION", "eu-west-1"))
        self.enable_transcribe = os.getenv("ENABLE_TRANSCRIBE", "true").lower() == "true"
        self.aws_region = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "eu-west-1"))
        self.log_level = os.getenv("LOG_LEVEL", "INFO")

    def get_aws_config(self) -> AWSConfig:
        """Get AWS configuration."""
        return AWSConfig(
            region=self.aws_region,
            bedrock_model_id=self.bedrock_model_id,
            bedrock_region=self.bedrock_region or self.aws_region,
            enable_transcribe=self.enable_transcribe,
        )

    def get_style_config(self, manifest: dict[str, Any] | None = None) -> StyleConfig:
        """Get style configuration from manifest or defaults."""
        if not manifest:
            return StyleConfig()
        
        style = manifest.get("style", {}) or {}
        res = style.get("resolution", {}) or {}
        music = style.get("music", {}) or {}
        captions = style.get("captions", {}) or {}
        
        return StyleConfig(
            mode=str(style.get("mode", "aftermovie")).strip().lower(),
            target_duration_seconds=int(style.get("target_duration_seconds", 180)),
            render=RenderConfig(
                fps=int(style.get("fps", 30)),
                target_width=int(res.get("w", 1920)),
                target_height=int(res.get("h", 1080)),
                fade_seconds=float(style.get("fade_seconds", 0.5)),
                captions_enabled=bool(captions.get("enabled", True)),
            ),
            music=MusicConfig(
                enabled=bool(music.get("enabled", True)),
                duck=bool(music.get("duck", True)),
                volume=float(music.get("volume", 0.20)),
            ),
        )

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if not self.input_bucket:
            errors.append("INPUT_BUCKET environment variable is required")
        if not self.output_bucket:
            errors.append("OUTPUT_BUCKET environment variable is required")
        
        return errors
