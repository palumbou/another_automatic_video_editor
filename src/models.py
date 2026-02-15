"""Data models for Another Automatic Video Editor."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VideoInfo:
    """Information about a video file."""
    
    duration_s: float
    width: int
    height: int
    has_audio: bool


@dataclass
class MediaItem:
    """A media item (image or video) in the catalog."""
    
    id: str
    type: str  # "image" or "video"
    filename: str
    local_path: str
    size_bytes: int
    labels: list[str] = field(default_factory=list)
    exif_datetime: str | None = None
    # Video-specific fields
    duration_s: float = 0.0
    width: int = 0
    height: int = 0
    has_audio: bool = False
    transcript_excerpt: str | None = None
    best_moments: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Segment:
    """A segment in the video timeline."""
    
    source_id: str
    type: str  # "image" or "video"
    duration: float
    chapter: int
    caption: str = ""
    in_seconds: float = 0.0  # for video only


@dataclass
class Chapter:
    """A chapter in the video."""
    
    title: str
    start_seconds: float = 0.0


@dataclass
class VideoPlan:
    """The complete plan for video generation."""
    
    video_title: str
    youtube_description: str
    chapters: list[Chapter]
    segments: list[Segment]
    notes: str = ""


@dataclass
class Catalog:
    """The media catalog with all analyzed items."""
    
    generated_at: str
    items: list[MediaItem]
    counts: dict[str, int]
    best_moments_enabled: bool = False


@dataclass
class RenderResult:
    """Result of the rendering process."""
    
    output_path: Path
    chapters_txt: str
    description_md: str
    title_txt: str
    total_duration_s: float
    music_path: Path | None = None
    music_generated: bool = False
