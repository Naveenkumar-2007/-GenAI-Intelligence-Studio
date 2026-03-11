"""
Video processing module.
"""

__all__ = ["VideoProcessor"]


def __getattr__(name):
    if name == "VideoProcessor":
        from .video_processor import VideoProcessor
        return VideoProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
