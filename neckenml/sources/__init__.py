"""Audio source interfaces and implementations."""

from neckenml.sources.base import AudioSource
from neckenml.sources.file_source import FileAudioSource

__all__ = ["AudioSource", "FileAudioSource"]
