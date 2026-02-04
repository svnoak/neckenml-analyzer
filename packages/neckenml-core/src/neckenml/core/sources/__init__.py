"""Audio source interfaces and implementations."""

from neckenml.core.sources.base import AudioSource
from neckenml.core.sources.file_source import FileAudioSource

__all__ = ["AudioSource", "FileAudioSource"]
