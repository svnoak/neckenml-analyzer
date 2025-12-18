"""File-based audio source implementation."""

import os
from .base import AudioSource


class FileAudioSource(AudioSource):
    """
    Simple file-based audio source that looks up audio files in a directory.

    This is the simplest implementation - it assumes audio files already exist
    on the local filesystem with predictable naming.
    """

    def __init__(self, audio_dir: str, file_extension: str = "mp3"):
        """
        Initialize file-based audio source.

        Args:
            audio_dir: Directory containing audio files
            file_extension: File extension to look for (default: mp3)
        """
        self.audio_dir = os.path.expanduser(audio_dir)
        self.file_extension = file_extension.lstrip(".")

        if not os.path.exists(self.audio_dir):
            raise ValueError(f"Audio directory does not exist: {self.audio_dir}")

    def fetch_audio(self, track_id: str) -> str:
        """
        Fetch audio file by looking for {track_id}.{extension} in audio_dir.

        Args:
            track_id: Track identifier (used as filename without extension)

        Returns:
            str: Full path to the audio file

        Raises:
            FileNotFoundError: If audio file doesn't exist
        """
        # Try the specified extension first
        file_path = os.path.join(self.audio_dir, f"{track_id}.{self.file_extension}")

        if os.path.exists(file_path):
            return file_path

        # Try common audio extensions as fallback
        for ext in ["mp3", "wav", "flac", "m4a", "ogg"]:
            if ext == self.file_extension:
                continue  # Already tried this one
            fallback_path = os.path.join(self.audio_dir, f"{track_id}.{ext}")
            if os.path.exists(fallback_path):
                return fallback_path

        # Not found
        raise FileNotFoundError(
            f"Audio file not found: {file_path} "
            f"(also tried wav, flac, m4a, ogg extensions)"
        )

    def cleanup(self, file_path: str) -> None:
        """
        No cleanup needed for persistent files.

        Args:
            file_path: Path that was returned by fetch_audio()
        """
        pass  # Files are permanent, don't delete them
