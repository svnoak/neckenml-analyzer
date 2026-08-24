"""Abstract base class for audio source implementations."""

from abc import ABC, abstractmethod


class AudioSource(ABC):
    """
    Abstract interface for audio acquisition.

    Implementations of this class provide audio files for analysis from various sources
    (local files, cloud storage, etc.).
    """

    @abstractmethod
    def fetch_audio(self, track_id: str) -> str:
        """
        Fetch audio file for the given track identifier.

        Args:
            track_id: Unique identifier for the track

        Returns:
            str: Local file path to the audio file (MP3, WAV, etc.)

        Raises:
            FileNotFoundError: If audio cannot be found
            Exception: For other acquisition errors
        """
        pass

    @abstractmethod
    def cleanup(self, file_path: str) -> None:
        """
        Optional cleanup of temporary files after analysis.

        Args:
            file_path: Path to the file that was returned by fetch_audio()

        Note:
            This is called after analysis completes, regardless of success/failure.
            Implementations should handle cleanup safely (check file exists, etc.).
        """
        pass
