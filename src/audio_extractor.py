"""
Audio Extraction Module
Extracts audio from video files for speech processing
"""

import os
import subprocess
from typing import Optional

# moviepy 2.x removed the legacy `moviepy.editor` shim, so the top-level
# package is the only supported import path now.
try:
    from moviepy import VideoFileClip
except ImportError:  # moviepy 1.x
    from moviepy.editor import VideoFileClip

try:
    from .ffmpeg_setup import find_ffmpeg
except ImportError:  # running as a standalone script
    from ffmpeg_setup import find_ffmpeg

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioExtractor:
    """
    Handles audio extraction from video files
    Supports multiple video formats: mp4, mkv, avi, webm
    """
    
    SUPPORTED_FORMATS = ['.mp4', '.mkv', '.avi', '.webm', '.mov']
    OUTPUT_FORMAT = 'wav'  # Best for speech recognition
    SAMPLE_RATE = 16000    # Whisper recommended rate
    
    def __init__(self, temp_dir: str = './temp'):
        """
        Initialize AudioExtractor
        
        Args:
            temp_dir: Directory for temporary files
        """
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)
    
    def extract_audio(self, video_path: str, output_path: Optional[str] = None) -> str:
        """
        Extract audio from video file
        
        Args:
            video_path: Path to input video file
            output_path: Optional custom output path
            
        Returns:
            Path to extracted audio file
            
        Raises:
            ValueError: If video format not supported
            FileNotFoundError: If video file doesn't exist
        """
        # Validate video file
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        file_ext = os.path.splitext(video_path)[1].lower()
        if file_ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {file_ext}")
        
        # Generate output path
        if output_path is None:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(self.temp_dir, f"{video_name}_audio.{self.OUTPUT_FORMAT}")
        
        logger.info(f"Extracting audio from: {video_path}")
        
        try:
            # Method 1: Using FFmpeg (faster and more reliable)
            self._extract_with_ffmpeg(video_path, output_path)
            
        except Exception as e:
            logger.warning(f"FFmpeg extraction failed: {e}. Trying MoviePy...")
            try:
                # Method 2: Fallback to MoviePy
                self._extract_with_moviepy(video_path, output_path)
            except Exception as e2:
                logger.error(f"Audio extraction failed: {e2}")
                raise
        
        logger.info(f"Audio extracted successfully: {output_path}")
        return output_path
    
    def _extract_with_ffmpeg(self, video_path: str, output_path: str):
        """Extract audio using FFmpeg (preferred method)"""
        ffmpeg = find_ffmpeg('ffmpeg')
        if ffmpeg is None:
            raise FileNotFoundError(
                "ffmpeg not found. Install it, or run: pip install imageio-ffmpeg"
            )

        command = [
            ffmpeg,
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM encoding
            '-ar', str(self.SAMPLE_RATE),  # Sample rate
            '-ac', '1',  # Mono channel
            '-y',  # Overwrite output
            output_path
        ]
        
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        
        # check=True already raises CalledProcessError on a non-zero exit, so
        # there is no separate returncode branch to handle here.
        logger.debug(f"ffmpeg wrote {result.stdout and len(result.stdout) or 0} bytes to stdout")
    
    def _extract_with_moviepy(self, video_path: str, output_path: str):
        """Extract audio using MoviePy (fallback method)"""
        video = VideoFileClip(video_path)
        audio = video.audio
        
        audio.write_audiofile(
            output_path,
            fps=self.SAMPLE_RATE,
            nbytes=2,
            codec='pcm_s16le'
        )
        
        video.close()
        audio.close()
    
    def get_audio_duration(self, audio_path: str) -> float:
        """
        Get duration of audio file in seconds

        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in seconds
        """
        ffprobe = find_ffmpeg('ffprobe')
        if ffprobe is not None:
            command = [
                ffprobe,
                '-i', audio_path,
                '-show_entries', 'format=duration',
                '-v', 'quiet',
                '-of', 'csv=p=0'
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())

        # ffprobe is not always bundled alongside ffmpeg, so fall back to
        # reading the WAV header with the standard library.
        import wave
        with wave.open(audio_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate() or 1
            return frames / float(rate)
    
    def cleanup(self, file_path: str):
        """Remove temporary file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up: {file_path}")
        except Exception as e:
            logger.warning(f"Cleanup failed for {file_path}: {e}")


# Usage example
if __name__ == "__main__":
    extractor = AudioExtractor()
    
    # Test with a video file
    try:
        audio_file = extractor.extract_audio("sample_lecture.mp4")
        duration = extractor.get_audio_duration(audio_file)
        print(f"Audio extracted: {audio_file}")
        print(f"Duration: {duration} seconds")
    except Exception as e:
        print(f"Error: {e}")
