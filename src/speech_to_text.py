"""
Speech-to-Text Module with Automatic Language Detection
Uses OpenAI Whisper for accurate transcription
"""

import whisper
import os
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass

try:
    from .ffmpeg_setup import ensure_ffmpeg_on_path, find_ffmpeg
except ImportError:  # running as a standalone script
    from ffmpeg_setup import ensure_ffmpeg_on_path, find_ffmpeg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TranscriptionSegment:
    """Single transcription segment with timing"""
    start: float
    end: float
    text: str
    language: str


class SpeechToText:
    """
    Handles speech-to-text conversion with automatic language detection
    Uses OpenAI Whisper model
    """
    
    # Language codes Whisper supports
    SUPPORTED_LANGUAGES = {
        'en': 'English',
        'ur': 'Urdu',
        'hi': 'Hindi',
        'ar': 'Arabic',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ko': 'Korean'
    }
    
    def __init__(self, model_size: str = 'base'):
        """
        Initialize Speech-to-Text processor
        
        Args:
            model_size: Whisper model size
                       'tiny', 'base', 'small', 'medium', 'large'
                       Larger = more accurate but slower
        """
        self.model_size = model_size
        self.model = None
        logger.info(f"Initializing Whisper model: {model_size}")
        
    def load_model(self):
        """Load Whisper model (lazy loading for memory efficiency)"""
        if self.model is None:
            # whisper.load_audio / transcribe shell out to a real ffmpeg
            # binary, so it has to be on PATH before we touch the model.
            if ensure_ffmpeg_on_path() is None:
                raise RuntimeError(
                    "ffmpeg not found. Install it, or run: pip install imageio-ffmpeg"
                )
            try:
                self.model = whisper.load_model(self.model_size)
                logger.info(f"Whisper {self.model_size} model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise
    
    @staticmethod
    def load_audio(audio_path: str):
        """
        Load an audio file as a mono 16 kHz float32 numpy array.

        whisper.load_audio() shells out to a bare `ffmpeg` command, which
        fails unless ffmpeg is installed system-wide. We already know the
        path to a usable ffmpeg, so decode the audio here instead and hand
        whisper a plain array, which it accepts directly.
        """
        import numpy as np
        import subprocess
        import tempfile
        import wave

        exe = find_ffmpeg("ffmpeg")
        if exe is None:
            raise RuntimeError(
                "ffmpeg not found. Install it, or run: pip install imageio-ffmpeg"
            )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            subprocess.run(
                [
                    exe, "-i", audio_path,
                    "-f", "wav",
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1",
                    "-y", tmp_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

            with wave.open(tmp_path, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                width = wf.getsampwidth()
                dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
                if dtype is None:
                    raise RuntimeError(f"Unsupported WAV sample width: {width}")
                audio = np.frombuffer(frames, dtype=dtype).astype(np.float32) / 32768.0
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        return audio

    def detect_language(self, audio_path: str) -> Tuple[str, float]:
        """
        Detect spoken language in audio
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (language_code, confidence_score)
        """
        self.load_model()
        
        logger.info("Detecting language...")
        
        # Load audio and pad/trim to 30 seconds
        audio = self.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        
        # Make log-Mel spectrogram
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        
        # Detect language
        _, probs = self.model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)
        confidence = probs[detected_lang]
        
        lang_name = self.SUPPORTED_LANGUAGES.get(detected_lang, detected_lang)
        logger.info(f"Detected language: {lang_name} ({detected_lang}) - Confidence: {confidence:.2%}")
        
        return detected_lang, confidence
    
    def transcribe(
        self, 
        audio_path: str, 
        language: str = None,
        word_timestamps: bool = True
    ) -> Dict:
        """
        Transcribe audio to text with timestamps
        
        Args:
            audio_path: Path to audio file
            language: Force specific language (None for auto-detect)
            word_timestamps: Include word-level timestamps
            
        Returns:
            Dictionary with transcription results
        """
        self.load_model()
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Transcribing audio: {audio_path}")
        
        # Transcription options
        options = {
            'language': language,
            'task': 'transcribe',
            'word_timestamps': word_timestamps,
            'verbose': False
        }
        
        # Decode the audio ourselves and pass a numpy array, so whisper never
        # has to shell out to a system ffmpeg. See load_audio() for details.
        audio = self.load_audio(audio_path)
        
        # Perform transcription
        result = self.model.transcribe(audio, **options)
        
        # Extract segments
        segments = self._process_segments(result['segments'], result['language'])
        
        logger.info(f"Transcription complete: {len(segments)} segments")
        
        return {
            'language': result['language'],
            'text': result['text'],
            'segments': segments,
            'duration': result['segments'][-1]['end'] if result['segments'] else 0
        }
    
    def _process_segments(self, raw_segments: List, language: str) -> List[TranscriptionSegment]:
        """Convert raw Whisper segments to structured format"""
        segments = []
        
        for seg in raw_segments:
            segment = TranscriptionSegment(
                start=seg['start'],
                end=seg['end'],
                text=seg['text'].strip(),
                language=language
            )
            segments.append(segment)
        
        return segments
    
    def transcribe_with_auto_detect(self, audio_path: str) -> Dict:
        """
        Transcribe with automatic language detection
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Complete transcription result with detected language
        """
        # First detect language
        detected_lang, confidence = self.detect_language(audio_path)
        
        # Only proceed if confidence is reasonable
        if confidence < 0.3:
            logger.warning(f"Low language detection confidence: {confidence:.2%}")
        
        # Transcribe with detected language
        result = self.transcribe(audio_path, language=detected_lang)
        result['detection_confidence'] = confidence
        
        return result
    
    def get_full_text(self, segments: List[TranscriptionSegment]) -> str:
        """Combine all segment texts into single string"""
        return ' '.join(seg.text for seg in segments)


# Usage example
if __name__ == "__main__":
    stt = SpeechToText(model_size='base')
    
    try:
        # Example transcription
        result = stt.transcribe_with_auto_detect("sample_audio.wav")
        
        print(f"Detected Language: {result['language']}")
        print(f"Confidence: {result['detection_confidence']:.2%}")
        print(f"\nFull Transcription:\n{result['text']}")
        print(f"\nTotal Segments: {len(result['segments'])}")
        
    except Exception as e:
        print(f"Error: {e}")
