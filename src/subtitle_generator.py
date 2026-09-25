"""
Subtitle Generator Module
Creates SRT format subtitles with proper timing
"""

import os
import logging
from typing import List
from datetime import timedelta
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SubtitleEntry:
    """Single subtitle entry"""
    index: int
    start_time: float
    end_time: float
    text: str


class SubtitleGenerator:
    """
    Generates subtitle files in SRT format
    Industry-standard format supported by all video players
    """
    
    def __init__(self, output_dir: str = './subtitles'):
        """
        Initialize subtitle generator
        
        Args:
            output_dir: Directory to save subtitle files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Subtitle generator initialized: {output_dir}")
    
    def format_timestamp(self, seconds: float) -> str:
        """
        Convert seconds to SRT timestamp format
        
        Format: HH:MM:SS,mmm
        Example: 00:01:23,456
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp string
        """
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        secs = int(td.total_seconds() % 60)
        millis = int((td.total_seconds() % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def create_srt_content(self, segments: List) -> str:
        """
        Create SRT format content from segments
        
        Args:
            segments: List of translated segments with timing
            
        Returns:
            Complete SRT format string
        """
        srt_entries = []
        
        for idx, seg in enumerate(segments, start=1):
            # Format: Index, Start --> End, Text, Blank line
            entry = f"{idx}\n"
            entry += f"{self.format_timestamp(seg.start)} --> {self.format_timestamp(seg.end)}\n"
            entry += f"{seg.translated_text}\n\n"
            srt_entries.append(entry)
        
        return ''.join(srt_entries)
    
    def generate_srt(
        self, 
        segments: List, 
        filename: str,
        language: str = 'urdu'
    ) -> str:
        """
        Generate SRT subtitle file
        
        Args:
            segments: List of translated segments
            filename: Output filename (without extension)
            language: Language identifier for filename
            
        Returns:
            Path to generated SRT file
        """
        logger.info(f"Generating SRT file: {filename}")
        
        # Create SRT content
        srt_content = self.create_srt_content(segments)
        
        # Generate output path
        output_filename = f"{filename}_{language}.srt"
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Write to file with UTF-8 encoding (important for Urdu)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        
        logger.info(f"SRT file generated: {output_path}")
        logger.info(f"Total subtitles: {len(segments)}")
        
        return output_path
    
    def generate_vtt(self, segments: List, filename: str) -> str:
        """
        Generate WebVTT format (for web players)
        
        Args:
            segments: List of translated segments
            filename: Output filename
            
        Returns:
            Path to generated VTT file
        """
        logger.info(f"Generating VTT file: {filename}")
        
        vtt_content = "WEBVTT\n\n"
        
        for idx, seg in enumerate(segments, start=1):
            # VTT uses dots instead of commas for milliseconds
            start = self.format_timestamp(seg.start).replace(',', '.')
            end = self.format_timestamp(seg.end).replace(',', '.')
            
            vtt_content += f"{idx}\n"
            vtt_content += f"{start} --> {end}\n"
            vtt_content += f"{seg.translated_text}\n\n"
        
        output_path = os.path.join(self.output_dir, f"{filename}_urdu.vtt")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(vtt_content)
        
        logger.info(f"VTT file generated: {output_path}")
        return output_path
    
    def validate_timing(self, segments: List) -> bool:
        """
        Validate subtitle timing consistency
        
        Args:
            segments: List of segments
            
        Returns:
            True if timing is valid
        """
        if not segments:
            return True
        
        for i, current in enumerate(segments):
            # Check for negative duration. This has to run for every segment
            # including the last one, which a len-1 loop would skip.
            if current.end <= current.start:
                logger.warning(f"Invalid duration in segment {i + 1}")
                return False
            
            # Check for overlaps with the following segment
            if i + 1 < len(segments) and current.end > segments[i + 1].start:
                logger.warning(f"Timing overlap between segments {i + 1} and {i + 2}")
                return False
        
        return True
    
    def optimize_timing(self, segments: List, max_chars_per_line: int = 42) -> List:
        """
        Optimize subtitle timing and text length
        
        Args:
            segments: Original segments
            max_chars_per_line: Maximum characters per line
            
        Returns:
            Optimized segments
        """
        optimized = []
        
        for seg in segments:
            # Split long text into multiple lines
            text = seg.translated_text
            
            if len(text) > max_chars_per_line:
                # Simple split at word boundary
                words = text.split()
                lines = []
                current_line = []
                current_length = 0
                
                for word in words:
                    if current_length + len(word) + 1 <= max_chars_per_line:
                        current_line.append(word)
                        current_length += len(word) + 1
                    else:
                        if current_line:
                            lines.append(' '.join(current_line))
                        current_line = [word]
                        current_length = len(word)
                
                if current_line:
                    lines.append(' '.join(current_line))
                
                text = '\n'.join(lines)
            
            # Create optimized segment
            optimized_seg = type(seg)(
                start=seg.start,
                end=seg.end,
                original_text=seg.original_text if hasattr(seg, 'original_text') else '',
                translated_text=text,
                source_lang=seg.source_lang if hasattr(seg, 'source_lang') else 'en'
            )
            optimized.append(optimized_seg)
        
        return optimized
    
    def generate_dual_language_srt(
        self, 
        segments: List, 
        filename: str
    ) -> str:
        """
        Generate SRT with both original and translated text
        
        Args:
            segments: List of translated segments
            filename: Output filename
            
        Returns:
            Path to dual-language SRT file
        """
        srt_entries = []
        
        for idx, seg in enumerate(segments, start=1):
            entry = f"{idx}\n"
            entry += f"{self.format_timestamp(seg.start)} --> {self.format_timestamp(seg.end)}\n"
            entry += f"{seg.original_text}\n"
            entry += f"{seg.translated_text}\n\n"
            srt_entries.append(entry)
        
        output_path = os.path.join(self.output_dir, f"{filename}_dual.srt")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(''.join(srt_entries))
        
        logger.info(f"Dual-language SRT generated: {output_path}")
        return output_path


# Usage example
if __name__ == "__main__":
    from translator_module import TranslatedSegment
    
    # Sample data
    segments = [
        TranslatedSegment(0.0, 3.5, "Hello everyone", "سلام سب کو", "en"),
        TranslatedSegment(3.5, 7.0, "Welcome to this lecture", "اس لیکچر میں خوش آمدید", "en"),
    ]
    
    generator = SubtitleGenerator()
    
    # Generate SRT
    srt_path = generator.generate_srt(segments, "sample_lecture")
    print(f"Generated: {srt_path}")
    
    # Validate
    is_valid = generator.validate_timing(segments)
    print(f"Timing valid: {is_valid}")
