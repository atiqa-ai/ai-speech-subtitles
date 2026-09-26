"""
Translation Module: English to Urdu
Uses OpenAI GPT for context-aware translation
"""

import os
import logging
from typing import List, Dict
from openai import OpenAI
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TranslatedSegment:
    """Translated text segment with metadata"""
    start: float
    end: float
    original_text: str
    translated_text: str
    source_lang: str
    target_lang: str = 'ur'


class UrduTranslator:
    """
    Handles translation from any language to Urdu
    Specializes in educational and technical content
    """
    
    TRANSLATION_PROMPT = """You are an expert translator specializing in educational content translation to Urdu.

RULES:
1. Translate the following text to simple, clear Urdu
2. Keep technical terms in English if commonly used (e.g., "algorithm", "database")
3. Use Roman Urdu if Urdu script is unavailable
4. Maintain the natural flow and meaning
5. Keep the translation concise and lecture-appropriate
6. For academic content, balance between accuracy and simplicity

Text to translate:
{text}

Provide ONLY the Urdu translation, nothing else."""

    BATCH_PROMPT = """Translate the following lecture segments to Urdu. Return ONLY the translations, one per line, in the same order.

RULES:
- Simple, educational Urdu
- Keep technical terms in English when appropriate
- Natural, conversational tone suitable for lectures

Segments:
{segments}

Translations (one per line):"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize translator
        
        Args:
            api_key: OpenAI API key (or from environment)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key required")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o-mini"  # Fast and cost-effective
        logger.info("Urdu Translator initialized")
    
    def translate_text(self, text: str, source_lang: str = 'en') -> str:
        """
        Translate single text to Urdu
        
        Args:
            text: Text to translate
            source_lang: Source language code
            
        Returns:
            Translated Urdu text
        """
        if not text.strip():
            return ""
        
        logger.info(f"Translating text ({len(text)} chars) from {source_lang} to Urdu")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert educational translator."},
                    {"role": "user", "content": self.TRANSLATION_PROMPT.format(text=text)}
                ],
                temperature=0.3,  # Lower temperature for consistent translations
                max_tokens=2000
            )
            
            translation = response.choices[0].message.content.strip()
            logger.info("Translation completed")
            return translation
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise
    
    def translate_segments(
        self, 
        segments: List[Dict], 
        batch_size: int = 10
    ) -> List[TranslatedSegment]:
        """
        Translate multiple segments efficiently
        
        Args:
            segments: List of transcription segments
            batch_size: Number of segments to translate together
            
        Returns:
            List of translated segments with timing info
        """
        logger.info(f"Translating {len(segments)} segments in batches of {batch_size}")
        
        translated_segments = []
        
        # Process in batches for efficiency
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            
            try:
                # Prepare batch text
                batch_texts = [
                    f"{idx + 1}. {seg.text}" 
                    for idx, seg in enumerate(batch)
                ]
                batch_input = '\n'.join(batch_texts)
                
                # Translate batch
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert educational translator."},
                        {"role": "user", "content": self.BATCH_PROMPT.format(segments=batch_input)}
                    ],
                    temperature=0.3,
                    max_tokens=3000
                )
                
                # Parse translations
                translations = response.choices[0].message.content.strip().split('\n')
                
                # The model does not always return exactly one line per
                # segment; it may merge or drop some. zip() would silently drop
                # the leftovers, leaving gaps in the subtitles, so translate any
                # unmatched segments one at a time instead.
                if len(translations) != len(batch):
                    logger.warning(
                        f"Batch returned {len(translations)} translations for "
                        f"{len(batch)} segments; repairing the remainder individually."
                    )
                
                for idx, seg in enumerate(batch):
                    cleaned_trans = None

                    if idx < len(translations):
                        trans = translations[idx]
                        candidate = trans.split('. ', 1)[-1] if '. ' in trans else trans
                        # A blank line in the reply is a *missing* translation,
                        # not a request for an empty subtitle. Treating it as
                        # present left a visible gap in the subtitle file, so
                        # blanks are repaired like any other shortfall.
                        if candidate.strip():
                            cleaned_trans = candidate

                    if cleaned_trans is None:
                        try:
                            cleaned_trans = self.translate_text(seg.text, seg.language)
                        except Exception as e2:
                            logger.error(f"Repair translation failed: {e2}")
                            cleaned_trans = seg.text

                    # Last resort: never emit an empty cue.
                    if not cleaned_trans.strip():
                        logger.warning(
                            f"Empty translation for segment {idx + 1}; "
                            f"falling back to the original text."
                        )
                        cleaned_trans = seg.text

                    translated_seg = TranslatedSegment(
                        start=seg.start,
                        end=seg.end,
                        original_text=seg.text,
                        translated_text=cleaned_trans.strip(),
                        source_lang=seg.language
                    )
                    translated_segments.append(translated_seg)
                
                logger.info(f"Batch {i // batch_size + 1} translated ({len(batch)} segments)")
                
            except Exception as e:
                logger.error(f"Batch translation failed: {e}")
                # Fallback: translate individually
                for seg in batch:
                    try:
                        trans_text = self.translate_text(seg.text, seg.language)
                        translated_seg = TranslatedSegment(
                            start=seg.start,
                            end=seg.end,
                            original_text=seg.text,
                            translated_text=trans_text,
                            source_lang=seg.language
                        )
                        translated_segments.append(translated_seg)
                    except Exception as e2:
                        logger.error(f"Individual translation failed for segment: {e2}")
                        # Use original text as fallback
                        translated_segments.append(TranslatedSegment(
                            start=seg.start,
                            end=seg.end,
                            original_text=seg.text,
                            translated_text=seg.text,
                            source_lang=seg.language
                        ))
        
        logger.info(f"Translation complete: {len(translated_segments)} segments")
        return translated_segments
    
    def translate_with_context(
        self, 
        text: str, 
        context: str = None,
        domain: str = "general"
    ) -> str:
        """
        Translate with additional context for better accuracy
        
        Args:
            text: Text to translate
            context: Additional context about the content
            domain: Content domain (general, technical, medical, etc.)
            
        Returns:
            Context-aware translation
        """
        enhanced_prompt = f"""Translate this {domain} lecture content to Urdu.

Context: {context or 'Educational lecture content'}

Text: {text}

Provide clear, simple Urdu translation:"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Expert educational translator"},
                {"role": "user", "content": enhanced_prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        return response.choices[0].message.content.strip()


# Usage example
if __name__ == "__main__":
    translator = UrduTranslator()
    
    # Test single translation
    english_text = "Welcome to this lecture on machine learning algorithms."
    urdu_text = translator.translate_text(english_text)
    
    print(f"English: {english_text}")
    print(f"Urdu: {urdu_text}")
