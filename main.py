"""
Lecture video -> Urdu subtitle pipeline.

Chains the four modules together:

    video  ->  AudioExtractor  ->  SpeechToText  ->  UrduTranslator  ->  SubtitleGenerator
             (wav, 16 kHz)     (Whisper + auto  (GPT-4o-mini)        (.srt / .vtt)
                                  lang detect)

Usage:
    # auto-detect the spoken language, translate to Urdu
    python main.py lecture.mp4

    # force a source language and skip translation
    python main.py lecture.mp4 --language en --no-translate

    # also emit WebVTT and a dual-language SRT
    python main.py lecture.mp4 --vtt --dual

Requires OPENAI_API_KEY in the environment for the translation step.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.audio_extractor import AudioExtractor          # noqa: E402
from src.speech_to_text import SpeechToText              # noqa: E402
from src.subtitle_generator import SubtitleGenerator    # noqa: E402
from src.translator_module import UrduTranslator        # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe a lecture video and produce Urdu subtitles."
    )
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument(
        "--language",
        default=None,
        help="Source language code (e.g. en, ur). Omit to auto-detect.",
    )
    parser.add_argument(
        "--model",
        default="base",
        help="Whisper model size: tiny, base, small, medium, large (default: base)",
    )
    parser.add_argument(
        "--no-translate",
        action="store_true",
        help="Skip translation and subtitle the transcription as-is.",
    )
    parser.add_argument(
        "--vtt", action="store_true", help="Also write a WebVTT subtitle file."
    )
    parser.add_argument(
        "--dual",
        action="store_true",
        help="Also write an SRT containing both the original and translated text.",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep the intermediate .wav file instead of deleting it.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: video not found: {args.video}")
        return 1

    base_name = os.path.splitext(os.path.basename(args.video))[0]

    # Anchor all output to this script's folder so results never scatter into
    # whatever directory the user happened to run the command from.
    out_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    temp_dir = os.path.join(out_root, "temp")
    subs_dir = os.path.join(out_root, "subtitles")

    # 1. Extract audio
    print("[1/4] Extracting audio...")
    extractor = AudioExtractor(temp_dir=temp_dir)
    audio_path = extractor.extract_audio(args.video)
    print(f"      duration: {extractor.get_audio_duration(audio_path):.1f}s")

    # 2. Transcribe
    print("[2/4] Transcribing...")
    stt = SpeechToText(model_size=args.model)
    if args.language:
        result = stt.transcribe(audio_path, language=args.language)
        detected = f"{args.language} (forced)"
        confidence = None
    else:
        detected, confidence = stt.detect_language(audio_path)
        print(f"      detected language: {detected} ({confidence:.0%} confidence)")
        result = stt.transcribe(audio_path, language=detected)

    segments = result["segments"]
    print(f"      {len(segments)} segments transcribed")

    # 3. Translate
    if args.no_translate:
        print("[3/4] Skipping translation (--no-translate)")
        from src.translator_module import TranslatedSegment

        translated = [
            TranslatedSegment(s.start, s.end, s.text, s.text, s.language)
            for s in segments
        ]
    else:
        print("[3/4] Translating to Urdu...")
        if not os.getenv("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY is not set. Use --no-translate to skip.")
            return 1
        translated = UrduTranslator().translate_segments(segments)
        if not translated:
            print("Error: translation produced no segments.")
            return 1

    # 4. Write subtitles
    print("[4/4] Writing subtitles...")
    generator = SubtitleGenerator(output_dir=subs_dir)

    # Long lines make subtitles unreadable, so wrap them first.
    translated = generator.optimize_timing(translated)

    if not generator.validate_timing(translated):
        print("Warning: subtitle timings overlap; the file may play back oddly.")

    srt_path = generator.generate_srt(translated, base_name)
    print(f"      SRT: {srt_path}")

    if args.vtt:
        print(f"      VTT: {generator.generate_vtt(translated, base_name)}")
    if args.dual:
        print(f"      dual: {generator.generate_dual_language_srt(translated, base_name)}")

    if not args.keep_audio:
        extractor.cleanup(audio_path)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
