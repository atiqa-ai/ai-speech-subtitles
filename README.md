# AI Speech Subtitles

Turn a recorded lecture into **Urdu subtitles**.

The pipeline extracts audio from a video, transcribes it with OpenAI Whisper
(auto-detecting the spoken language), translates the transcript to Urdu with
GPT, and writes SRT / WebVTT subtitle files.

```
video ──▶ AudioExtractor ──▶ SpeechToText ──▶ UrduTranslator ──▶ SubtitleGenerator
          16 kHz mono WAV     Whisper +         GPT-4o-mini        .srt / .vtt
                               language detect
```

## Features

- **Automatic language detection** — Whisper identifies the language, with a confidence score
- **Context-aware translation** — tuned for educational and technical content
- **Batch translation** — segments are translated in batches to cut API cost and latency
- **Three subtitle formats** — SRT, WebVTT, and a dual-language SRT showing original + translation
- **Automatic line wrapping** — long lines are wrapped at 42 characters so subtitles stay readable
- **Timing validation** — overlapping or zero-length cues are detected and reported
- **No system-wide ffmpeg needed** — falls back to the binary bundled in `imageio-ffmpeg`

## Requirements

- Python 3.9+
- An OpenAI API key (only needed for the translation step)

## Installation

```bash
git clone https://github.com/<your-username>/ai-speech-subtitles.git
cd ai-speech-subtitles

python -m venv .venv
# Windows:   .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Installing the CPU-only torch wheel first avoids pulling in the multi-gigabyte
CUDA build. If you have an NVIDIA GPU and want the faster path, use plain
`pip install torch` instead.

## Usage

```bash
# Auto-detect the language and translate to Urdu
python main.py lecture.mp4

# Skip translation and just transcribe
python main.py lecture.mp4 --no-translate

# Force a source language
python main.py lecture.mp4 --language en

# Also emit WebVTT and a dual-language SRT
python main.py lecture.mp4 --vtt --dual

# Use a larger Whisper model for better accuracy (slower)
python main.py lecture.mp4 --model small
```

Set your API key before running:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."

# macOS / Linux
export OPENAI_API_KEY="sk-..."
```

### Options

| Flag | Default | Description |
| --- | --- | --- |
| `video` | *(required)* | Path to the input video |
| `--language` | auto-detect | Source language code, e.g. `en`, `ur` |
| `--model` | `base` | Whisper size: `tiny`, `base`, `small`, `medium`, `large` |
| `--no-translate` | off | Transcribe only, skip the API |
| `--vtt` | off | Also write a WebVTT file |
| `--dual` | off | Also write a bilingual SRT |
| `--keep-audio` | off | Keep the intermediate WAV |

## Output

Everything is written to `output/` next to the script, regardless of the
directory you run from:

```
output/
├── temp/                       intermediate audio (deleted unless --keep-audio)
│   └── lecture_audio.wav
└── subtitles/
    ├── lecture_urdu.srt
    ├── lecture_urdu.vtt
    └── lecture_dual.srt
```

Load the `.srt` next to your video — most players (VLC, MPV, YouTube) pick it up automatically.

## Project layout

| File | Purpose |
| --- | --- |
| `main.py` | CLI entry point that chains the four modules together |
| `src/audio_extractor.py` | Pulls 16 kHz mono audio out of a video (ffmpeg, MoviePy fallback) |
| `src/speech_to_text.py` | Whisper transcription and language detection |
| `src/translator_module.py` | GPT translation, with batch and fallback handling |
| `src/subtitle_generator.py` | Builds SRT / WebVTT, wraps lines, validates timing |
| `src/ffmpeg_setup.py` | Locates an ffmpeg binary (system or bundled) |

Each module also runs standalone for quick testing:

```bash
python src/audio_extractor.py
python src/subtitle_generator.py
```

## Notes

- Whisper model weights download on first run (~75 MB for `tiny`, ~290 MB for `base`) and are cached afterwards.
- Translation quality depends on the model size — `base` is a reasonable speed/accuracy trade-off for clear lecture audio.
- Costs come from the OpenAI translation API only; transcription runs locally.

## License

MIT
