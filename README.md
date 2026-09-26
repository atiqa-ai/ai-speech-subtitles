# AI Speech Subtitles

Turn a recorded lecture into **Urdu subtitles**.

The pipeline extracts audio from a video, transcribes it with OpenAI Whisper
(auto-detecting the spoken language), translates the transcript to Urdu with
GPT, and writes SRT / WebVTT subtitle files.

```
video â”€â”€â–¶ AudioExtractor â”€â”€â–¶ SpeechToText â”€â”€â–¶ UrduTranslator â”€â”€â–¶ SubtitleGenerator
          16 kHz mono WAV     Whisper +         GPT-4o-mini        .srt / .vtt
                               language detect
```

## Features

- **Automatic language detection** â€” Whisper identifies the language, with a confidence score
- **Context-aware translation** â€” tuned for educational and technical content
- **Batch translation** â€” segments are translated in batches to cut API cost and latency
- **Three subtitle formats** â€” SRT, WebVTT, and a dual-language SRT showing original + translation
- **Automatic line wrapping** â€” long lines are wrapped at 42 characters so subtitles stay readable
- **Timing validation** â€” overlapping or zero-length cues are detected and reported
- **No system-wide ffmpeg needed** â€” falls back to the binary bundled in `imageio-ffmpeg`

## Requirements

- Python 3.10+ (CI covers 3.10, 3.11, and 3.12)
- An OpenAI API key (only needed for the translation step)

## Installation

```bash
git clone https://github.com/atiqa-ai/ai-speech-subtitles.git
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
â”œâ”€â”€ temp/                       intermediate audio (deleted unless --keep-audio)
â”‚   â””â”€â”€ lecture_audio.wav
â””â”€â”€ subtitles/
    â”œâ”€â”€ lecture_urdu.srt
    â”œâ”€â”€ lecture_urdu.vtt
    â””â”€â”€ lecture_dual.srt
```

Load the `.srt` next to your video â€” most players (VLC, MPV, YouTube) pick it up automatically.

## Project layout

| File | Purpose |
| --- | --- |
| `main.py` | CLI entry point that chains the four modules together |
| `src/audio_extractor.py` | Pulls 16 kHz mono audio out of a video (ffmpeg, MoviePy fallback) |
| `src/speech_to_text.py` | Whisper transcription and language detection |
| `src/translator_module.py` | GPT translation, with batch and fallback handling |
| `src/subtitle_generator.py` | Builds SRT / WebVTT, wraps lines, validates timing |
| `src/ffmpeg_setup.py` | Locates an ffmpeg binary (system or bundled) |
| `tests/` | 43 tests covering translation and subtitle logic |

Each module also runs standalone for quick testing:

```bash
python src/audio_extractor.py
python src/subtitle_generator.py
```

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

The test requirements are deliberately minimal. The two modules with real logic
â€” `translator_module` and `subtitle_generator` â€” depend only on the OpenAI SDK,
so the suite runs in a couple of seconds without installing torch or Whisper.

No test contacts the network or needs an API key: the OpenAI client is replaced
with a fake that returns canned replies, which is what makes the batching,
numbering-cleanup, and subtitle-timing behaviour testable at all. One CI step
runs the whole suite with `OPENAI_API_KEY` set to an empty string to prove it.

Three real bugs were found and fixed while writing these tests:

- **A short batch reply silently dropped subtitles.** If the model returned
  fewer lines than segments sent, the extra segments were lost, leaving gaps in
  the file. They are now re-translated individually.
- **A blank line in the reply produced an empty cue.** Blanks were treated as
  valid translations, so a subtitle could come out blank. They are now repaired
  like any other shortfall, and the original text is used as a last resort.
- **The last segment's timing was never validated.** The overlap check looped
  to `len-1`, so a final cue with a negative duration slipped through and broke
  playback. Every segment is now checked.

## Docker

```bash
docker build -t ai-speech-subtitles .
docker run --rm -v "$PWD/output:/app/output" -e OPENAI_API_KEY="sk-..." \
    ai-speech-subtitles lecture.mp4
```

The image installs the CPU-only torch wheel, so it stays far smaller than a
default `pip install torch` would produce. It is still large â€” roughly 2.4 GB,
because torch and Whisper are most of it. `tests/` is included in the image and
CI runs the suite inside the container, which catches a dependency that works on
the host but is missing from the image.

## Continuous integration

`.github/workflows/ci.yml` runs on every push:

| Job | What it does |
| --- | --- |
| `test` | 43 tests on Python 3.10, 3.11, and 3.12, plus a no-API-key run |
| `docker` | Builds the image, checks the entry point, runs the suite in the container |

## Notes

- Whisper model weights download on first run (~75 MB for `tiny`, ~290 MB for `base`) and are cached afterwards.
- Translation quality depends on the model size â€” `base` is a reasonable speed/accuracy trade-off for clear lecture audio.
- Costs come from the OpenAI translation API only; transcription runs locally.

