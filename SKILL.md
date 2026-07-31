---
name: yt-transcribe
description: >
  Transcribe speech to a timestamped plain-text transcript with a local ASR
  model (whisper.cpp), for a YouTube video or a local audio/video file. Use
  when a video has no caption track — yt-summarize exited 2 with NO_TRANSCRIPT
  — or to transcribe a podcast, meeting recording, or media file.
---

# yt-transcribe

Speech-to-text for audio that has no captions. Primary deliverable is a
**plain-text transcript** in the current working directory.

## Output

After a successful run:

```
./<video-id>.txt
[00:00:04] first segment text
[00:00:12] second segment text
…
```

Also writes (unless `--no-cache-write`):

```
~/.cache/yt-summarize/<video-id>.transcript.json
```

so a later `yt-summarize.py <id>` can reuse the ASR work if desired. **Do not
automatically run yt-summarize after this skill** — hand the user the `.txt`
(or summarize only if they ask).

Honor `YT_SUMMARIZE_CACHE_DIR` if set (JSON cache location).

## Usage

```bash
uv run /Users/wei/.pi/agent/skills/yt-transcribe/yt-transcribe.py <video-id-or-url> [flags]
```

Copy that path **exactly as written** and stay in your current working
directory — the `.txt` is written to the cwd (override with `--out`).

**Set your shell tool's timeout to at least 1800 seconds.** Transcription is
minutes of compute, not a network fetch. Measured on a 16GB M1 MacBook Air
with the default model: **~6.4x realtime**, i.e. an 18:39 video took 2:55 of
ASR, so budget **~10 minutes per hour of audio**, plus the audio download and a
one-time ~550MB model download on first run.

Local files work too:

```bash
uv run …/yt-transcribe.py ~/Recordings/standup.m4a --out standup.txt
```

## Workflow

When yt-summarize has already reported `NO_TRANSCRIPT` / exit 2, or the user
asks for a transcript:

1. **Ask the user first** if this was only offered as an expensive alternative
   to captions (minutes of CPU/GPU, possible model download). Run only when
   they want it.
2. Run `yt-transcribe.py <id>` with timeout ≥ 1800s.
3. Read `ASR_QUALITY:`, exit code, and `TRANSCRIPT_FILE:`.
4. On success, **show the user the path to the `.txt`** (and optionally a short
   preview). Do **not** chain into yt-summarize unless the user asks.

Re-running with a cache hit rewrites the `.txt` from the JSON without ASR
(`TRANSCRIPT_CACHE: hit`). Use `--force` to re-transcribe.

## Exit codes

| Exit | Meaning | What to do |
|------|---------|------------|
| `0` | Transcript written, `ASR_QUALITY: ok` | Point user at `TRANSCRIPT_FILE` |
| `1` | Ordinary failure (bad id, download error, engine missing) | Read the error; one retry may be worth it |
| `3` | Transcript written but `ASR_QUALITY: suspect` | **Stop and check with the user** |

**Exit 3 is the one that matters.** Videos with captions disabled skew heavily
toward music and studio content, and Whisper answers non-speech audio by
looping one plausible sentence for the whole runtime. That output reads fine
and is entirely fabricated. The script measures two signals and prints them:

- `SPEECH_RATIO: N%` — how much of the runtime actual speech covers. With VAD
  on, music and silence are skipped, so a low ratio means "not a talking
  video" (default threshold 30%).
- `REPETITION: N%` — share of duplicate segments (default threshold 55%), plus
  a check for ≥10 identical segments in a row. Duplicate detection keeps CJK
  and other non-ASCII letters (not ASCII-only).
- `LANGUAGE:` / `LANGUAGE_DETECTED:` — requested language and, when the engine
  reports it, the detected code.

On exit 3, tell the user the audio appears to be music/non-speech **or a
wrong-language force** and quote the numbers. If language was not `auto`,
suggest re-running with `--language auto` (forcing `en` on Mandarin/etc. is a
classic Whisper loop). The `.txt` is still written — review before trusting it.

## Models

```bash
uv run …/yt-transcribe.py --list-models
```

Weights are downloaded on demand to `~/.cache/yt-transcribe/models`
(`YT_TRANSCRIBE_MODEL_DIR`) and reused. **The model loads once per file, not
once per segment** — the engine streams the whole audio through in one process,
then exits, leaving nothing resident.

### whisper-cpp (default backend)

`brew install whisper-cpp`. A single Metal-accelerated binary, ~1–2s mmap'd
model load, no Python import cost, native segment timestamps.

| `--asr-model` | Size | Notes |
|---------------|------|-------|
| **`large-v3-turbo-q5_0`** *(default)* | ~547MB | multilingual large-v3 turbo at the smallest footprint |
| `large-v3-turbo-q8_0` | ~833MB | a little more headroom |
| `large-v3-turbo` | ~1.5GB | unquantized; rarely worth the extra GB |
| `large-v3-q5_0` | ~1.0GB | non-turbo: several times slower, marginal English gain |
| `medium.en-q5_0` | ~514MB | turbo-q5_0 is the same size and better |
| `small.en-q5_1` | ~181MB | fast triage only |
| `base.en` / `tiny.en` | ~141/74MB | smoke-testing the pipeline |

### mlx-whisper

`--backend mlx-whisper`, run via `uvx` so nothing installs permanently. Costs
a Python + MLX cold start (~5–10s vs ~1–2s) and a throwaway venv on first use.
Useful as an MLX-native comparison point.

### Adding a backend

Subclass `Backend` in `yt-transcribe.py` (implement `unavailable_reason` and
`transcribe`, declare a `models` dict) and add one line to `BACKENDS`. The
contract is narrow: given a 16kHz mono WAV, return
`[{start, end, text}]` in playback order. `parakeet-mlx`
(`parakeet-tdt-0.6b-v3`) is the obvious next one — faster than Whisper turbo
on Apple silicon and it does not loop on non-speech.

## Flags

| Flag | Effect |
|------|--------|
| *(none)* | whisper-cpp + `large-v3-turbo-q5_0` + VAD, language `auto`; write `./<id>.txt` |
| `--backend {whisper-cpp,mlx-whisper}` | ASR engine |
| `--asr-model <key>` | Model within the backend |
| `--list-models` | Backends, models, availability — then exit |
| `--language <code>` | Spoken language (`en`, `zh`, …), or `auto` to detect (default `auto`) |
| `--no-vad` | Disable Silero VAD (VAD is what suppresses music hallucination) |
| `--threads N` | ASR threads — see the note below before raising it |
| `--prompt <text>` | Initial prompt to bias spelling of names/jargon |
| `--video-id <id>` | Cache key / default `.txt` stem (for local files, or to override) |
| `--force` | Re-transcribe even when a cached transcript exists |
| `--no-cache-write` | Don't write the yt-summarize JSON cache |
| `--out PATH` | Plain-text path (default: `./<video-id>.txt`) |
| `--keep-audio DIR` | Keep downloaded/normalized audio instead of a temp dir |
| `--cookies PATH` | Netscape cookies.txt for restricted videos |
| `--engine-arg ARG` | Raw passthrough to the engine (repeatable) |

**Accepted inputs:** bare id, `watch?v=`, `youtu.be`, `/embed/`, `/shorts/`,
or a path to any local audio/video file.

## Notes

- **Pipeline:** `yt-dlp -f bestaudio --extract-audio` → ffmpeg to 16kHz mono
  PCM WAV (whisper requires it) → engine → segments → `.txt` + optional JSON
  cache. Audio lands in a temp dir that is deleted afterwards unless
  `--keep-audio` is set.
- **Don't raise `--threads`.** Measured on an M1 Air over the same 18:39 clip:
  4 threads (whisper-cli's default) 2:55, `-t 8` **3:16** — slower, because the
  encode runs on Metal and eight ASR threads contend for four performance
  cores. Dropping VAD didn't help either (3:12), so the default of VAD-on at 4
  threads is both the fastest and the safest setting.
- **`--prompt` is worth using** on technical talks: it biases spelling of
  product names and jargon that ASR otherwise mangles.
- **Memory on a 16GB machine:** the engine is a subprocess that exits when
  done, so a resident llama-server and a ~550MB Whisper never need to be
  loaded at once.
- **yt-dlp runs with `--ignore-config`**, so a personal
  `~/.config/yt-dlp/config` cannot leak format selectors or its own
  `--cookies` line into the audio fetch. Cookies come only from `--cookies` /
  `YT_TRANSCRIBE_COOKIES` / `YT_SUMMARIZE_COOKIES` / `~/cookies.txt`.
- Deps are [PEP 723](https://peps.python.org/pep-0723/) inline and **stdlib
  only**. Required on `PATH`: `ffmpeg`/`ffprobe`, plus `yt-dlp` for YouTube
  sources and an ASR engine per `--list-models`.
- The Silero VAD model (`ggml-silero-v5.1.2.bin`) is fetched once alongside the
  first whisper-cpp model. If that download fails, the run continues without
  VAD and says so.
- **Language:** default is `auto`. Pin with `--language en` only when you know
  the audio is English (small speed/accuracy edge). For Chinese, Japanese,
  bilingual docs, etc., leave `auto` or set the right code (`zh`, `ja`, …).
  English-only models (`*.en*`) cannot emit other languages — the script warns
  if you force a non-English code with one of them.
- Env: `YT_TRANSCRIBE_BACKEND`, `YT_TRANSCRIBE_MODEL`,
  `YT_TRANSCRIBE_MODEL_DIR`, `YT_TRANSCRIBE_LANGUAGE` (default `auto`),
  `YT_TRANSCRIBE_WHISPER_CPP` (binary path), `YT_TRANSCRIBE_FFMPEG`,
  `YT_TRANSCRIBE_FFPROBE`, `YT_TRANSCRIBE_MIN_SPEECH_RATIO` (0.30),
  `YT_TRANSCRIBE_MAX_REPETITION` (0.55), `YT_TRANSCRIBE_COOKIES`,
  `YT_SUMMARIZE_CACHE_DIR`, `YT_SUMMARIZE_YTDLP`.
