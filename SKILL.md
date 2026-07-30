---
name: yt-transcribe
description: >
  Transcribe speech to a timestamped transcript with a local ASR model
  (whisper.cpp by default), for a YouTube video or a local audio/video file.
  Use when a video has no caption track — i.e. yt-summarize exited 2 with
  NO_TRANSCRIPT — or to transcribe a podcast, meeting recording, or media file.
  Writes yt-summarize's transcript cache, so afterwards a plain
  `yt-summarize.py <id>` summarizes it with no extra flags. Runs the model
  on demand in a subprocess (nothing stays resident) and reports
  ASR_QUALITY: suspect with exit 3 when the audio is music or non-speech.
---

# yt-transcribe

Speech-to-text for audio that has no captions, producing the same timestamped
transcript shape that [[yt-summarize]] consumes.

## How it composes with yt-summarize

There is no call graph between the two skills — they share a **file
contract**. `yt-summarize.py`'s `fetch_transcript()` checks its transcript
cache first, so this script's only integration duty is to write that file:

```
~/.cache/yt-summarize/<video-id>.transcript.json
[{"start": 12.34, "seconds": 12, "text": "..."}, ...]
```

Once that exists, `yt-summarize.py <id>` finds it on the cache-hit path and
summarizes normally. **Do not pipe the transcript through your own context to
hand it to yt-summarize** — an hour of speech is 50–100k characters, and the
cache already carries it losslessly.

Honor `YT_SUMMARIZE_CACHE_DIR` if it is set; both scripts read the same var.

## Usage

```bash
uv run /Users/wei/.pi/agent/skills/yt-transcribe/yt-transcribe.py <video-id-or-url> [flags]
```

Copy that path **exactly as written** and stay in your current working
directory — same rule as yt-summarize. Nothing is written to the cwd unless
you pass `--out`.

**Set your shell tool's timeout to at least 1800 seconds.** Transcription is
minutes of compute, not a network fetch. Measured on a 16GB M1 MacBook Air
with the default model: **~6.4x realtime**, i.e. an 18:39 video took 2:55 of
ASR, so budget **~10 minutes per hour of audio**, plus the audio download and a
one-time ~550MB model download on first run.

Local files work too, and are the standalone use of this skill:

```bash
uv run …/yt-transcribe.py ~/Recordings/standup.m4a --out standup.txt
```

## The normal sequence

When yt-summarize has already reported `NO_TRANSCRIPT` / exit 2:

1. **Ask the user first.** yt-summarize's `NO_TRANSCRIPT` verdict is terminal
   for the caption route and stays that way — transcription is a different,
   far more expensive route (minutes of CPU/GPU, a model download). Offer it;
   run it only when the user says yes.
2. Run `yt-transcribe.py <id>`.
3. Read `ASR_QUALITY:` and the exit code (below).
4. On success, run `yt-summarize.py <id>` exactly as you normally would. It
   will print `TRANSCRIPT_CACHE: hit`. Everything downstream — chunking, the
   local summarizer, the Part A/Part B spec — is unchanged.

## Exit codes — read these before summarizing

| Exit | Meaning | What to do |
|------|---------|------------|
| `0` | Transcript written, `ASR_QUALITY: ok` | Proceed to yt-summarize |
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
  a check for ≥10 identical segments in a row.

On exit 3, tell the user the audio appears to be music/non-speech and quote the
numbers. Do **not** summarize it as though it were a talk unless they confirm.
The transcript is still cached, so proceeding later costs nothing extra.

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
| **`large-v3-turbo-q5_0`** *(default)* | ~547MB | large-v3-class English accuracy at the smallest turbo footprint |
| `large-v3-turbo-q8_0` | ~833MB | a little more headroom |
| `large-v3-turbo` | ~1.5GB | unquantized; rarely worth the extra GB |
| `large-v3-q5_0` | ~1.0GB | non-turbo: several times slower, marginal English gain |
| `medium.en-q5_0` | ~514MB | turbo-q5_0 is the same size and better |
| `small.en-q5_1` | ~181MB | fast triage only |
| `base.en` / `tiny.en` | ~141/74MB | smoke-testing the pipeline |

Avoid the small models for real summaries: the output format anchors **every
paragraph** to a transcript timestamp, and timestamp drift at that end of the
range corrupts the anchors even where the words are right.

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
| *(none)* | whisper-cpp + `large-v3-turbo-q5_0` + VAD, English |
| `--backend {whisper-cpp,mlx-whisper}` | ASR engine |
| `--asr-model <key>` | Model within the backend |
| `--list-models` | Backends, models, availability — then exit |
| `--language <code>` | Spoken language, or `auto` to detect (default `en`) |
| `--no-vad` | Disable Silero VAD (VAD is what suppresses music hallucination) |
| `--threads N` | ASR threads — see the note below before raising it |
| `--prompt <text>` | Initial prompt to bias spelling of names/jargon |
| `--video-id <id>` | Cache key to write under (for local files, or to override) |
| `--force` | Re-transcribe even when a cached transcript exists |
| `--no-cache-write` | Don't touch yt-summarize's cache |
| `--out PATH` | Also write a plain `[m:ss]` transcript |
| `--keep-audio DIR` | Keep downloaded/normalized audio instead of a temp dir |
| `--cookies PATH` | Netscape cookies.txt for restricted videos |
| `--engine-arg ARG` | Raw passthrough to the engine (repeatable) |

**Accepted inputs:** bare id, `watch?v=`, `youtu.be`, `/embed/`, `/shorts/`,
or a path to any local audio/video file.

Re-running with a cached transcript present is a no-op that prints
`TRANSCRIPT_CACHE: hit` and exits 0 — safe to call speculatively, and `--force`
is the only way to spend the compute again.

## Notes

- **Pipeline:** `yt-dlp -f bestaudio --extract-audio` → ffmpeg to 16kHz mono
  PCM WAV (whisper requires it) → engine → segments → cache. Audio lands in a
  temp dir that is deleted afterwards unless `--keep-audio` is set.
- **Don't raise `--threads`.** Measured on an M1 Air over the same 18:39 clip:
  4 threads (whisper-cli's default) 2:55, `-t 8` **3:16** — slower, because the
  encode runs on Metal and eight ASR threads contend for four performance
  cores. Dropping VAD didn't help either (3:12), so the default of VAD-on at 4
  threads is both the fastest and the safest setting.
- **`--prompt` is worth using** on technical talks: it biases spelling of
  product names and jargon that ASR otherwise mangles. Note that yt-summarize's
  spec tells the summarizer to copy technical tokens *exactly as the transcript
  renders them*, so a misheard product name propagates into the summary
  verbatim. Fixing it at the ASR step is the only clean fix.
- **Memory on a 16GB machine:** the engine is a subprocess that exits before
  you invoke yt-summarize, so a resident llama-server on `:8080` (several GB
  for a 4-bit 9B) and a ~550MB Whisper never need to be loaded at once. Don't
  restructure this into one long-lived process without a reason.
- **yt-dlp runs with `--ignore-config`**, so a personal
  `~/.config/yt-dlp/config` cannot leak format selectors or its own
  `--cookies` line into the audio fetch. Cookies come only from `--cookies` /
  `YT_TRANSCRIBE_COOKIES` / `YT_SUMMARIZE_COOKIES` / `~/cookies.txt`.
- Deps are [PEP 723](https://peps.python.org/pep-0723/) inline and **stdlib
  only** — deliberately, so yt-summarize's fast caption path never pays for
  MLX/torch resolution. Required on `PATH`: `ffmpeg`/`ffprobe`, plus `yt-dlp`
  for YouTube sources and an ASR engine per `--list-models`.
- The Silero VAD model (`ggml-silero-v5.1.2.bin`) is fetched once alongside the
  first whisper-cpp model. If that download fails, the run continues without
  VAD and says so.
- Env: `YT_TRANSCRIBE_BACKEND`, `YT_TRANSCRIBE_MODEL`,
  `YT_TRANSCRIBE_MODEL_DIR`, `YT_TRANSCRIBE_LANGUAGE`,
  `YT_TRANSCRIBE_WHISPER_CPP` (binary path), `YT_TRANSCRIBE_FFMPEG`,
  `YT_TRANSCRIBE_FFPROBE`, `YT_TRANSCRIBE_MIN_SPEECH_RATIO` (0.30),
  `YT_TRANSCRIBE_MAX_REPETITION` (0.55), `YT_TRANSCRIBE_COOKIES`,
  `YT_SUMMARIZE_CACHE_DIR`, `YT_SUMMARIZE_YTDLP`.
