# yt-transcribe

Local speech-to-text for audio that has no captions. Writes a **plain-text
transcript** with one timestamped line per segment, and optionally the JSON
cache that [yt-summarize](../yt-summarize) can reuse later.

```bash
# YouTube (or bare id) → ./I2CK-j-pR7M.txt
uv run yt-transcribe.py <video-id-or-url>

# local recording
uv run yt-transcribe.py ~/Recordings/standup.m4a --out standup.txt
```

## Output

```
[00:00:04] first segment
[00:00:12] second segment
```

Default path: `./<video-id>.txt` in the current working directory (`--out` to
override). On a cache hit, the script re-exports that `.txt` from the JSON
without re-running ASR (`--force` to re-transcribe).

JSON cache (optional integration with yt-summarize):

```
~/.cache/yt-summarize/<video-id>.transcript.json
```

There is **no automatic handoff** to yt-summarize — run that only if you want
a structured summary after the raw transcript exists.

## Requirements

- `ffmpeg` / `ffprobe` (audio normalization to the 16kHz mono WAV whisper needs)
- `yt-dlp` for YouTube sources
- an ASR engine — `brew install whisper-cpp` for the default backend

Model weights download on demand to `~/.cache/yt-transcribe/models` and are
reused. The script itself is stdlib-only ([PEP 723](https://peps.python.org/pep-0723/)
inline, `uv run`).

## Models

```bash
uv run yt-transcribe.py --list-models
```

Default is whisper.cpp `large-v3-turbo-q5_0` (~547MB): multilingual large-v3
turbo at the smallest footprint, ~1–2s model load, Metal-accelerated. Language
defaults to **`auto`** (override with `--language en` / `zh` / …). Forcing the
wrong language — especially `en` on Mandarin or other non-English speech — is a
common cause of hallucination loops; prefer `auto` unless you know the language.
`--backend mlx-whisper` runs Whisper on MLX via `uvx` instead. Adding an engine
is one `Backend` subclass plus a line in `BACKENDS`.

Measured on a 16GB M1 MacBook Air, default model and settings: **~6.4x
realtime** (an 18:39 video took 2:55), so budget roughly 10 minutes per hour of
audio. Raising `--threads` makes it *slower* — the encode runs on Metal and
extra threads contend for four performance cores.

## Quality gating

Videos with captions disabled skew toward music and studio content, and Whisper
answers non-speech audio (or audio in a **wrong forced language**) by looping
one plausible sentence for the whole runtime — output that reads fine and is
entirely fabricated. Every run reports `SPEECH_RATIO` and `REPETITION`, and
exits **3** with `ASR_QUALITY: suspect` when the audio looks like music or the
output looks looped. Silero VAD is on by default, which suppresses most of it.
Repetition detection is script-aware (CJK included), not ASCII-only.

Exit codes: `0` transcript written · `1` ordinary failure · `3` written but
quality suspect — review the `.txt` before trusting it.

See [SKILL.md](SKILL.md) for the full flag and environment reference.
