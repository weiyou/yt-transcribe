# yt-transcribe

Local speech-to-text for audio that has no captions, producing the timestamped
transcript format that [yt-summarize](../yt-summarize) consumes.

Built for the case where `yt-summarize` exits 2 with `NO_TRANSCRIPT` — the
uploader disabled captions and YouTube never generated an ASR track — but it
works equally well on any local audio or video file.

```bash
# a YouTube video with no captions
uv run yt-transcribe.py <video-id-or-url>

# then summarize it exactly as usual — the transcript is already cached
uv run ../yt-summarize/yt-summarize.py <video-id>

# or a local recording
uv run yt-transcribe.py ~/Recordings/standup.m4a --out standup.txt
```

## How it composes

The two skills share a **file contract**, not a call graph.
`yt-summarize.py`'s `fetch_transcript()` checks its transcript cache before
hitting the network, so this script's whole integration duty is to write that
file:

```
~/.cache/yt-summarize/<video-id>.transcript.json
[{"start": 12.34, "seconds": 12, "text": "..."}, ...]
```

After that, a plain `yt-summarize.py <id>` finds it on the cache-hit path and
behaves identically to a captioned video — same chunking, same summarizer, same
Part A/Part B output. No flags, and no code change on the yt-summarize side.

Keeping the engine in a separate process also keeps the dependencies apart:
yt-summarize's fast caption path stays at two light Python deps and never pays
for an MLX/ASR resolution, and the ~550MB Whisper model is unloaded before a
resident `llama-server` is asked to summarize — which matters on a 16GB
machine.

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

Default is whisper.cpp `large-v3-turbo-q5_0` (~547MB): large-v3-class English
accuracy at the smallest turbo footprint, ~1–2s model load, Metal-accelerated.
`--backend mlx-whisper` runs Whisper on MLX via `uvx` instead. Adding an engine
is one `Backend` subclass plus a line in `BACKENDS`.

Measured on a 16GB M1 MacBook Air, default model and settings: **~6.4x
realtime** (an 18:39 video took 2:55), so budget roughly 10 minutes per hour of
audio. Raising `--threads` makes it *slower* — the encode runs on Metal and
extra threads contend for four performance cores.

## Quality gating

Videos with captions disabled skew toward music and studio content, and Whisper
answers non-speech audio by looping one plausible sentence for the whole
runtime — output that reads fine and is entirely fabricated. Every run reports
`SPEECH_RATIO` and `REPETITION`, and exits **3** with `ASR_QUALITY: suspect`
when the audio looks like music or the output looks looped. Silero VAD is on by
default, which suppresses most of it.

Exit codes: `0` transcript written · `1` ordinary failure · `3` written but
quality suspect — review before summarizing.

See [SKILL.md](SKILL.md) for the full flag and environment reference.
