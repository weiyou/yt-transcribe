#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///

"""
Transcribe speech to a timestamped transcript with a local ASR model.

Writes:
  - a raw plain-text transcript in the cwd: <id>.txt  ([HH:MM:SS] line per
    segment), override with --out
  - optionally yt-summarize's JSON cache (~/.cache/yt-summarize/…) so a later
    summarize can reuse the audio work

Usage:
    uv run yt-transcribe.py <video-id-or-url>
    uv run yt-transcribe.py ./recording.m4a --video-id my-recording
    uv run yt-transcribe.py <id> --out /tmp/talk.txt
    uv run yt-transcribe.py --list-models

Deps are stdlib only; the ASR engines are external binaries/tools discovered
on PATH (see BACKENDS). Requires ffmpeg, plus yt-dlp for YouTube sources.

Exit codes:
    0  transcript written
    1  ordinary failure (bad id, download error, engine missing/failed)
    3  transcript written but quality is suspect (music / non-speech / looped
       output) — review before treating it as speech
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, urlparse

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_SUSPECT = 3

# Transcript cache shared with yt-summarize. Writing here is the whole
# integration: yt-summarize's fetch_transcript() reads this path first, so a
# transcript produced from audio needs no special flag on the summarize side.
CACHE_DIR = Path(
    os.environ.get(
        "YT_SUMMARIZE_CACHE_DIR",
        os.environ.get("YT_TRANSCRIBE_CACHE_DIR", "~/.cache/yt-summarize"),
    )
).expanduser()

# ASR weights live apart from the transcript cache: they are large, shared
# across videos, and must survive a cache wipe.
MODEL_DIR = Path(
    os.environ.get("YT_TRANSCRIBE_MODEL_DIR", "~/.cache/yt-transcribe/models")
).expanduser()

YTDLP_BIN = os.environ.get("YT_SUMMARIZE_YTDLP", "yt-dlp")
FFMPEG_BIN = os.environ.get("YT_TRANSCRIBE_FFMPEG", "ffmpeg")
FFPROBE_BIN = os.environ.get("YT_TRANSCRIBE_FFPROBE", "ffprobe")

DEFAULT_BACKEND = os.environ.get("YT_TRANSCRIBE_BACKEND", "whisper-cpp")
# Default auto-detect: forcing `en` on Mandarin/etc. is a classic Whisper
# failure mode (plausible-sentence loops). Pin with --language en when you
# know the audio is English and want a small speed/accuracy edge.
DEFAULT_LANGUAGE = os.environ.get("YT_TRANSCRIBE_LANGUAGE", "auto")

# Whisper wants 16 kHz mono PCM; anything else is resampled internally at best
# and rejected at worst, so normalize once up front.
TARGET_RATE = 16000

# --ignore-config keeps a personal ~/.config/yt-dlp/config (format selectors,
# --embed-thumbnail, its own --cookies line) out of our calls: this script
# decides cookies and formats itself.
YTDLP_BASE_ARGS = ["--ignore-config", "--no-warnings"]

_COOKIES_ENV = os.environ.get(
    "YT_TRANSCRIBE_COOKIES", os.environ.get("YT_SUMMARIZE_COOKIES")
)
DEFAULT_COOKIES_PATH = Path("~/cookies.txt").expanduser()

# Quality gates. Videos with captions disabled skew heavily toward music and
# studio content, and Whisper answers non-speech audio by looping a plausible
# sentence forever. A summary built on that is confident fiction, so measure
# and say so rather than handing it downstream silently.
MIN_SPEECH_RATIO = float(os.environ.get("YT_TRANSCRIBE_MIN_SPEECH_RATIO", "0.30"))
MAX_REPETITION = float(os.environ.get("YT_TRANSCRIBE_MAX_REPETITION", "0.55"))


# ---------------------------------------------------------------------------
# small shared helpers (conventions mirrored from yt-summarize.py)
# ---------------------------------------------------------------------------

def extract_video_id(raw: str) -> str | None:
    """Extract a YouTube video ID from a URL or bare ID."""
    raw = raw.strip()

    if re.fullmatch(r"[A-Za-z0-9_-]{11}", raw):
        return raw

    if not raw.startswith(("http://", "https://")):
        raw = "https://" + raw

    parsed = urlparse(raw)
    hostname = (parsed.hostname or "").lower()

    if "youtube.com" in hostname:
        qs = parse_qs(parsed.query)
        vid = qs.get("v", [None])[0]
        if vid:
            return vid
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 2 and parts[-1]:
            return parts[-1]
        return None

    if "youtu.be" in hostname:
        vid = parsed.path.strip("/")
        return vid if re.fullmatch(r"[A-Za-z0-9_-]{11}", vid) else None

    return None


def format_timestamp(seconds: float) -> str:
    """Always [HH:MM:SS]-ready: zero-padded hours, minutes, seconds."""
    total = max(int(seconds), 0)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def resolve_cookies_path(explicit: str | None = None) -> Path | None:
    """Return a usable Netscape cookies.txt path, or None.

    Priority: --cookies > YT_TRANSCRIBE_COOKIES / YT_SUMMARIZE_COOKIES >
    ~/cookies.txt. Env values 0/none/off/false/"" disable the auto-detect.
    """
    if explicit is not None and str(explicit).strip():
        p = Path(str(explicit)).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"cookies file not found: {p}")
        return p

    if _COOKIES_ENV is not None:
        raw = _COOKIES_ENV.strip()
        if raw.lower() in ("", "0", "none", "off", "false", "no"):
            return None
        p = Path(raw).expanduser()
        if not p.is_file():
            print(
                f"COOKIES: {raw!r} from env not found; continuing without",
                file=sys.stderr,
            )
            return None
        return p

    return DEFAULT_COOKIES_PATH if DEFAULT_COOKIES_PATH.is_file() else None


def ytdlp_cookie_args(cookies_path: Path | None) -> list[str]:
    return [] if cookies_path is None else ["--cookies", str(cookies_path)]


def which(binary: str) -> str | None:
    """Resolve a binary on PATH, or accept it as a direct path."""
    return shutil.which(binary) or (binary if Path(binary).is_file() else None)


def _tail(output: str | None, lines: int = 5) -> str:
    """Last few lines of captured output, formatted for an error message."""
    body = "\n".join((output or "").strip().splitlines()[-lines:])
    return f"\n{body}" if body else ""


def human_bytes(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.1f}GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.0f}MB"
    return f"{n / 1024:.0f}KB"


def download(url: str, dest: Path, label: str) -> None:
    """Stream a file to dest, atomically, with coarse progress on stderr."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"DOWNLOAD: {label} → {dest}", file=sys.stderr)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "yt-transcribe"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("content-length") or 0)
            done = 0
            step = max(total // 10, 1 << 23) if total else 1 << 23
            next_mark = step
            with tmp.open("wb") as fh:
                while True:
                    buf = resp.read(1 << 20)
                    if not buf:
                        break
                    fh.write(buf)
                    done += len(buf)
                    if done >= next_mark:
                        pct = f" ({done * 100 // total}%)" if total else ""
                        print(
                            f"  … {human_bytes(done)}{pct}",
                            file=sys.stderr,
                            flush=True,
                        )
                        next_mark += step
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"download failed for {label}: {e}") from e
    tmp.replace(dest)
    print(f"DOWNLOAD: done ({human_bytes(dest.stat().st_size)})", file=sys.stderr)


# ---------------------------------------------------------------------------
# ASR backends
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    """One selectable ASR model within a backend."""

    key: str
    ref: str  # filename for whisper-cpp, HF repo id for mlx-whisper
    size_mb: int
    note: str
    url: str | None = None  # weights source, when the backend fetches by hand


@dataclass
class TranscribeOptions:
    language: str = DEFAULT_LANGUAGE
    vad: bool = True
    threads: int | None = None
    initial_prompt: str | None = None
    extra_args: list[str] = field(default_factory=list)


class Backend(ABC):
    """One ASR engine.

    Adding an engine means one subclass plus a line in BACKENDS. The contract
    is deliberately narrow: hand it a 16 kHz mono WAV, get segments back.
    """

    key: str = ""
    label: str = ""
    default_model: str = ""
    models: dict[str, ModelSpec] = {}

    @abstractmethod
    def unavailable_reason(self) -> str | None:
        """None when usable, else a one-line install hint."""

    @abstractmethod
    def transcribe(
        self, wav: Path, model: ModelSpec, opts: TranscribeOptions
    ) -> list[dict]:
        """Return [{start, end, text}] with float seconds, in playback order."""

    def resolve_model(self, name: str | None) -> ModelSpec:
        key = name or self.default_model
        if key not in self.models:
            known = ", ".join(self.models)
            raise RuntimeError(
                f"unknown model {key!r} for backend {self.key}. Known: {known}"
            )
        return self.models[key]


# --- whisper.cpp -----------------------------------------------------------

WHISPER_CPP_REPO = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
WHISPER_VAD_REPO = "https://huggingface.co/ggml-org/whisper-vad/resolve/main"
WHISPER_VAD_FILE = "ggml-silero-v5.1.2.bin"


def _ggml(key: str, filename: str, size_mb: int, note: str) -> ModelSpec:
    return ModelSpec(
        key=key,
        ref=filename,
        size_mb=size_mb,
        note=note,
        url=f"{WHISPER_CPP_REPO}/{filename}",
    )


class WhisperCppBackend(Backend):
    """ggml whisper.cpp via the whisper-cli binary (Metal-accelerated).

    Preferred default: a single static binary, ~1-2s mmap'd model load, no
    Python import cost, and native segment timestamps. One process transcribes
    a whole file — the model loads once, not once per segment.
    """

    key = "whisper-cpp"
    label = "whisper.cpp (whisper-cli)"
    default_model = "large-v3-turbo-q5_0"
    models = {
        m.key: m
        for m in [
            _ggml(
                "large-v3-turbo-q5_0",
                "ggml-large-v3-turbo-q5_0.bin",
                547,
                "recommended: multilingual large-v3 turbo, smallest footprint",
            ),
            _ggml(
                "large-v3-turbo-q8_0",
                "ggml-large-v3-turbo-q8_0.bin",
                833,
                "turbo with a little more headroom than q5_0",
            ),
            _ggml(
                "large-v3-turbo",
                "ggml-large-v3-turbo.bin",
                1549,
                "unquantized turbo; rarely worth the extra GB over q8_0",
            ),
            _ggml(
                "large-v3-q5_0",
                "ggml-large-v3-q5_0.bin",
                1031,
                "non-turbo large-v3: several times slower, marginal English gain",
            ),
            _ggml(
                "medium.en-q5_0",
                "ggml-medium.en-q5_0.bin",
                514,
                "English-only medium; turbo-q5_0 is same size and better",
            ),
            _ggml(
                "small.en-q5_1",
                "ggml-small.en-q5_1.bin",
                181,
                "fast triage only; timestamp drift hurts anchored summaries",
            ),
            _ggml(
                "base.en",
                "ggml-base.en.bin",
                141,
                "smoke-testing the pipeline, not for real summaries",
            ),
            _ggml(
                "tiny.en",
                "ggml-tiny.en.bin",
                74,
                "smoke-testing only",
            ),
        ]
    }

    # brew ships whisper-cli; older builds and source trees use other names.
    BINARIES = ("whisper-cli", "whisper-cpp", "whisper")

    def __init__(self) -> None:
        override = os.environ.get("YT_TRANSCRIBE_WHISPER_CPP")
        candidates = (override,) if override else self.BINARIES
        self.binary = next((p for p in (which(c) for c in candidates if c) if p), None)

    def unavailable_reason(self) -> str | None:
        if self.binary:
            return None
        return (
            "whisper-cli not found. Install with: brew install whisper-cpp "
            "(or set YT_TRANSCRIBE_WHISPER_CPP=/path/to/whisper-cli)"
        )

    def ensure_weights(self, model: ModelSpec) -> Path:
        dest = MODEL_DIR / model.ref
        if dest.is_file() and dest.stat().st_size > 1 << 20:
            return dest
        download(model.url or "", dest, f"{model.key} (~{model.size_mb}MB)")
        return dest

    def ensure_vad(self) -> Path | None:
        dest = MODEL_DIR / WHISPER_VAD_FILE
        if dest.is_file() and dest.stat().st_size > 1 << 10:
            return dest
        try:
            download(f"{WHISPER_VAD_REPO}/{WHISPER_VAD_FILE}", dest, "Silero VAD")
        except RuntimeError as e:
            # VAD is a quality/speed win, not a requirement.
            print(f"VAD: unavailable ({e}); continuing without", file=sys.stderr)
            return None
        return dest

    def transcribe(
        self, wav: Path, model: ModelSpec, opts: TranscribeOptions
    ) -> list[dict]:
        weights = self.ensure_weights(model)
        lang = (opts.language or "auto").strip() or "auto"
        if _is_english_only_model(model.key) and lang not in ("en", "auto"):
            # English-only weights cannot emit other languages.
            print(
                f"LANGUAGE_WARN: model {model.key} is English-only; "
                f"--language {lang} will not produce {lang} text. "
                "Use a multilingual model (default large-v3-turbo-*) "
                "or --language en/auto.",
                file=sys.stderr,
            )
        with tempfile.TemporaryDirectory(prefix="yt-transcribe-cpp-") as td:
            out_prefix = Path(td) / "out"
            cmd = [
                str(self.binary),
                "--model", str(weights),
                "--file", str(wav),
                "--output-json",
                "--output-file", str(out_prefix),
                "--print-progress",
            ]
            cmd += ["--language", lang if lang != "auto" else "auto"]
            if opts.threads:
                cmd += ["--threads", str(opts.threads)]
            if opts.initial_prompt:
                cmd += ["--prompt", opts.initial_prompt]
            if opts.vad:
                vad_model = self.ensure_vad()
                if vad_model:
                    cmd += ["--vad", "--vad-model", str(vad_model)]
            cmd += opts.extra_args

            print(f"ASR: {self.label} · {model.key}", flush=True)
            # stdout is captured only to keep the plain-text transcript off the
            # console (we read the JSON file instead); stderr streams through so
            # progress and engine diagnostics stay visible live.
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"whisper-cli exited {proc.returncode}{_tail(proc.stdout)}"
                )

            json_path = out_prefix.with_suffix(".json")
            if not json_path.is_file():
                raise RuntimeError(f"whisper-cli wrote no JSON at {json_path}")
            data = json.loads(json_path.read_text(encoding="utf-8"))

        detected = _language_from_whisper_json(data)
        if detected:
            print(f"LANGUAGE_DETECTED: {detected}")

        segments: list[dict] = []
        for item in data.get("transcription") or []:
            if not isinstance(item, dict):
                continue
            offsets = item.get("offsets") or {}
            text = (item.get("text") or "").strip()
            if not text:
                continue
            try:
                start = float(offsets.get("from", 0)) / 1000.0
                end = float(offsets.get("to", 0)) / 1000.0
            except (TypeError, ValueError):
                continue
            segments.append({"start": start, "end": max(end, start), "text": text})
        return segments


# --- mlx-whisper ----------------------------------------------------------

class MlxWhisperBackend(Backend):
    """Whisper on MLX, run through `uvx` so nothing is installed permanently.

    Costs a Python import + weight load (~5-10s) against whisper.cpp's ~1-2s,
    and pulls MLX into a throwaway venv on first use. Kept as a second real
    backend mostly to prove the abstraction and to have an MLX-native path
    available on Apple silicon.
    """

    key = "mlx-whisper"
    label = "mlx-whisper (MLX)"
    default_model = "large-v3-turbo"
    models = {
        m.key: m
        for m in [
            ModelSpec(
                "large-v3-turbo",
                "mlx-community/whisper-large-v3-turbo",
                1600,
                "MLX turbo, fp16",
            ),
            ModelSpec(
                "large-v3-turbo-q4",
                "mlx-community/whisper-large-v3-turbo-q4",
                460,
                "4-bit turbo; smaller, some accuracy cost",
            ),
            ModelSpec(
                "large-v3",
                "mlx-community/whisper-large-v3-mlx",
                3100,
                "non-turbo large-v3; slow on an M1 Air",
            ),
        ]
    }

    def __init__(self) -> None:
        self.uvx = which("uvx") or which("uv")

    def unavailable_reason(self) -> str | None:
        if not self.uvx:
            return "uvx not found (ships with uv). Install uv, or use --backend whisper-cpp"
        return None

    def transcribe(
        self, wav: Path, model: ModelSpec, opts: TranscribeOptions
    ) -> list[dict]:
        base = [str(self.uvx)]
        if Path(str(self.uvx)).name == "uv":
            base += ["tool", "run"]
        with tempfile.TemporaryDirectory(prefix="yt-transcribe-mlx-") as td:
            cmd = base + [
                "--from", "mlx-whisper",
                "mlx_whisper",
                str(wav),
                "--model", model.ref,
                "--output-format", "json",
                "--output-dir", td,
            ]
            lang = (opts.language or "auto").strip() or "auto"
            if lang and lang != "auto":
                cmd += ["--language", lang]
            if opts.initial_prompt:
                cmd += ["--initial-prompt", opts.initial_prompt]
            cmd += opts.extra_args

            print(f"ASR: {self.label} · {model.key}", flush=True)
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"mlx_whisper exited {proc.returncode}{_tail(proc.stdout)}"
                )

            produced = sorted(Path(td).glob("*.json"))
            if not produced:
                raise RuntimeError("mlx_whisper wrote no JSON output")
            data = json.loads(produced[0].read_text(encoding="utf-8"))

        detected = data.get("language") or _language_from_whisper_json(data)
        if detected:
            print(f"LANGUAGE_DETECTED: {detected}")

        segments: list[dict] = []
        for item in data.get("segments") or []:
            if not isinstance(item, dict):
                continue
            text = (item.get("text") or "").strip()
            if not text:
                continue
            try:
                start = float(item.get("start", 0))
                end = float(item.get("end", start))
            except (TypeError, ValueError):
                continue
            segments.append({"start": start, "end": max(end, start), "text": text})
        return segments


BACKENDS: dict[str, type[Backend]] = {
    WhisperCppBackend.key: WhisperCppBackend,
    MlxWhisperBackend.key: MlxWhisperBackend,
}


def build_backend(key: str) -> Backend:
    if key not in BACKENDS:
        known = ", ".join(BACKENDS)
        raise RuntimeError(f"unknown backend {key!r}. Known: {known}")
    return BACKENDS[key]()


def print_model_catalog() -> None:
    for key, cls in BACKENDS.items():
        backend = cls()
        reason = backend.unavailable_reason()
        status = "available" if reason is None else f"unavailable — {reason}"
        print(f"\n{backend.label}  [--backend {key}]")
        print(f"  status: {status}")
        for m in backend.models.values():
            star = " *" if m.key == backend.default_model else "  "
            print(f" {star} {m.key:<24} ~{m.size_mb:>5}MB  {m.note}")
    print("\n* = backend default. Pick with --asr-model <key>.")


# ---------------------------------------------------------------------------
# audio acquisition
# ---------------------------------------------------------------------------

def download_audio(video_id: str, cookies_path: Path | None, workdir: Path) -> Path:
    """Pull bestaudio for a video id into workdir, returning the file path."""
    ytdlp = which(YTDLP_BIN)
    if not ytdlp:
        raise RuntimeError(f"{YTDLP_BIN} not found (needed to fetch YouTube audio)")
    out_tmpl = str(workdir / "audio.%(ext)s")
    cmd = [
        ytdlp,
        *YTDLP_BASE_ARGS,
        "--format", "bestaudio/best",
        "--extract-audio",
        "--no-playlist",
        "--output", out_tmpl,
        *ytdlp_cookie_args(cookies_path),
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    print("AUDIO: downloading via yt-dlp…", flush=True)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout or "").strip().splitlines()[-5:])
        raise RuntimeError(f"yt-dlp audio download failed (exit {proc.returncode})\n{tail}")
    produced = [p for p in sorted(workdir.iterdir()) if p.name.startswith("audio.")]
    if not produced:
        raise RuntimeError("yt-dlp reported success but produced no audio file")
    return produced[0]


def to_wav16k(src: Path, workdir: Path) -> Path:
    """Transcode any input to the 16 kHz mono PCM WAV whisper expects."""
    ffmpeg = which(FFMPEG_BIN)
    if not ffmpeg:
        raise RuntimeError(f"{FFMPEG_BIN} not found (needed to normalize audio)")
    wav = workdir / "audio16k.wav"
    cmd = [
        ffmpeg, "-nostdin", "-y", "-loglevel", "error",
        "-i", str(src),
        "-vn", "-ac", "1", "-ar", str(TARGET_RATE),
        "-c:a", "pcm_s16le",
        str(wav),
    ]
    print(f"AUDIO: normalizing to {TARGET_RATE} Hz mono WAV…", flush=True)
    proc = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {proc.returncode}): {(proc.stderr or '').strip()[:400]}"
        )
    if not wav.is_file() or wav.stat().st_size < 1024:
        raise RuntimeError("ffmpeg produced no usable WAV")
    return wav


def probe_duration(path: Path) -> float | None:
    ffprobe = which(FFPROBE_BIN)
    if not ffprobe:
        return None
    proc = subprocess.run(
        [
            ffprobe, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    try:
        return float((proc.stdout or "").strip())
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# quality assessment
# ---------------------------------------------------------------------------

# Hiragana, Katakana, CJK ideographs, Hangul — space-less scripts for token counts.
_CJK_LIKE_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]"
)


def _language_from_whisper_json(data: dict) -> str | None:
    """Pull detected/forced language from whisper.cpp or mlx JSON if present."""
    result = data.get("result")
    if isinstance(result, dict):
        lang = result.get("language")
        if lang:
            return str(lang)
    lang = data.get("language")
    if lang:
        return str(lang)
    params = data.get("params")
    if isinstance(params, dict) and params.get("language"):
        return str(params["language"])
    return None


def _is_english_only_model(model_key: str) -> bool:
    """True for whisper.cpp keys like tiny.en / medium.en-q5_0 / small.en-q5_1."""
    k = model_key.lower()
    return ".en" in k or k.endswith("en")


def _norm(text: str) -> str:
    """Normalize segment text for duplicate detection across scripts.

    Earlier this stripped everything but ASCII, which made every CJK segment
    compare as empty — so Mandarin/Japanese loops were invisible to the
    repetition gate. Keep letters from any script (Unicode \\w) plus digits.
    """
    t = text.casefold().strip()
    t = re.sub(r"[^\w\s]+", "", t, flags=re.UNICODE)
    return re.sub(r"\s+", " ", t).strip()


def _token_count(text: str) -> int:
    """Rough token count: whitespace words for spaced scripts, chars for CJK."""
    if not text:
        return 0
    cjk_chars = len(_CJK_LIKE_RE.findall(text))
    remainder = _CJK_LIKE_RE.sub(" ", text)
    latin_words = len(remainder.split())
    return cjk_chars + latin_words


def assess_quality(
    segments: list[dict],
    duration: float | None,
    *,
    language: str | None = None,
) -> dict:
    """Score a transcript for the two ways ASR fails on non-speech audio.

    speech_ratio  — how much of the runtime the segments actually cover. With
                    VAD on, music and silence are skipped, so a low ratio is a
                    strong "this isn't a talking video" signal.
    repetition    — share of segments whose text duplicates another segment.
                    Whisper answers non-speech (and wrong-language audio) by
                    looping one plausible line.
    """
    texts = [_norm(s["text"]) for s in segments if _norm(s.get("text", ""))]
    covered = sum(max(s["end"] - s["start"], 0.0) for s in segments)
    speech_ratio = (covered / duration) if duration and duration > 0 else None

    repetition = 0.0
    longest_run = 0
    if texts:
        repetition = 1.0 - (len(set(texts)) / len(texts))
        run = 1
        for prev, cur in zip(texts, texts[1:]):
            run = run + 1 if cur == prev else 1
            longest_run = max(longest_run, run)
        longest_run = max(longest_run, 1)

    reasons: list[str] = []
    if not segments:
        reasons.append("no speech segments at all")
    if speech_ratio is not None and speech_ratio < MIN_SPEECH_RATIO:
        reasons.append(
            f"speech covers only {speech_ratio:.0%} of the runtime "
            f"(threshold {MIN_SPEECH_RATIO:.0%})"
        )
    if repetition > MAX_REPETITION:
        reasons.append(
            f"{repetition:.0%} of segments are duplicates "
            f"(threshold {MAX_REPETITION:.0%}) — likely a hallucination loop"
        )
    if longest_run >= 10:
        reasons.append(f"{longest_run} identical segments in a row")

    # Wrong forced language (e.g. --language en on Mandarin) produces the same
    # loop pattern as music. Surface a fix when the operator pinned a language.
    lang = (language or "").strip().lower()
    if reasons and lang and lang not in ("auto", ""):
        reasons.append(
            f"language was forced to {lang!r}; if the audio is another "
            "language (or bilingual), re-run with --language auto"
        )

    return {
        "speech_ratio": speech_ratio,
        "repetition": repetition,
        "longest_run": longest_run,
        "segments": len(segments),
        "words": sum(_token_count(s.get("text", "")) for s in segments),
        "suspect": bool(reasons),
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

def to_entries(segments: list[dict]) -> list[dict]:
    """Map ASR segments to yt-summarize's cache entry shape.

    yt-summarize regroups these into ~45s windows itself, so segment-level
    granularity is all that is needed — word timestamps would be discarded.
    """
    return [
        {"start": float(s["start"]), "seconds": int(s["start"]), "text": s["text"]}
        for s in segments
    ]


def cache_path_for(video_id: str) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", video_id)
    return CACHE_DIR / f"{safe_id}.transcript.json"


def default_transcript_path(video_id: str) -> Path:
    """Cwd plain-text path: <video-id>.txt"""
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", video_id)
    return Path.cwd() / f"{safe_id}.txt"


def resolve_out_path(video_id: str, explicit: str | None) -> Path:
    if explicit is not None and str(explicit).strip():
        return Path(str(explicit)).expanduser()
    return default_transcript_path(video_id)


def load_cache_entries(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"cache is not a JSON list: {path}")
    return data


def write_cache(video_id: str, entries: list[dict]) -> Path:
    path = cache_path_for(video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries), encoding="utf-8")
    tmp.replace(path)
    return path


def write_transcript_text(path: Path, entries: list[dict]) -> None:
    """One line per segment: [HH:MM:SS] text"""
    lines = []
    for e in entries:
        start = e.get("start", e.get("seconds", 0))
        try:
            start_f = float(start)
        except (TypeError, ValueError):
            start_f = 0.0
        text = (e.get("text") or "").strip()
        lines.append(f"[{format_timestamp(start_f)}] {text}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe a YouTube video or local media file with a local ASR "
            "model. Writes a plain [HH:MM:SS] .txt transcript (cwd) and "
            "optionally yt-summarize's JSON cache."
        )
    )
    parser.add_argument(
        "source",
        nargs="?",
        help="YouTube video ID/URL, or a path to a local audio/video file",
    )
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        choices=sorted(BACKENDS),
        help=f"ASR engine (default: {DEFAULT_BACKEND})",
    )
    parser.add_argument(
        "--asr-model",
        default=os.environ.get("YT_TRANSCRIBE_MODEL"),
        help="Model key within the backend (see --list-models)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List backends and their models with availability, then exit",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help=(
            f"Spoken language code (e.g. en, zh, ja), or 'auto' to detect "
            f"(default: {DEFAULT_LANGUAGE}). Forcing the wrong language "
            "often causes repetition loops — prefer auto unless sure."
        ),
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable Silero VAD (VAD cuts hallucination on music/silence)",
    )
    parser.add_argument("--threads", type=int, help="ASR threads (engine default if unset)")
    parser.add_argument(
        "--prompt",
        help="Initial prompt to bias spelling of names/jargon",
    )
    parser.add_argument(
        "--video-id",
        help=(
            "Cache key to write under. Required for local files unless the "
            "filename is already a video id"
        ),
    )
    parser.add_argument(
        "--cookies",
        help="Netscape cookies.txt for age/region-restricted videos",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-transcribe even when a cached transcript already exists",
    )
    parser.add_argument(
        "--no-cache-write",
        action="store_true",
        help="Do not write the yt-summarize transcript cache",
    )
    parser.add_argument(
        "--out",
        help=(
            "Plain-text transcript path ([HH:MM:SS] per line). "
            "Default: ./<video-id>.txt in the current working directory"
        ),
    )
    parser.add_argument(
        "--keep-audio",
        metavar="DIR",
        help="Keep the downloaded/normalized audio in DIR instead of a temp dir",
    )
    parser.add_argument(
        "--engine-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra raw argument passed through to the ASR engine (repeatable)",
    )
    args = parser.parse_args()

    if args.list_models:
        print_model_catalog()
        return

    if not args.source:
        print("Error: a video ID/URL or a local file path is required.", file=sys.stderr)
        sys.exit(EXIT_FAIL)

    # Resolve the source into either a YouTube id or a local file.
    local_file: Path | None = None
    candidate = Path(args.source).expanduser()
    if candidate.is_file():
        local_file = candidate
        video_id = args.video_id or re.sub(r"[^A-Za-z0-9_-]", "_", candidate.stem)
    else:
        video_id = extract_video_id(args.source) or ""
        if not video_id:
            print(
                f"Error: {args.source!r} is neither a readable file nor a "
                "YouTube ID/URL.",
                file=sys.stderr,
            )
            sys.exit(EXIT_FAIL)
        if args.video_id:
            video_id = args.video_id

    print(f"SOURCE: {'file ' + str(local_file) if local_file else 'youtube ' + video_id}")
    print(f"CACHE_KEY: {video_id}")

    # Validate the backend and model key before the cache shortcut below: both
    # are pure dict lookups, and a typo'd --asr-model must not slip through as
    # exit 0 just because a transcript happened to be cached already.
    backend = build_backend(args.backend)
    try:
        model = backend.resolve_model(args.asr_model)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(EXIT_FAIL)

    out_path = resolve_out_path(video_id, args.out)
    cache_file = cache_path_for(video_id)

    # Cache hit: re-export plain text from the JSON; no ASR re-run.
    if cache_file.exists() and not args.force and not args.no_cache_write:
        print(f"TRANSCRIPT_CACHE: hit ({cache_file})")
        try:
            entries = load_cache_entries(cache_file)
        except (OSError, json.JSONDecodeError, ValueError) as e:
            print(f"Error reading cache: {e}", file=sys.stderr)
            sys.exit(EXIT_FAIL)
        write_transcript_text(out_path, entries)
        print(f"TRANSCRIPT_FILE: {out_path.resolve()}")
        print(f"SEGMENTS: {len(entries)}")
        preview = entries[: min(3, len(entries))]
        print("PREVIEW:")
        for e in preview:
            text = (e.get("text") or "")
            if len(text) > 100:
                text = text[:97] + "…"
            start = e.get("start", e.get("seconds", 0))
            try:
                start_f = float(start)
            except (TypeError, ValueError):
                start_f = 0.0
            print(f"  [{format_timestamp(start_f)}] {text}")
        print("Pass --force to re-transcribe.")
        return

    # Engine availability is checked only once we know we must actually run:
    # a cached transcript needs nothing on PATH.
    reason = backend.unavailable_reason()
    if reason:
        print(f"Error: backend {backend.key} unavailable — {reason}", file=sys.stderr)
        sys.exit(EXIT_FAIL)

    print(f"BACKEND: {backend.key}")
    print(f"ASR_MODEL: {model.key}")
    print(f"LANGUAGE: {args.language}")

    try:
        cookies_path = resolve_cookies_path(args.cookies)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(EXIT_FAIL)
    print(f"COOKIES: {cookies_path if cookies_path else 'none'}")

    opts = TranscribeOptions(
        language=args.language,
        vad=not args.no_vad,
        threads=args.threads,
        initial_prompt=args.prompt,
        extra_args=list(args.engine_arg),
    )

    keep_dir = Path(args.keep_audio).expanduser() if args.keep_audio else None
    if keep_dir:
        keep_dir.mkdir(parents=True, exist_ok=True)
    tmp_ctx = (
        tempfile.TemporaryDirectory(prefix="yt-transcribe-")
        if keep_dir is None
        else None
    )
    workdir = keep_dir if keep_dir else Path(tmp_ctx.name)  # type: ignore[union-attr]

    try:
        try:
            src = local_file if local_file else download_audio(
                video_id, cookies_path, workdir
            )
            wav = to_wav16k(src, workdir)
        except RuntimeError as e:
            print(f"Error preparing audio: {e}", file=sys.stderr)
            sys.exit(EXIT_FAIL)

        duration = probe_duration(wav)
        if duration:
            print(f"DURATION: {format_timestamp(duration)}")

        try:
            segments = backend.transcribe(wav, model, opts)
        except (RuntimeError, json.JSONDecodeError, OSError) as e:
            print(f"Error during transcription: {e}", file=sys.stderr)
            sys.exit(EXIT_FAIL)
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    if not segments:
        print(
            "ASR_EMPTY: the engine returned no speech segments. This audio "
            "carries no recognizable speech (music-only or silent).",
            file=sys.stderr,
        )
        print("ASR_QUALITY: suspect — no speech segments")
        sys.exit(EXIT_SUSPECT)

    entries = to_entries(segments)
    quality = assess_quality(segments, duration, language=args.language)

    print(f"SEGMENTS: {quality['segments']}")
    print(f"WORDS: {quality['words']}")
    if quality["speech_ratio"] is not None:
        print(f"SPEECH_RATIO: {quality['speech_ratio']:.0%}")
    print(f"REPETITION: {quality['repetition']:.0%}")

    if not args.no_cache_write:
        written = write_cache(video_id, entries)
        print(f"TRANSCRIPT_CACHE: wrote {written}")

    write_transcript_text(out_path, entries)
    print(f"TRANSCRIPT_FILE: {out_path.resolve()}")

    preview = entries[: min(3, len(entries))]
    print("PREVIEW:")
    for e in preview:
        text = e["text"] if len(e["text"]) <= 100 else e["text"][:97] + "…"
        print(f"  [{format_timestamp(e['start'])}] {text}")

    if quality["suspect"]:
        print("ASR_QUALITY: suspect")
        for r in quality["reasons"]:
            print(f"  - {r}")
        print(
            "This usually means the audio is music, non-speech, or the wrong "
            "language was forced (try --language auto). Review the "
            "TRANSCRIPT_FILE before treating it as speech."
        )
        sys.exit(EXIT_SUSPECT)

    print("ASR_QUALITY: ok")


if __name__ == "__main__":
    main()
