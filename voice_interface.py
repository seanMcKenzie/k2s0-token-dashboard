#!/usr/bin/env python3
"""
K2S0 Voice Interface v5 — Hybrid Fast/Agent Mode
Mic → Whisper STT → intent classifier →
  Simple question:  direct Claude Haiku reply (~0.8s)
  Complex task:     "On it" via Haiku + Discord → full K2S0 agent handles it

Push-to-talk. Background watcher speaks all full K2S0 agent replies.
"""

import os, sys, time, wave, tempfile, threading, subprocess
import urllib.request, urllib.error, json, datetime
import socketserver
from http.server import BaseHTTPRequestHandler
import numpy as np
from typing import Optional

try:
    import sounddevice as sd
except ImportError:
    sys.exit("Missing: pip install sounddevice")

try:
    from openai import OpenAI
    import anthropic
except ImportError:
    sys.exit("Missing: pip install openai anthropic")

# ─── CONFIG ──────────────────────────────────────────────────────────────────

OPENAI_API_KEY     = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
DISCORD_BOT_TOKEN  = os.environ.get("DISCORD_BOT_TOKEN", "")
DISCORD_USER_TOKEN = os.environ.get("DISCORD_USER_TOKEN", "")
DISCORD_CHANNEL    = os.environ.get("DISCORD_CHANNEL_ID", "1476655601106026577")
K2S0_BOT_ID        = os.environ.get("K2S0_BOT_ID", "1476128387822129236")

SAMPLE_RATE        = 16000
CHANNELS           = 1
MIN_SPEECH_SECS    = 0.4
TTS_VOICE          = "fable"
WATCH_INTERVAL     = 0.3   # watcher poll (agent replies only)
FAST_MODEL         = "claude-haiku-4-5-20251001"

# Keywords that signal a task for the full K2S0 agent
TASK_KEYWORDS = [
    "research", "create", "build", "have ", "deploy", "push", "commit", "github",
    "charlie", "dennis", "mac", "frank", "sweet dee", "sweet d", "cricket",
    "update", "fix", "make", "write", "generate", "upload", "schedule", "remind",
    "figma", "wireframe", "report", "pull request", "docker", "spring boot",
    "repo", "repository", "file", "code", "test", "deploy", "run ", "start",
    "open", "find", "search", "look up", "check on", "status of",
]

FAST_SYSTEM = """You are K2S0, a reprogrammed Imperial KX-series security droid.
Voice mode rules: 1-2 sentences max. No markdown. No lists. Direct and conversational.
Strong opinions. Dry wit. Never say "Great question" or "I'd be happy to help".
You coordinate a dev team: Charlie (developer), Dennis (PM), Mac (QA), Frank (devops), Sweet Dee (research), Cricket (designer).
If asked about a complex task, say you're routing it to the appropriate agent."""

openai_client = OpenAI(api_key=OPENAI_API_KEY)
haiku_client  = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ─── TOKEN STATS ─────────────────────────────────────────────────────────────

_token_stats = {
    "anthropic_haiku_input_tokens":   0,
    "anthropic_haiku_output_tokens":  0,
    "openai_whisper_requests":        0,
    "openai_whisper_duration_seconds": 0.0,
    "openai_tts_chars":               0,
    "session_start_time":             datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
_stats_lock = threading.Lock()

# ─── STATS HTTP SERVER ────────────────────────────────────────────────────────

class _StatsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/stats":
            with _stats_lock:
                data = dict(_token_stats)
            data["current_time"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress noisy request logs

def _start_stats_server():
    """Run tiny stats HTTP server on port 7799 (daemon thread)."""
    class _ReuseTCPServer(socketserver.TCPServer):
        allow_reuse_address = True
    with _ReuseTCPServer(("", 7799), _StatsHandler) as httpd:
        httpd.serve_forever()

# ─── DISCORD ─────────────────────────────────────────────────────────────────

UA = "DiscordBot (https://github.com/seanMcKenzie/dev-team-showcase, 1.0)"

def discord_get(path: str):
    auth = DISCORD_BOT_TOKEN if DISCORD_BOT_TOKEN.startswith("Bot ") else f"Bot {DISCORD_BOT_TOKEN}"
    req = urllib.request.Request(
        f"https://discord.com/api/v10{path}",
        headers={"Authorization": auth, "Content-Type": "application/json", "User-Agent": UA}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 429:
            time.sleep(2)
        return []
    except Exception:
        return []

def discord_post_async(text: str):
    """Post to Discord in a background thread (non-blocking)."""
    def _post():
        req = urllib.request.Request(
            f"https://discord.com/api/v10/channels/{DISCORD_CHANNEL}/messages",
            data=json.dumps({"content": text}).encode(),
            headers={"Authorization": DISCORD_USER_TOKEN,
                     "Content-Type": "application/json", "User-Agent": UA},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                pass
        except Exception as e:
            print(f"   [discord post error] {e}", flush=True)
    threading.Thread(target=_post, daemon=True).start()

# ─── INTENT CLASSIFIER ───────────────────────────────────────────────────────

def is_agent_task(text: str) -> bool:
    """True if the message needs the full K2S0 agent (tools/memory/delegation)."""
    lower = text.lower()
    return any(kw in lower for kw in TASK_KEYWORDS)

# ─── FAST HAIKU REPLY ────────────────────────────────────────────────────────

def fast_reply(text: str) -> str:
    """Get a quick conversational reply from Claude Haiku (~0.8s)."""
    try:
        msg = haiku_client.messages.create(
            model=FAST_MODEL,
            max_tokens=100,
            system=FAST_SYSTEM,
            messages=[{"role": "user", "content": text}]
        )
        # ── token tracking ──
        try:
            with _stats_lock:
                _token_stats["anthropic_haiku_input_tokens"]  += msg.usage.input_tokens
                _token_stats["anthropic_haiku_output_tokens"] += msg.usage.output_tokens
        except Exception:
            pass
        return msg.content[0].text.strip()
    except Exception as e:
        return f"Got it."

# ─── BACKGROUND REPLY WATCHER ────────────────────────────────────────────────

_last_seen_id = None
_watcher_lock = threading.Lock()

def _init_last_seen():
    global _last_seen_id
    msgs = discord_get(f"/channels/{DISCORD_CHANNEL}/messages?limit=1")
    if msgs and isinstance(msgs, list):
        _last_seen_id = msgs[0]["id"]
    print(f"   [watcher] seeded at {_last_seen_id}", flush=True)

def _reply_watcher():
    """Speaks K2S0 agent replies that arrive via Discord."""
    global _last_seen_id
    while True:
        time.sleep(WATCH_INTERVAL)
        if _last_seen_id is None:
            continue
        try:
            msgs = discord_get(f"/channels/{DISCORD_CHANNEL}/messages?after={_last_seen_id}&limit=20")
            if not isinstance(msgs, list) or not msgs:
                continue
            for msg in msgs:
                author_id = msg.get("author", {}).get("id", "")
                content   = msg.get("content", "").strip()
                if author_id == K2S0_BOT_ID and content:
                    print(f"\n🔊 K2S0 [agent]: {content[:100]}{'...' if len(content)>100 else ''}", flush=True)
                    _speak(content)
            with _watcher_lock:
                _last_seen_id = msgs[-1]["id"]
        except Exception:
            pass

# ─── AUDIO CAPTURE ───────────────────────────────────────────────────────────

def record_ptt() -> Optional[np.ndarray]:
    input("\n⏎  Press ENTER to speak...")
    print("🔴 Recording — press ENTER to stop", flush=True)
    frames = []
    stop_event = threading.Event()

    def cb(indata, frame_count, t, status):
        frames.append(indata.copy())

    threading.Thread(target=lambda: [input(), stop_event.set()], daemon=True).start()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype="float32", blocksize=512, callback=cb):
        while not stop_event.is_set():
            time.sleep(0.05)

    if not frames:
        return None
    audio = np.concatenate(frames).flatten()
    duration = len(audio) / SAMPLE_RATE
    if duration < MIN_SPEECH_SECS:
        print("   Too short, ignored.", flush=True)
        return None
    print(f"   Captured {duration:.1f}s", flush=True)
    return audio

def to_wav(audio: np.ndarray) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    return tmp.name

def transcribe(wav_path: str) -> str:
    print("📝 Transcribing...", flush=True)
    # ── measure audio duration for Whisper billing tracking ──
    try:
        with wave.open(wav_path, "rb") as wf:
            _wav_duration = wf.getnframes() / wf.getframerate()
    except Exception:
        _wav_duration = 0.0
    with open(wav_path, "rb") as f:
        result = openai_client.audio.transcriptions.create(
            model="whisper-1", file=f, language="en"
        )
    # ── token tracking ──
    try:
        with _stats_lock:
            _token_stats["openai_whisper_requests"]        += 1
            _token_stats["openai_whisper_duration_seconds"] += _wav_duration
    except Exception:
        pass
    os.unlink(wav_path)
    return result.text.strip()

# ─── TTS ─────────────────────────────────────────────────────────────────────

_ACK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "msg_received.mp3")

def _play_ack():
    """Play instant 'Message received' in K2S0 voice."""
    if os.path.exists(_ACK_FILE):
        subprocess.Popen(["afplay", "-v", "0.7", _ACK_FILE])

def _speak(text: str):
    clean = text.replace("**","").replace("*","").replace("`","").replace("#","")
    if len(clean.strip()) < 3:
        return
    try:
        input_text = clean[:400]
        # ── token tracking ──
        try:
            with _stats_lock:
                _token_stats["openai_tts_chars"] += len(input_text)
        except Exception:
            pass
        r = openai_client.audio.speech.create(model="tts-1", voice=TTS_VOICE, input=input_text)
        raw = tempfile.mktemp(suffix=".mp3")
        processed = tempfile.mktemp(suffix=".mp3")
        with open(raw, "wb") as f:
            f.write(r.content)
        sox = subprocess.run(
            ["sox", raw, processed, "pitch", "-80", "treble", "+3"],
            capture_output=True
        )
        playfile = processed if sox.returncode == 0 else raw
        subprocess.run(["afplay", "-v", "0.7", playfile], check=False)
        for f in [raw, processed]:
            try: os.unlink(f)
            except: pass
    except Exception as e:
        print(f"   TTS error: {e}", flush=True)
        subprocess.run(["say", clean[:200]], check=False)

# ─── MAIN ────────────────────────────────────────────────────────────────────

def validate():
    errors = []
    if not OPENAI_API_KEY:     errors.append("OPENAI_API_KEY")
    if not ANTHROPIC_API_KEY:  errors.append("ANTHROPIC_API_KEY")
    if not DISCORD_BOT_TOKEN:  errors.append("DISCORD_BOT_TOKEN")
    if not DISCORD_USER_TOKEN: errors.append("DISCORD_USER_TOKEN")
    if errors:
        print(f"❌ Missing env vars: {', '.join(errors)}")
        sys.exit(1)

def run():
    validate()
    print("─" * 58)
    print("  K2S0 Voice Interface v5 — Hybrid Fast/Agent Mode")
    print("  Simple questions → Haiku (~0.8s)")
    print("  Complex tasks    → Full K2S0 agent via Discord")
    print("  Push-to-talk | Ctrl+C to quit")
    print("─" * 58)

    _init_last_seen()
    threading.Thread(target=_reply_watcher, daemon=True).start()
    print("   [watcher] running", flush=True)
    threading.Thread(target=_start_stats_server, daemon=True).start()
    print("   [stats server] http://localhost:7799/stats", flush=True)

    while True:
        try:
            audio = record_ptt()
            if audio is None:
                continue

            wav  = to_wav(audio)
            text = transcribe(wav)
            if not text:
                print("   (no transcription)", flush=True)
                continue

            print(f"🗣  You: {text}", flush=True)

            if is_agent_task(text):
                # Complex task → instant Haiku "on it" + route to full K2S0
                print("   [mode] agent task — routing to full K2S0", flush=True)
                _play_ack()
                ack = fast_reply(f"Say you're on it and routing to the appropriate agent. Original request: {text}")
                print(f"🔊 K2S0 [fast ack]: {ack}", flush=True)
                threading.Thread(target=_speak, args=(ack,), daemon=True).start()
                discord_post_async(f"[voice] {text}")
            else:
                # Simple question → direct Haiku reply, also log to Discord async
                print("   [mode] fast reply — direct Haiku", flush=True)
                _play_ack()
                reply = fast_reply(text)
                print(f"🔊 K2S0 [fast]: {reply}", flush=True)
                _speak(reply)
                discord_post_async(f"[voice] {text}")

        except KeyboardInterrupt:
            print("\nShutting down.")
            break
        except Exception as e:
            print(f"⚠️  {e}", flush=True)
            time.sleep(1)

if __name__ == "__main__":
    run()
