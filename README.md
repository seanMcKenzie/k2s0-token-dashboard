# K2S0 Token Dashboard — Live

Real-time token usage dashboard for the K2S0 voice interface. Tracks API costs across all providers as you talk.

## What it tracks

| Provider | Metric | Rate |
|----------|--------|------|
| 🧠 Anthropic Haiku | Input + output tokens | $0.00080/1K in · $0.00400/1K out |
| 🎤 OpenAI Whisper | Transcription requests | $0.006/min |
| 🔊 OpenAI TTS | Characters sent | $0.015/1K chars |

## How to use

1. Start the voice interface (it runs a local HTTP server on port 7799):
   ```bash
   cd ~/.openclaw/workspace/voice_interface
   source venv/bin/activate
   set -a && source ~/.openclaw/.env && set +a
   python voice_interface.py
   ```

2. Open `token-dashboard.html` in your browser — it auto-polls `http://localhost:7799/stats` every 2 seconds.

3. Talk to K2S0. Watch the numbers update live.

## Files

- `token-dashboard.html` — the live dashboard (open this in a browser)
- `voice_interface.py` — K2S0 voice interface v5 with token tracking + HTTP stats server

## Team

- **Charlie** (`agentId: developer`) — built this dashboard
- **K2S0** (`agentId: main`) — coordinator
- Repo: [dev-team-showcase](https://github.com/seanMcKenzie/dev-team-showcase)
