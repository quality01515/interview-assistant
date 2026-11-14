# AetherStack Interview Assistant

Live interview helper. Load your resume, click **Start interview**, and the app listens to the conversation, transcribes it, and streams a short suggested answer — in your voice, grounded in your resume — whenever the interviewer asks a question.

Two apps live in this repo:

| App | Entry point | Runs on | Purpose |
|---|---|---|---|
| **Interview Assistant** | [app.py](app.py) | The machine you read answers from | Audio → STT → question detection → LLM → UI |
| **Sender** | [sender/sender_app.py](sender/sender_app.py) | The interview computer (network mode only) | Tray-only app that streams mic + system audio to the Assistant over WebSocket |

Windows is the primary platform; macOS (11+) is supported and built in CI.

## Modes

The mode picker appears on every launch.

| Mode | Audio | Speaker labels | When to use |
|---|---|---|---|
| **Same laptop** | Local mic + system-audio loopback (WASAPI on Windows, a virtual device such as BlackHole on macOS) | Known from the channel: mic = candidate, loopback = interviewer | You run the interview and the assistant on one machine. Never screen-share the assistant window. |
| **Helper laptop — network** (recommended) | Sender streams the interview computer's mic + loopback over the LAN | Known from the channel | A second laptop; the Sender runs on the interview computer. Two clean streams, no diarization. |
| **Helper laptop — acoustic** | One microphone hears both people | Inferred by voice (see below) | Nothing can be installed on the interview computer. Less reliable in noisy rooms. |

**Acoustic mode speaker detection:**
- **With voice enrollment** (offered the first time you pick this mode; *Re-record voice…* on the setup screen): each finalized clip is embedded with Resemblyzer, clustered locally, and the cluster closest to your enrolled voice is labelled *candidate*.
- **Without enrollment**: clusters are labelled behaviourally by the `AutoLabeler` (question rate, utterance length, who spoke first).
- **Swap speakers** in the interview view flips the labels if a guess is wrong.

## Requirements and API keys

- Python 3.10–3.12 (PySide6 does not support 3.13+; the Anthropic and OpenAI SDKs need 3.10+).
- An LLM key — **either** Anthropic **or** OpenAI. Chosen in the first-launch dialog (*Change API…* later).
  - **Anthropic** (default): Claude Opus 5.5 for live answers at `low` effort (fast), and at `medium` effort for **Deeper**. Every request opts into server-side refusal fallbacks, so a false-positive safety decline is retried on another model rather than leaving the answer empty.
  - **OpenAI**: GPT-6.1 Terra with reasoning off (`none`) for live answers, GPT-6.1 Sol at `medium` for **Deeper** and the assessment.
- Optional: a **Deepgram** key for cloud STT (the default, lowest latency). Without one the app transparently uses on-device **whisper.cpp**. Set it under *STT settings…*.

Keys can also come from environment variables, which always win over the config file and are never written to disk:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
$env:OPENAI_API_KEY    = "sk-..."
$env:DEEPGRAM_API_KEY  = "..."
```

## Install

**Windows**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
.\install.ps1
```

`install.ps1` installs `requirements.txt`, then installs `resemblyzer` with `--no-deps` — its `webrtcvad` dependency has no Windows wheel and needs MSVC, and the app never calls that code path.

**macOS**

```bash
chmod +x install_macos.sh && ./install_macos.sh
```

The script finds a working Python 3.10–3.12 (preferring the python.org framework build), installs Homebrew prerequisites (portaudio, plus compilers for scipy), creates `.venv`, and installs dependencies from `requirements_macos.txt`. For same-laptop mode, install a loopback device such as BlackHole (`brew install blackhole-2ch`) and select it under *Audio devices…*.

## Run from source

```powershell
.\run.bat          # python app.py
.\run_sender.bat   # python -m sender.sender_app
```

No build step is needed to test a change.

## Using the Assistant

1. First launch only: choose the LLM provider and paste its key. (A 30-day trial starts on first run; after that a license key for this computer is required — see Licensing.)
2. Pick a mode. Network mode asks for the Sender's host and port (default `8765`); acoustic mode offers voice enrollment.
3. On the setup screen, load your resume (PDF / DOCX / TXT) and optionally fill in job title, job description, and personal context (salary expectations, start date, anything the resume doesn't cover). All of it is remembered for next launch. Adjust *STT settings…* / *Audio devices…* if needed.
4. Click **Start interview**. A loading dialog fetches the whisper.cpp model if needed and pre-warms the LLM connection and prompt cache.
5. Watch the live transcript on the left and the suggested answer on the right.
6. Click **End interview** for a post-interview assessment (0–100 score, verdict, strengths, concerns), then you're returned to the setup screen.

### During the interview

| Control | Effect |
|---|---|
| **Ctrl + Space** / **Regenerate** | Answer the latest interviewer turn now |
| **Shorter** | Re-answer more briefly |
| **More technical** | Re-answer with concrete technical detail |
| **Deeper** | Re-answer with the deep model (`deep_model` / `openai_deep_model`) |
| **Swap speakers** | Invert candidate / interviewer labels from now on |
| **End interview** | Stop listening and run the assessment |
| **A− / A+**, **Ctrl + − / Ctrl + +** | Answer font size (persisted) |
| **Ctrl + \\** / **◀** | Collapse / expand the transcript panel |

Every question and answer is appended to `~/.interview_assistant/qa_log.txt`, with a separator per session.

## The Sender (network mode)

Run it on the interview computer. It has no main window — only a tray icon, so it can't leak into a screen share. It starts streaming automatically on launch with the saved devices and port; the tray menu offers Stop/Start, *Settings…* (devices + port), *Show local IP addresses*, and Quit. Settings are stored in `~/.interview_assistant/sender_config.json`.

Wire format: binary WebSocket frames, byte 0 = speaker tag (`0x01` candidate / mic, `0x02` interviewer / loopback), the rest = 16 kHz mono int16 PCM.

## How it works

```
AudioSource ──► Segmenter ──► STT backend ──► speaker resolution ──► bleed / echo filters
(dual / single     (silero VAD,    (Deepgram or       (channel tag, or          (mic gate, text echo,
 mic / network)     or pass-through  whisper.cpp)       voice clustering)         overlap checks)
                    for Deepgram)                                                        │
                                                                                        ▼
UI ◄── answer worker (streaming LLM) ◄── question detector + timers ◄────────── Transcript
```

- **STT.** Deepgram opens one streaming WebSocket per speaker and does its own endpointing, so local VAD is bypassed. whisper.cpp uses the bundled silero-vad ONNX model to segment speech and re-transcribes the growing buffer every ~0.6 s for live partials. If the configured engine fails at start, the session falls back to whisper.cpp without changing the saved setting.
- **Bleed suppression** (same-laptop / network). Candidate-channel speech is dropped when its peak loudness is under `mic_gate_rms`, when it closely matches recent interviewer text, when it's a short phrase right after the interviewer spoke, or when it overlaps live interviewer audio without being long and distinct. Dropped partials are retracted from the UI.
- **Answer triggers.** An interviewer final ending in `?` or starting with a question opener ("walk me through…", "how would you…") fires after a 700 ms debounce, so a question split by a breath becomes one LLM call. Statements of 5+ words fire after `question_silence_ms` of silence. Nothing fires while the candidate is speaking (4 s grace), during a 25 s post-answer cooldown until the candidate speaks, or when the transcript has no interviewer turns.
- **LLM context.** Resume, role, and personal context go in a prompt-cached system block (pre-warmed at start). The whole conversation is sent verbatim with each request; only when it exceeds `context_token_budget` are the oldest turns folded into a running summary by a background LLM call, always keeping at least `min_verbatim_turns` recent turns verbatim.
- **Answer streaming.** Each answer runs on its own worker thread with its own cancel event; a main-thread generation check drops any stale chunk, so a new answer never mixes with an old one.
- **Answer style.** `SYSTEM_RULES` in [pipeline/answer_prompt.py](pipeline/answer_prompt.py), shared by both providers, describes how a strong candidate actually talks: the answer first, one concrete detail from the resume, first-person ownership, the reasoning and trade-off, varied openings, everyday words. It stays short and speakable — no stock phrases, filler, buzzwords, lists, code, or invented employers and metrics.

## Configuration

Everything lives in `~/.interview_assistant/config.json` (created on first save). Defaults are in [config.py](config.py); most are also editable from the UI. Commonly tuned fields:

```jsonc
{
  "provider": "anthropic",            // "anthropic" | "openai"
  "model": "claude-opus-5-5",         // live answers (Anthropic)
  "deep_model": "claude-opus-5-5",    // "Deeper" button + end-of-interview assessment
  "effort": "low",                    // live answers: low | medium | high | xhigh | max
  "deep_effort": "medium",            // "Deeper" button
  "openai_model": "gpt-6.1-terra",    // live answers (OpenAI)
  "openai_deep_model": "gpt-6.1-sol", // "Deeper" button + assessment
  "openai_effort": "none",            // "" for a non-reasoning model
  "openai_deep_effort": "medium",
  "max_tokens": 4096,                 // cap for thinking + answer; the prompt sets answer length

  "stt_engine": "deepgram",           // "deepgram" | "whispercpp" (no key → whispercpp)
  "deepgram_model": "nova-3",
  "whisper_model": "base.en",         // tiny.en / base.en / small.en / medium.en / large-v3
  "whisper_device": "cpu",            // "cpu" | "cuda"

  "mic_device_index": null,           // null = default mic
  "loopback_device_index": null,      // null = default output loopback
  "vad_silence_ms": 800,
  "vad_threshold": 0.35,
  "mic_gate_rms": 600,                // lower if your own speech gets dropped
  "question_silence_ms": 1200,

  "context_token_budget": 6000,
  "min_verbatim_turns": 12
}
```

The config directory also holds `qa_log.txt` and `sender_config.json`. whisper.cpp models download to the Hugging Face cache (`~/.cache/huggingface/hub`) on first use.

## Testing

```powershell
python smoke_test.py
```

Offline, dependency-light checks for the transcript, question detector, auto-labeler, echo filter, health score, license, and context summarizer. Exit code 0 = pass. Audio capture, STT, and LLM calls are not covered — exercise those manually; [test_interview_script.md](test_interview_script.md) is a read-aloud script that marks which lines should and shouldn't trigger an answer.

## Building

**Windows** (PyInstaller, onedir — onefile crashes with the bundled native libraries):

```powershell
.\build.bat          # Assistant → dist\aetherstack-interview-assistant\
.\build_sender.bat   # Sender    → dist\aetherstack-sender\
```

Each script cleans only its own `build\` / `dist\` subfolders, so the two builds don't clobber each other. Zip the output folder to ship it.

**macOS**: run the *Build macOS app* workflow ([.github/workflows/build-macos.yml](.github/workflows/build-macos.yml)) manually from the GitHub Actions tab. It builds both `.app` bundles from [app_macos.spec](app_macos.spec) / [sender_macos.spec](sender_macos.spec) targeting macOS 11, uploads them as zipped artifacts, then runs [scripts/check_macos_min_version.py](scripts/check_macos_min_version.py) to confirm every bundled binary loads on macOS 11.

## Licensing

A 30-day trial starts on first launch. After that the app needs a license key that is **tied to one computer** and **expires** on a date you choose.

**How keys work.** Each key contains the computer's machine ID, an expiry date and an optional licensee name, signed with an Ed25519 **private key that only you hold**. The app ships only the matching public key ([pipeline/license_pubkey.py](pipeline/license_pubkey.py)), which can check a key but can't create one, so extracting it from the `.exe` doesn't let anyone make keys. Details are in [pipeline/license.py](pipeline/license.py).

**Issuing a key**
1. The customer opens **License…** on the setup screen (or the dialog that appears when the trial ends), clicks **Copy** next to their machine ID, and sends it to you.
2. You run, with the Python environment that has `cryptography` installed:
   ```powershell
   python tools/license_admin.py issue --machine ABCD-EFGH-IJKL-MNOP --days 30 --name "Jane Doe"
   ```
   (`--until 2027-06-30` instead of `--days` sets a fixed end date.)
3. They paste the printed `AS2-…` key into the License dialog. Renewal is the same: issue a new key for the same machine ID.

`python tools/license_admin.py inspect <key>` shows what a key contains; `machine-id` prints this computer's ID.

**The signing key** lives at `~/.aetherstack_license/signing_key.pem`, outside the repo (`*.pem` is git-ignored). **Back it up privately.** If it's lost you can't issue keys that existing builds accept; if it leaks, anyone can. `python tools/license_admin.py init` creates it if missing and writes its public half into the app; running `init` with a new key file invalidates every key issued so far.

**Other rules**
- **Clock rollback:** the app remembers the latest date it has seen. If the computer's clock is set back more than a day before that (or before a key's issue date), it refuses to run until the clock is corrected.
- **Retired shared key:** the old `XXXX-XXXX-…` key that worked on every copy is no longer accepted. Installs that already used it get 7 days from their first launch of this version, with a warning on the setup screen, to switch to a personal key.
- **Machine ID:** derived from Windows' `MachineGuid` (macOS: the hardware UUID), hashed. It changes when Windows is reinstalled, so the customer needs a new key then. VMs cloned from one image share an ID.
- **Limits:** this is an offline check. Someone who modifies the app's code can bypass it; the goal is to stop key sharing and casual copying. Deleting `config.json` also restarts the 30-day trial.

## Repo layout

```
app.py                  # Assistant entry point + pipeline controller (App)
config.py               # Config dataclass, load/save, env-var overrides
paths.py                # bundled-asset paths (source vs PyInstaller)
llm_provider.py         # LLMClient protocol + factory
claude_client.py        # Anthropic client: cached context, streaming, summary, assessment
openai_client.py        # OpenAI client, same interface
resume_loader.py        # PDF / DOCX / TXT → text
smoke_test.py           # offline tests
audio/
  source.py             # AudioSource base class
  dual_stream.py        # same-laptop: mic + loopback
  single_mic.py         # helper-acoustic: one mic
  network_source.py     # helper-network: WebSocket client to the Sender
  pyaudio_compat.py     # pyaudiowpatch (Windows) vs pyaudio (macOS) shim
  diarizer.py           # Resemblyzer embeddings + online clustering / anchor matching
  auto_labeler.py       # behavioural cluster → candidate/interviewer labels
  voice_enroll.py       # voice-enrollment recording + embedding
  _pcm.py               # mono / 16 kHz / int16 conversion
pipeline/
  vad.py, onnx_vad.py   # silero-vad (ONNX) segmenter
  stt_backend.py        # STTBackend interface + STTEvent
  stt_engines.py        # whisper.cpp backend + engine selection/factory
  deepgram_stt.py       # Deepgram streaming backend
  transcript.py         # thread-safe transcript with partials
  question_detector.py  # when to answer
  echo_filter.py        # text-overlap bleed detection
  answer_prompt.py      # system prompt for live answers (both providers)
  context_summary.py    # token-budget summary safety valve
  interview_health.py   # live health score
  evaluation.py         # end-of-interview assessment schema + prompt
  model_preloader.py    # pre-start model download / LLM warmup
  license.py            # trial, signed license keys, clock + old-key rules
  license_pubkey.py     # public half of the signing key (written by tools/license_admin.py)
  machine_id.py         # per-computer ID that keys are bound to
  types.py              # Turn
sender/
  sender_app.py         # tray app + settings dialog
  sender_streamer.py    # capture + WebSocket broadcast
ui/                     # PySide6 windows and dialogs (setup, interview, mode picker, settings…)
assets/                 # icons + silero_vad.onnx
hooks/                  # PyInstaller hooks
scripts/                # macOS icon generation + min-version check
tools/license_admin.py  # seller-only: create the signing key, issue / inspect license keys (not bundled)
*.spec, build*.bat      # PyInstaller builds
install.ps1, install_macos.sh
```

## Not built yet

- TTS (reading the answer into an earbud).
