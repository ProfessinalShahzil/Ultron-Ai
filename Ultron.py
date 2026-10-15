
import os
import re
import sys
import time
import io
import asyncio
import tempfile
import threading

import requests
import speech_recognition as sr
import edge_tts
import pygame

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_file, render_template_string

# ============== CONFIGURATION ==============

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "nvidia/nemotron-3-super-120b-a12b:free"
).strip()

API_URL = "https://openrouter.ai/api/v1/chat/completions"

FALLBACK_MODELS = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "openrouter/free",
]

ENGLISH_VOICE = "en-IN-PrabhatNeural"
HINDI_VOICE = "hi-IN-MadhurNeural"

VOICE_RATE = "+15%"
VOICE_PITCH = "-12Hz"

SILENCE_SECONDS = 1
START_TIMEOUT = 5
PHRASE_LIMIT = 12
NOISE_CALIBRATION = 0.3
MAX_TOKENS = 999

speech_lock = threading.Lock()
ai_lock = threading.Lock()

# ============== FLASK WEB APP ==============

app = Flask(__name__)

PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#070b14">
<title>ULTRON AI</title>

<style>
* { box-sizing: border-box; }

body {
    margin: 0;
    min-height: 100vh;
    background:
        radial-gradient(ellipse at top, #12223b 0%, #080d18 48%, #03050a 100%);
    color: #eaf4ff;
    font-family: Arial, Helvetica, sans-serif;
}

.app {
    max-width: 850px;
    margin: auto;
    min-height: 100vh;
    padding: 20px;
    display: flex;
    flex-direction: column;
}

header {
    display: flex;
    align-items: center;
    gap: 13px;
    padding: 10px 0 20px;
    border-bottom: 1px solid #263c53;
}

.logo {
    width: 48px;
    height: 48px;
    border: 2px solid #28d7ff;
    border-radius: 50%;
    display: grid;
    place-items: center;
    color: #28d7ff;
    font-weight: bold;
    box-shadow: 0 0 18px #00bfff44;
}

h1 {
    font-size: 22px;
    letter-spacing: 4px;
    margin: 0;
}

.subtitle {
    color: #86a5c5;
    font-size: 12px;
    margin-top: 6px;
}

.status {
    margin-left: auto;
    font-size: 12px;
    color: #54f0bb;
}

.status::before {
    content: "";
    display: inline-block;
    width: 7px;
    height: 7px;
    background: currentColor;
    border-radius: 50%;
    margin-right: 6px;
}

.orb-area {
    display: grid;
    place-items: center;
    padding: 24px 0 12px;
}

.orb {
    width: 112px;
    height: 112px;
    border: 2px solid #21c8ff;
    border-radius: 50%;
    display: grid;
    place-items: center;
    color: #48dfff;
    font-size: 29px;
    background: radial-gradient(circle, #153451, #08111f 72%);
    box-shadow: 0 0 25px #00bfff2b, inset 0 0 22px #00bfff22;
}

.orb.active {
    animation: pulse 1.2s infinite alternate;
}

@keyframes pulse {
    from { box-shadow: 0 0 12px #00bfff44; }
    to { box-shadow: 0 0 38px #00bfffaa; }
}

#statusText {
    color: #9bb5cf;
    font-size: 13px;
    text-align: center;
    min-height: 18px;
}

.chat {
    flex: 1;
    min-height: 220px;
    max-height: 45vh;
    overflow-y: auto;
    padding: 14px 0;
}

.message {
    padding: 13px 15px;
    margin: 10px 0;
    border-radius: 12px;
    line-height: 1.55;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}

.user {
    background: #102b43;
    border: 1px solid #1e5274;
    margin-left: 12%;
}

.bot {
    background: #101824;
    border: 1px solid #263b50;
    margin-right: 5%;
}

.who {
    color: #42d7ff;
    font-size: 10px;
    letter-spacing: 2px;
    margin-bottom: 7px;
}

.controls {
    border-top: 1px solid #263c53;
    padding-top: 16px;
}

.input-row {
    display: flex;
    gap: 9px;
}

input {
    flex: 1;
    min-width: 0;
    padding: 14px;
    border-radius: 10px;
    background: #0b1421;
    border: 1px solid #2b435d;
    color: white;
    font-size: 15px;
    outline: none;
}

input:focus { border-color: #28d7ff; }

button {
    cursor: pointer;
    border: 1px solid #28bde8;
    background: #102c43;
    color: #dff8ff;
    border-radius: 10px;
    padding: 12px 15px;
    font-weight: bold;
}

button:disabled {
    opacity: .5;
    cursor: wait;
}

button:hover { background: #17415e; }

.secondary-row {
    display: flex;
    gap: 9px;
    margin-top: 10px;
}

.secondary-row button {
    flex: 1;
    font-size: 12px;
}

#micBtn.recording {
    background: #713039;
    border-color: #ff6576;
}

footer {
    text-align: center;
    color: #627c96;
    font-size: 11px;
    padding: 18px 0 3px;
}

@media (max-width: 480px) {
    .app { padding: 14px; }
    h1 { font-size: 19px; }
    .orb { width: 94px; height: 94px; }
    .user { margin-left: 5%; }
    .bot { margin-right: 2%; }
    button { padding: 12px; }
}
</style>
</head>

<body>
<div class="app">
    <header>
        <div class="logo">U</div>
        <div>
            <h1>ULTRON</h1>
            <div class="subtitle">PERSONAL AI ASSISTANT</div>
        </div>
        <div class="status" id="connection">ONLINE</div>
    </header>

    <section class="orb-area">
        <div class="orb" id="orb">U</div>
    </section>
    <div id="statusText">Ready, Boss.</div>

    <main class="chat" id="chat">
        <div class="message bot">
            <div class="who">ULTRON</div>
            Online, Boss. How can I help you?
        </div>
    </main>

    <section class="controls">
        <form id="chatForm">
            <div class="input-row">
                <input id="message" autocomplete="off"
                       placeholder="Ask ULTRON anything..." required>
                <button id="sendBtn" type="submit">SEND</button>
            </div>
        </form>

        <div class="secondary-row">
            <button type="button" id="micBtn">🎙 SPEAK</button>
            <button type="button" id="voiceBtn">🔊 VOICE: ON</button>
            <button type="button" id="resetBtn">RESET CHAT</button>
        </div>
    </section>

    <footer>ULTRON AI · OpenRouter · Edge Text-to-Speech</footer>
</div>

<script>
const chat = document.getElementById("chat");
const form = document.getElementById("chatForm");
const input = document.getElementById("message");
const sendBtn = document.getElementById("sendBtn");
const micBtn = document.getElementById("micBtn");
const voiceBtn = document.getElementById("voiceBtn");
const resetBtn = document.getElementById("resetBtn");
const statusText = document.getElementById("statusText");
const orb = document.getElementById("orb");
const connection = document.getElementById("connection");

let busy = false;
let voiceEnabled = true;
let currentAudio = null;

function setStatus(text, active = false) {
    statusText.textContent = text;
    orb.classList.toggle("active", active);
}

function addMessage(who, text) {
    const box = document.createElement("div");
    box.className = "message " + (who === "YOU" ? "user" : "bot");

    const label = document.createElement("div");
    label.className = "who";
    label.textContent = who;

    const body = document.createElement("div");
    body.textContent = text;

    box.appendChild(label);
    box.appendChild(body);
    chat.appendChild(box);
    chat.scrollTop = chat.scrollHeight;
    return box;
}

function setBusy(value) {
    busy = value;
    sendBtn.disabled = value;
    input.disabled = value;
    connection.textContent = value ? "THINKING" : "ONLINE";
}

async function speakReply(text) {
    if (!voiceEnabled) return;

    try {
        setStatus("ULTRON is speaking...", true);

        const response = await fetch("/api/speak", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({text})
        });

        if (!response.ok) throw new Error("Speech generation failed");

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        currentAudio = new Audio(url);

        await new Promise((resolve, reject) => {
            currentAudio.onended = resolve;
            currentAudio.onerror = reject;
            currentAudio.play().catch(reject);
        });

        URL.revokeObjectURL(url);
    } catch (error) {
        console.error(error);
        setStatus("Text reply ready. Audio could not play.");
        return;
    }

    setStatus("Ready, Boss.");
}

async function sendMessage(text) {
    text = text.trim();
    if (!text || busy) return;

    addMessage("YOU", text);
    input.value = "";
    setBusy(true);
    setStatus("Processing your request...", true);

    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({message: text})
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Request failed");
        }

        addMessage("ULTRON", data.reply);

        if (voiceEnabled) {
            await speakReply(data.reply);
        } else {
            setStatus("Ready, Boss.");
        }
    } catch (error) {
        addMessage("SYSTEM", error.message || "Connection error.");
        setStatus("Request failed.");
    } finally {
        setBusy(false);
        input.focus();
    }
}

form.addEventListener("submit", event => {
    event.preventDefault();
    sendMessage(input.value);
});

voiceBtn.addEventListener("click", () => {
    voiceEnabled = !voiceEnabled;
    voiceBtn.textContent = voiceEnabled ? "🔊 VOICE: ON" : "🔇 VOICE: OFF";

    if (!voiceEnabled && currentAudio) {
        currentAudio.pause();
        currentAudio = null;
        setStatus("Voice disabled.");
    }
});

resetBtn.addEventListener("click", async () => {
    if (busy) return;

    try {
        const response = await fetch("/api/reset", {method: "POST"});
        if (!response.ok) throw new Error("Reset failed");

        chat.replaceChildren();
        addMessage("ULTRON", "Memory cleared for this conversation. Ready, Boss.");
        setStatus("Chat reset.");
    } catch (error) {
        setStatus("Could not reset chat.");
    }
});

// Browser speech recognition: Chrome/Edge support varies by device.
const SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition;

if (SpeechRecognition) {
    const recognition = new SpeechRecognition();
    recognition.lang = "en-IN";
    recognition.interimResults = false;
    recognition.continuous = false;

    recognition.onstart = () => {
        micBtn.classList.add("recording");
        micBtn.textContent = "LISTENING...";
        setStatus("Listening. Speak now.", true);
    };

    recognition.onresult = event => {
        const spokenText = event.results[0][0].transcript;
        input.value = spokenText;
        sendMessage(spokenText);
    };

    recognition.onerror = event => {
        setStatus("Microphone error: " + event.error);
    };

    recognition.onend = () => {
        micBtn.classList.remove("recording");
        micBtn.textContent = "🎙 SPEAK";
        if (!busy) setStatus("Ready, Boss.");
    };

    micBtn.addEventListener("click", () => {
        if (busy) return;
        try {
            recognition.lang = /[\u0900-\u097F]/.test(input.value)
                ? "hi-IN" : "en-IN";
            recognition.start();
        } catch (error) {
            setStatus("Microphone is already active.");
        }
    });
} else {
    micBtn.disabled = true;
    micBtn.textContent = "MIC NOT SUPPORTED";
    micBtn.title = "Try an up-to-date Chrome or Edge browser.";
}

fetch("/api/status")
    .then(response => response.json())
    .then(data => {
        if (!data.api_configured) {
            connection.textContent = "API KEY MISSING";
            setStatus("Add OPENROUTER_API_KEY to your .env file.");
        }
    })
    .catch(() => {
        connection.textContent = "OFFLINE";
    });
</script>
</body>
</html>
"""


# ============== CONVERSATION MEMORY ==============

conversation = [
    {
        "role": "system",
        "content": (
            "You are ULTRON, a personal AI voice assistant. "
            "Address the user as Boss when appropriate. "
            "Understand Hindi, Hinglish and English. "
            "Reply in the user's language. "
            "Give the final answer directly, not internal reasoning. "
            "Keep spoken answers concise unless asked for details. "
            "Never claim an action was completed unless it was."
        )
    }
]


# ============== LANGUAGE DETECTION ==============

def detect_language(text):
    if re.search(r"[\u0900-\u097F]", text):
        return "hindi"

    hinglish_words = {
        "hai", "hain", "ho", "kya", "kaise", "kyun",
        "batao", "bata", "karo", "nahi", "nahin",
        "mujhe", "tum", "aap", "mera", "meri",
        "chahiye", "bhai", "aur", "mein", "kaun",
        "karna", "kardo", "bataiye", "accha", "acha"
    }

    words = set(re.findall(r"[a-zA-Z]+", text.lower()))

    if len(words.intersection(hinglish_words)) >= 2:
        return "hindi"

    return "english"


def clean_text(text):
    text = re.sub(r"```.*?```", " code omitted ", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[*_#>]", "", text)
    return re.sub(r"\s+", " ", text).strip()


# ============== TEXT TO SPEECH ==============

async def generate_speech(text, voice, filename):
    communicator = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=VOICE_RATE,
        pitch=VOICE_PITCH
    )
    await communicator.save(filename)


async def generate_speech_bytes(text, voice):
    """Generate MP3 bytes for the web browser."""
    communicator = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=VOICE_RATE,
        pitch=VOICE_PITCH
    )

    audio_parts = []

    async for chunk in communicator.stream():
        if chunk.get("type") == "audio":
            audio_parts.append(chunk["data"])

    return b"".join(audio_parts)


def speak(text, language=None):
    text = clean_text(str(text))

    if not text:
        return

    if language is None:
        language = detect_language(text)

    voice = HINDI_VOICE if language == "hindi" else ENGLISH_VOICE
    filename = None

    with speech_lock:
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".mp3", delete=False
            ) as temp:
                filename = temp.name

            asyncio.run(generate_speech(text, voice, filename))

            if not pygame.mixer.get_init():
                pygame.mixer.init()

            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                time.sleep(0.05)

            pygame.mixer.music.unload()

        except Exception as error:
            print("TTS error:", error)

        finally:
            try:
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
            except Exception:
                pass

            if filename and os.path.exists(filename):
                try:
                    os.remove(filename)
                except OSError:
                    pass


# ============== SPEECH RECOGNITION ==============

recognizer = sr.Recognizer()
microphone = None


def setup_microphone():
    global microphone

    recognizer.pause_threshold = SILENCE_SECONDS
    recognizer.non_speaking_duration = 0.3
    recognizer.phrase_threshold = 0.2
    recognizer.dynamic_energy_threshold = True

    microphone = sr.Microphone()

    with microphone as source:
        print("Calibrating microphone once...")
        recognizer.adjust_for_ambient_noise(
            source,
            duration=NOISE_CALIBRATION
        )

    print("Microphone ready.")


def listen():
    try:
        print("\nListening... Speak now.")

        with microphone as source:
            audio = recognizer.listen(
                source,
                timeout=START_TIMEOUT,
                phrase_time_limit=PHRASE_LIMIT
            )

        print("Recognizing speech...")

        try:
            text = recognizer.recognize_google(
                audio, language="en-IN"
            )
        except sr.UnknownValueError:
            text = recognizer.recognize_google(
                audio, language="hi-IN"
            )

        text = text.strip()

        if text:
            print("You:", text)

        return text

    except sr.WaitTimeoutError:
        return ""

    except sr.UnknownValueError:
        print("Speech samajh nahi aayi.")
        return ""

    except sr.RequestError as error:
        print("Speech service/network error:", error)
        time.sleep(1)
        return ""

    except Exception as error:
        print("Microphone error:", error)
        time.sleep(0.5)
        return ""


# ============== OPENROUTER AI ==============

def ask_ai(user_text):
    if not API_KEY:
        return "Boss, OpenRouter API key missing hai."

    conversation.append({
        "role": "user",
        "content": user_text
    })

    models = list(dict.fromkeys([MODEL] + FALLBACK_MODELS))

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "models": models,
        "messages": conversation,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.4,
        "reasoning": {"effort": "low"}
    }

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=45
        )

        try:
            data = response.json()
        except ValueError:
            data = {}

        if response.ok and data.get("choices"):
            choice = data["choices"][0]
            message = choice.get("message", {})
            answer = message.get("content")

            if isinstance(answer, list):
                answer = " ".join(
                    part.get("text", "")
                    for part in answer
                    if isinstance(part, dict)
                )

            if answer and str(answer).strip():
                answer = str(answer).strip()

                conversation.append({
                    "role": "assistant",
                    "content": answer
                })

                if len(conversation) > 21:
                    conversation[:] = (
                        [conversation[0]] + conversation[-20:]
                    )

                print("Successful model:", data.get("model"))
                return answer

            print(
                "Model returned empty content.",
                "Finish reason:", choice.get("finish_reason")
            )

        else:
            print(
                "OpenRouter error:",
                response.status_code,
                data.get("error", data)
            )

    except requests.Timeout:
        print("AI request timed out.")

    except requests.RequestException as error:
        print("AI network error:", error)

    if conversation and conversation[-1]["role"] == "user":
        conversation.pop()

    return (
        "Boss, AI service abhi respond nahi kar rahi. "
        "Please dobara try karo."
    )


# ============== WEB API ROUTES ==============

@app.route("/")
def home():
    return render_template_string(PAGE)


@app.route("/api/status")
def web_status():
    return jsonify({
        "online": True,
        "api_configured": bool(API_KEY),
        "model": MODEL
    })


@app.route("/api/chat", methods=["POST"])
def web_chat():
    data = request.get_json(silent=True) or {}
    user_text = data.get("message", "")

    if not isinstance(user_text, str):
        return jsonify({"error": "Message must be text."}), 400

    user_text = user_text.strip()

    if not user_text:
        return jsonify({"error": "Message empty hai."}), 400

    if len(user_text) > 10000:
        return jsonify({"error": "Message too long hai."}), 413

    # Keep shared conversation memory safe between simultaneous requests.
    with ai_lock:
        answer = ask_ai(user_text)

    return jsonify({"success": True, "reply": answer})


@app.route("/api/speak", methods=["POST"])
def web_speak():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Speech text empty hai."}), 400

    if len(text) > 5000:
        return jsonify({"error": "Speech text too long hai."}), 413

    language = detect_language(text)
    voice = HINDI_VOICE if language == "hindi" else ENGLISH_VOICE

    try:
        audio_bytes = asyncio.run(
            generate_speech_bytes(clean_text(text), voice)
        )

        if not audio_bytes:
            return jsonify({"error": "Audio generate nahi hua."}), 500

        return send_file(
            io.BytesIO(audio_bytes),
            mimetype="audio/mpeg",
            download_name="ultron_reply.mp3"
        )

    except Exception:
        app.logger.exception("Edge TTS failed")
        return jsonify({"error": "Voice generation failed."}), 500


@app.route("/api/reset", methods=["POST"])
def web_reset():
    with ai_lock:
        conversation[:] = [conversation[0]]

    return jsonify({"success": True})


# ============== EXIT COMMANDS ==============

def should_exit(text):
    command = re.sub(r"[^\w\s]", "", text.lower()).strip()

    exit_commands = {
        "exit",
        "quit",
        "shutdown",
        "shut down",
        "stop ultron",
        "ultron stop",
        "ultron shutdown",
        "ultron band karo",
        "ultron ko band karo",
        "band ho jao",
        "program band karo"
    }

    return command in exit_commands


# ============== ORIGINAL VOICE MODE ==============

def main():
    if not API_KEY:
        print(
            "API key missing hai. Ultron.py ke same folder mein "
            ".env file check karo."
        )
        return

    try:
        pygame.mixer.init()
        setup_microphone()

        print("=" * 42)
        print("           ULTRON AI")
        print("          VOICE MODE")
        print("=" * 42)

        speak("Online, Boss. How can I help you?", "english")

        while True:
            user_text = listen()

            if not user_text:
                continue

            if should_exit(user_text):
                speak("Shutting down. Goodbye, Boss.", "english")
                break

            with ai_lock:
                answer = ask_ai(user_text)

            print("\nULTRON:", answer, "\n")
            speak(answer)

    except KeyboardInterrupt:
        print("\nULTRON stopped.")

    except Exception as error:
        print("Fatal error:", error)

    finally:
        try:
            pygame.mixer.quit()
        except Exception:
            pass


# ============== START APP ==============

if __name__ == "__main__":
    if "--voice" in sys.argv:
        main()
    else:
        print("=" * 42)
        print("       ULTRON WEB APP")
        print("=" * 42)
        print("Open on this PC: http://127.0.0.1:5000")
        print("Stop server: CTRL+C")

        app.run(
            host="0.0.0.0",
            port=int(os.getenv("PORT", "5000")),
            debug=False,
            threaded=True
        )
