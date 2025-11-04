# -*- coding: utf-8 -*-
import os, io, json, wave, uuid
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, send_from_directory
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
import requests

from db.models import Base, Conversation, Message
from rag.retriever import Retriever

# --- env / config ---
load_dotenv()
ROOT = Path(__file__).resolve().parent
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///chat.db")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")
VOSK_MODEL_DIR = os.getenv("VOSK_MODEL_DIR", "")
TOP_K = int(os.getenv("TOP_K", "5"))

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev")
app.config["UPLOAD_AUDIO_DIR"] = str(ROOT / "static" / "audio")
Path(app.config["UPLOAD_AUDIO_DIR"]).mkdir(parents=True, exist_ok=True)

# --- DB ---
engine = create_engine(DATABASE_URL, echo=False, future=True)
Base.metadata.create_all(engine)

# --- RAG ---
retriever = None
try:
    retriever = Retriever()
except Exception as e:
    print("⚠️ RAG not ready:", e)

# --- Optional local STT (Vosk) ---
vosk_model = None
if VOSK_MODEL_DIR and Path(VOSK_MODEL_DIR).exists():
    try:
        from vosk import Model, KaldiRecognizer
        vosk_model = Model(VOSK_MODEL_DIR)
        print("✅ Vosk model loaded.")
    except Exception as e:
        print("⚠️ Vosk not usable:", e)

def ollama_chat(prompt: str, history: list[dict]) -> str:
    """
    Calls Ollama /api/chat with gemma3:1b.
    history: [{"role":"user"/"assistant","content":"..."}]
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": m["role"], "content": m["content"]} for m in history] + [{"role":"user","content":prompt}],
        "options": {"temperature": 0.4},
        "stream": False
    }
    r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data.get("message", {}).get("content", "")

def build_system_prompt() -> str:
    return (
        "You are a helpful assistant. Answer using the provided context if relevant; "
        "if not found, say you don't have enough information. Keep answers concise.\n"
    )

def rag_prompt(user_text: str, retrieved: list[tuple[str,dict,float]]) -> str:
    ctx = "\n\n".join([f"[{i+1}] (score {round(s,3)} src:{m['source']})\n{c}" 
                       for i, (c, m, s) in enumerate(retrieved)])
    return f"{build_system_prompt()}\nContext:\n{ctx}\n\nUser:\n{user_text}"

@app.get("/")
def index():
    # auto-create a conversation for first load
    with Session(engine) as s:
        conv = Conversation(title="Session")
        s.add(conv); s.commit()
        conv_id = conv.id
    return render_template("index.html", conversation_id=conv_id)

@app.get("/api/conversations")
def list_conversations():
    with Session(engine) as s:
        rows = s.scalars(select(Conversation).order_by(Conversation.created_at.desc())).all()
        data = [{"id": c.id, "title": c.title, "created_at": c.created_at.isoformat()} for c in rows]
    return jsonify(data)

@app.get("/api/history/<int:conversation_id>")
def get_history(conversation_id: int):
    with Session(engine) as s:
        conv = s.get(Conversation, conversation_id)
        if not conv:
            return jsonify({"error": "Not found"}), 404
        msgs = [{"id": m.id, "role": m.role, "content": m.content, "audio_path": m.audio_path,
                 "created_at": m.created_at.isoformat()}
                for m in conv.messages]
    return jsonify(msgs)

@app.post("/api/chat")
def chat():
    """
    JSON: { conversation_id, text?, use_rag:bool=true }
    """
    data = request.get_json(force=True)
    conversation_id = int(data.get("conversation_id"))
    user_text = (data.get("text") or "").strip()
    use_rag = bool(data.get("use_rag", True))

    if not user_text:
        return jsonify({"error": "Empty input"}), 400

    # save user message
    with Session(engine) as s:
        conv = s.get(Conversation, conversation_id)
        if not conv: return jsonify({"error":"Conversation not found"}), 404
        s.add(Message(conversation_id=conversation_id, role="user", content=user_text))
        s.commit()

        # history for model
        history = [{"role": m.role, "content": m.content} for m in conv.messages]

    # RAG
    final_prompt = user_text
    if use_rag and retriever:
        R = retriever.search(user_text, k=TOP_K)
        final_prompt = rag_prompt(user_text, R)

    # LLM
    assistant_text = ollama_chat(final_prompt, history=[])

    # save assistant message
    with Session(engine) as s:
        s.add(Message(conversation_id=conversation_id, role="assistant", content=assistant_text))
        s.commit()

    return jsonify({"reply": assistant_text})

@app.post("/api/transcribe")
def transcribe():
    """
    Receives audio (wav/ogg/webm), saves file, transcribes via Vosk if available.
    Returns: { text, audio_url }
    """
    if "audio" not in request.files:
        return jsonify({"error":"audio file required"}), 400

    f = request.files["audio"]
    ext = Path(f.filename).suffix or ".wav"
    fname = f"{uuid.uuid4().hex}{ext}"
    out_path = Path(app.config["UPLOAD_AUDIO_DIR"]) / fname
    f.save(out_path)

    text = ""
    if vosk_model:
        try:
            from vosk import KaldiRecognizer
            import soundfile as sf
            data, sr = sf.read(out_path, dtype="int16")
            rec = KaldiRecognizer(vosk_model, sr)
            rec.SetWords(True)
            # stream in chunks
            chunk = 4000
            for i in range(0, len(data), chunk):
                rec.AcceptWaveform(data[i:i+chunk].tobytes())
            res = json.loads(rec.FinalResult())
            text = (res.get("text") or "").strip()
        except Exception as e:
            text = ""
            print("STT error:", e)

    return jsonify({
        "text": text,
        "audio_url": f"/static/audio/{fname}"
    })

@app.post("/api/say")
def say_and_respond():
    """
    Combo endpoint: receive audio, transcribe, show in chat, RAG answer.
    Form-Data: audio (file), conversation_id (int), use_rag (bool)
    """
    conversation_id = int(request.form.get("conversation_id", "0"))
    use_rag = request.form.get("use_rag", "true").lower() != "false"

    # reuse /api/transcribe
    if "audio" not in request.files:
        return jsonify({"error":"audio file required"}), 400
    f = request.files["audio"]
    ext = Path(f.filename).suffix or ".wav"
    fname = f"{uuid.uuid4().hex}{ext}"
    out_path = Path(app.config["UPLOAD_AUDIO_DIR"]) / fname
    f.save(out_path)

    # transcribe
    text = ""
    if vosk_model:
        try:
            from vosk import KaldiRecognizer
            import soundfile as sf
            data, sr = sf.read(out_path, dtype="int16")
            rec = KaldiRecognizer(vosk_model, sr)
            rec.SetWords(True)
            chunk = 4000
            for i in range(0, len(data), chunk):
                rec.AcceptWaveform(data[i:i+chunk].tobytes())
            res = json.loads(rec.FinalResult())
            text = (res.get("text") or "").strip()
        except Exception as e:
            print("STT error:", e)

    # Save user message with audio link
    with Session(engine) as s:
        conv = s.get(Conversation, conversation_id)
        if not conv: return jsonify({"error":"Conversation not found"}), 404
        s.add(Message(conversation_id=conversation_id, role="user", content=text or "(no speech)", audio_path=f"/static/audio/{fname}"))
        s.commit()

    # RAG + LLM
    final_prompt = text
    if use_rag and retriever and text:
        R = retriever.search(text, k=TOP_K)
        final_prompt = rag_prompt(text, R)
    assistant_text = ollama_chat(final_prompt or "User sent empty audio.", history=[])

    with Session(engine) as s:
        s.add(Message(conversation_id=conversation_id, role="assistant", content=assistant_text))
        s.commit()

    return jsonify({"transcript": text, "reply": assistant_text, "audio_url": f"/static/audio/{fname}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=bool(int(os.getenv("FLASK_DEBUG","1"))))
