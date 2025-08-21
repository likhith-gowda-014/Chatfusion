from flask import Flask, render_template, request, send_file, jsonify, redirect, url_for
import os
import uuid
import sqlite3
from faster_whisper import WhisperModel
from TTS.api import TTS
import requests
from pydub import AudioSegment
import json
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESPONSE_FOLDER'] = 'responses'
app.config['VOICE_MODEL_FOLDER'] = 'voice_models'
app.config['DB_PATH'] = 'database/voice_chatbot.db'

OPENROUTER_API_KEY = "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"

# Initialize STT
whisper_model = WhisperModel("tiny.en", compute_type="int8")

# ✅ Switch to XTTSv2 for real speaker cloning
tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=False)

# Ensure directories exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['RESPONSE_FOLDER'], app.config['VOICE_MODEL_FOLDER'], 'database']:
    os.makedirs(folder, exist_ok=True)

def init_db():
    with sqlite3.connect(app.config['DB_PATH']) as conn:
        cursor = conn.cursor()
        cursor.execute(''' 
            CREATE TABLE IF NOT EXISTS voice_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                path TEXT
            )
        ''')
        conn.commit()

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train-model')
def train_model_page():
    return render_template('train_model.html')

@app.route('/chat')
def chat_page():
    with sqlite3.connect(app.config['DB_PATH']) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM voice_models")
        voices = [row[0] for row in cursor.fetchall()]
    return render_template('chat.html', voices=voices)

@app.route('/train-model', methods=['POST'])
def train_voice_model():
    model_name = request.form.get('model_name')
    files = request.files.getlist('voice_samples')

    model_dir = os.path.join(app.config['VOICE_MODEL_FOLDER'], model_name)
    os.makedirs(model_dir, exist_ok=True)

    print(f"Training new model: {model_name}")
    print(f"Saving to directory: {model_dir}")

    for i, file in enumerate(files):
        original_filename = file.filename
        print(f"Processing file {original_filename}")
        try:
            audio = AudioSegment.from_file(file)
            converted_filename = f"{i}_{os.path.splitext(original_filename)[0]}.wav"
            save_path = os.path.join(model_dir, converted_filename)
            audio.export(save_path, format="wav")
            print(f"Saved converted WAV to: {save_path}")
        except Exception as e:
            print(f"Error converting {original_filename}: {e}")

    with sqlite3.connect(app.config['DB_PATH']) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO voice_models (name, path) VALUES (?, ?)", (model_name, model_dir))
        conn.commit()
        print(f"Model {model_name} registered in DB.")
    
    return redirect(url_for('chat_page'))

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    file = request.files['audio']
    # Load latest emotion from emotions.json
    emotion_file = os.path.join("data", "emotions.json")
    emotion = "neutral"

    try:
        with open(emotion_file, "r") as f:
            emotion_data = json.load(f)
            if emotion_data:
                # Parse timestamps and sort to get the most recent
                latest = sorted(emotion_data, key=lambda x: datetime.fromisoformat(x['timestamp']))[-1]
                emotion = latest["emotion"]
    except Exception as e:
        print(f"Could not load emotion from {emotion_file}: {e}")
    voice_id = request.form.get('voice_id', 'default')
    prompt_template = request.form.get('prompt-template', 'assistant')  # Capture prompt template

    print(f"Received audio for voice_id: {voice_id} with emotion: {emotion} and template: {prompt_template}")

    filename = str(uuid.uuid4()) + ".wav"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    print(f"Saved uploaded file to: {filepath}")

    if not filepath.endswith(".wav"):
        print(f"Converting {filepath} to .wav")
        sound = AudioSegment.from_file(filepath)
        filepath = filepath.replace(".webm", ".wav")
        sound.export(filepath, format="wav")

    print("Transcribing...")
    segments, _ = whisper_model.transcribe(filepath)
    text = "".join([seg.text for seg in segments])
    print(f"Transcribed text: {text}")

    ai_response_text = get_ai_response(text, prompt_template, emotion)  # Pass template to AI response
    response_text = ai_response_text
    print(f"AI Response: {response_text}")

    with sqlite3.connect(app.config['DB_PATH']) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT path FROM voice_models WHERE name = ?", (voice_id,))
        result = cursor.fetchone()

    if not result:
        print(f"Voice model '{voice_id}' not found.")
        return jsonify({"error": "Voice model not found"}), 400

    voice_model_path = result[0]
    print(f"Using voice model from path: {voice_model_path}")

    wav_files = [f for f in os.listdir(voice_model_path) if f.endswith('.wav')]
    print(f"Found {len(wav_files)} training samples.")

    if not wav_files:
        return jsonify({"error": "No .wav file found in the voice model folder"}), 400

    voice_model_wav = os.path.join(voice_model_path, wav_files[0])
    output_path = os.path.join(app.config['RESPONSE_FOLDER'], f"{uuid.uuid4()}.wav")

    try:
        print(f"Generating TTS with reference: {voice_model_wav}")
        tts_model.tts_to_file(
            text=response_text,
            speaker_wav=voice_model_wav,
            language="en",
            file_path=output_path
        )
        print(f"TTS saved at: {output_path}")
        return jsonify({"response_audio": output_path.replace('\\', '/')})
    except Exception as e:
        print(f"TTS Error: {e}")
        return jsonify({"error": "TTS generation failed", "details": str(e)}), 500

@app.route('/get-audio/<filename>')
def get_audio(filename):
    filepath = os.path.join(app.config['RESPONSE_FOLDER'], filename)
    return send_file(filepath, mimetype="audio/wav")

def get_ai_response(user_input, prompt_template, emotion):
    # Emotionally-aware and role-specific templates
    template_prompts = {
        "assistant": (
            "You are a compassionate assistant who supports users with empathy. "
            "Keep responses gentle, optimistic, and emotionally encouraging. "
            "Be brief, positive, and helpful. Respond to: "
        ),
        "friend": (
            "You are the user's supportive best friend. Be kind, encouraging, and fun. "
            "Use a warm and casual tone. Help cheer them up or talk to them like a close buddy. Respond to: "
        ),
        "tutor": (
            "You are a patient and understanding tutor who explains clearly and calmly. "
            "If the user seems down, motivate them. Otherwise, teach them simply. Respond to: "
        )
    }

    # Emotional modifiers
    emotion_modifiers = {
        "happy": "The user is feeling happy. Match their energy, celebrate with them, and keep your tone cheerful and light.",
        "sad": "The user is feeling sad. Respond with comfort, kindness, and emotional encouragement.",
        "angry": "The user is angry. Be calm, understanding, and help them relax or redirect their focus.",
        "neutral": "The user is feeling neutral. Engage in a balanced and steady way.",
        "fear": "The user is anxious or fearful. Reassure them gently and build confidence.",
        "surprise": "The user is surprised. Be curious, engaging, and match the excitement."
    }

    role_prompt = template_prompts.get(prompt_template, template_prompts["assistant"])
    emotion_prompt = emotion_modifiers.get(emotion, emotion_modifiers["neutral"])

    # Fallback to assistant if invalid
    full_prompt = f"{role_prompt}\n{emotion_prompt}\nConsidering how the user is feeling, craft a supportive and relevant response to their message:"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "meta-llama/llama-3-8b-instruct",
        "messages": [
            {
                "role": "system",
                "content": full_prompt
            },
            {
                "role": "user",
                "content": user_input
            }
        ]
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        ai_message = response.json()["choices"][0]["message"]["content"].strip()
        print(f"OpenRouter AI response: {ai_message}")
        return ai_message
    except Exception as e:
        print(f"OpenRouter Error: {e}")
        return "I'm really sorry, I'm having trouble right now. But I'm here for you!"

if __name__ == '__main__':
    app.run(debug=True)
