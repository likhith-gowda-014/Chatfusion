import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file, Response
import time
import threading
import bcrypt
import json
from datetime import datetime
import tempfile
from gtts import gTTS
import io
import sqlite3
from flask_session import Session
import requests
import re
import chromadb
from chromadb.utils import embedding_functions
import uuid
from pydub import AudioSegment

# Add these new imports for the cosine similarity calculation
from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key")
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Initialize VoiceEncoder for cosine similarity
voice_encoder = VoiceEncoder()

# SQLite Database Connection
def get_db_connection():
    try:
        conn = sqlite3.connect("database.db", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as err:
        print(f"Error connecting to database: {err}")
        return None

db = get_db_connection()

# PERSONA DB setup
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESPONSE_FOLDER'] = 'responses'
app.config['VOICE_MODEL_FOLDER'] = 'voice_models'
app.config['PERSONA_DB_PATH'] = 'database/voice_chatbot.db'

os.makedirs('database', exist_ok=True)

def init_persona_db():
    with sqlite3.connect(app.config['PERSONA_DB_PATH']) as conn:
        cursor = conn.cursor()
        cursor.execute(''' 
            CREATE TABLE IF NOT EXISTS voice_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                path TEXT
            )
        ''')
        conn.commit()

init_persona_db()

# ✅ FIX: LAZY LOAD ALL HEAVY MODELS
# A dictionary to cache loaded models
MODEL_CACHE = {
    "whisper": None,
    "tts": None,
    "deepface": None,
    "chroma_client": None,
    "chroma_collection": None
}

def get_whisper_model():
    if MODEL_CACHE["whisper"] is None:
        print("Loading Whisper model for the first time...")
        try:
            from faster_whisper import WhisperModel
            MODEL_CACHE["whisper"] = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        except Exception as e:
            print(f"Failed to load Whisper model: {e}")
            return None
    return MODEL_CACHE["whisper"]

def get_tts_model():
    if MODEL_CACHE["tts"] is None:
        print("Loading TTS model (XTTSv2) for the first time...")
        try:
            from TTS.api import TTS
            MODEL_CACHE["tts"] = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=False)
        except Exception as e:
            print(f"Failed to load XTTS model: {e}")
            return None
    return MODEL_CACHE["tts"]

def get_deepface_model():
    if MODEL_CACHE["deepface"] is None:
        print("Loading DeepFace model for the first time...")
        try:
            from deepface import DeepFace
            MODEL_CACHE["deepface"] = DeepFace
        except Exception as e:
            print(f"Failed to load DeepFace model: {e}")
            return None
    return MODEL_CACHE["deepface"]

# ✅ FIX: Lazy Load ChromaDB
def get_chroma_collection():
    if MODEL_CACHE["chroma_collection"] is None:
        print("Loading ChromaDB client and collection for the first time...")
        db_path = "./chroma_db_data"
        try:
            # We import here to prevent it from being loaded at the application startup
            import chromadb
            from chromadb.utils import embedding_functions
            client = chromadb.PersistentClient(path=db_path)
            sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            MODEL_CACHE["chroma_collection"] = client.get_or_create_collection(
                name="user_chat_history",
                embedding_function=sentence_transformer_ef
            )
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            return None
    return MODEL_CACHE["chroma_collection"]

# Initialize DB schema if not exists
with db:
    db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            userid TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        );
    """)

# Emotion tracking file path
MEMORY_FILE = os.path.join("data/emotions.json")

# Ensure the 'data' directory exists
if not os.path.exists(os.path.dirname(MEMORY_FILE)):
    os.makedirs(os.path.dirname(MEMORY_FILE))

# Function to load latest emotion
def load_latest_emotion():
    if not os.path.exists(MEMORY_FILE):
        print("Emotion file not found, creating a new one.")
        with open(MEMORY_FILE, "w") as file:
            json.dump([], file)
        return "neutral"
    try:
        with open(MEMORY_FILE, 'r') as file:
            emotions = json.load(file)
            if isinstance(emotions, list) and emotions:
                return emotions[-1].get("emotion","neutral")
    except json.JSONDecodeError:
        print("Error: Corrupt emotions.json file. Attempting recovery.")
        with open(MEMORY_FILE, "w") as file:
            json.dump([], file)
        return "neutral"
    except Exception as e:
        print(f"Error loading emotion data: {e}")
        return "neutral"

#Stores emotion
def store_emotion(emotion):
    try:
        if not os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, 'w') as file:
                json.dump([], file)
        with open(MEMORY_FILE, 'r') as file:
            try:
                emotions = json.load(file)
                if not isinstance(emotions, list):
                    emotions = []
            except json.JSONDecodeError:
                emotions = []
        new_entry = {"timestamp": str(datetime.now()), "emotion": emotion}
        emotions.append(new_entry)
        emotions = emotions[-7:]
        with open(MEMORY_FILE, 'w') as file:
            json.dump(emotions, file, indent=4)
    except Exception as e:
        print(f"Error storing emotion: {e}")

# ✅ FIX: Conditional Emotion Capture
ENABLE_LIVE_EMOTION = os.getenv("RENDER_EXTERNAL_URL") is None and os.getenv("IS_LOCAL", "True") == "True"

# Function to capture emotion continuously
def capture_emotion():
    if not ENABLE_LIVE_EMOTION:
        print("Live emotion capture is disabled in this environment (likely cloud deployment).")
        return
        
    # Get DeepFace model via lazy-loader
    deepface_model = get_deepface_model()
    if deepface_model is None:
        print("DeepFace model failed to load. Cannot run emotion detection.")
        return
        
    import cv2
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not accessible. Disabling live emotion thread.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        try:
            analysis = deepface_model.analyze(frame, actions=['emotion'], enforce_detection=False)
            if analysis and isinstance(analysis, list) and 'dominant_emotion' in analysis[0]:
                dominant_emotion = analysis[0]['dominant_emotion']
                print(f"Detected Emotion: {dominant_emotion}")
                store_emotion(dominant_emotion)
        except Exception as e:
            print(f"Emotion detection error: {e}")
        time.sleep(3)

    cap.release()
    cv2.destroyAllWindows()

# Start the thread ONLY if running locally
if ENABLE_LIVE_EMOTION:
    print("Starting live emotion capture thread...")
    threading.Thread(target=capture_emotion, daemon=True).start()
else:
    print("Running in headless environment. Live emotion feature will use existing file data.")

# Home Route
@app.route('/')
def home():
    return render_template('home.html')

# Sign Up Route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        userid = request.form['userid']
        email = request.form['email']
        password = request.form['password']

        try:
            cursor = db.cursor()
            cursor.execute("SELECT * FROM users WHERE userid = ? OR email = ?", (userid, email))
            existing_user = cursor.fetchone()

            if existing_user:
                if existing_user['userid'] == userid and existing_user['email'] == email:
                    error = "Both User ID and Email are already taken. Please use different ones."
                elif existing_user['userid'] == userid:
                    error = "User ID is already taken. Please choose a different one."
                elif existing_user['email'] == email:
                    error = "Email is already taken. Please use a different email."
                return render_template('signup.html', error=error)

            cursor.execute("INSERT INTO users (name, userid, email, password) VALUES (?, ?, ?, ?)",
                            (name, userid, email, password))
            db.commit()
            return redirect(url_for('signin'))

        except sqlite3.Error as e:
            error = f"Database Error: {str(e)}"
            return render_template('signup.html', error=error)

    return render_template('signup.html')

# Sign In Route
@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        userid = request.form['userid']
        password = request.form['password']

        try:
            cursor = db.cursor()
            cursor.execute("SELECT * FROM users WHERE userid = ? AND password = ?", (userid, password))
            user = cursor.fetchone()

            if user:
                session['userid'] = user['userid']
                session['name'] = user['name']
                return redirect(url_for('dashboard'))
            else:
                error = "Invalid User ID or Password. Please try again."
                return render_template('signin.html', error=error)

        except sqlite3.Error as e:
            error = f"Database Error: {str(e)}"
            return render_template('signin.html', error=error)

    return render_template('signin.html')

# Dashboard Route
@app.route('/dashboard')
def dashboard():
    if 'userid' in session:
        return render_template('dashboard.html', name=session['name'])
    else:
        return redirect(url_for('signin'))

# Logout Route
@app.route('/logout')
def logout():
    session.pop('userid', None)
    session.pop('name', None)
    return redirect(url_for('home'))

def analyze_emotion_trend():
    try:
        with open(MEMORY_FILE, 'r') as file:
            emotions = json.load(file)
            return emotions,[]
    except Exception as e:
        print(f"Error reading emotion file: {e}")
        return "neutral",[]
    
# Use a persistent client to save data to a directory
# This will store your vector database in a folder named 'chroma_db_data'.
def store_chat_interaction(user_id: str, user_message: str, bot_response: str):
    user_chat_collection = get_chroma_collection()
    if user_chat_collection is None:
        print("ChromaDB is not initialized. Cannot store chat history.")
        return
    import time
    timestamp = str(int(time.time() * 1000))
    try:
        user_chat_collection.add(
            documents=[user_message, bot_response],
            metadatas=[
                {"role": "user", "user_id": user_id, "timestamp": timestamp},
                {"role": "assistant", "user_id": user_id, "timestamp": timestamp}
            ],
            ids=[f"user_{user_id}_{timestamp}", f"bot_{user_id}_{timestamp}"]
        )
        print(f"Stored chat interaction for user {user_id}")
    except Exception as e:
        print(f"Error storing chat interaction in ChromaDB: {e}")

def get_relevant_context(user_id: str, query: str, n_results: int = 5):
    user_chat_collection = get_chroma_collection()
    if user_chat_collection is None:
        return ""
    try:
        print("\n" + "="*50)
        print(f"\n[DEBUG] Searching for context with query: '{query}'")
        results = user_chat_collection.query(
            query_texts=[query],
            n_results=n_results,
            where={"user_id": user_id}
        )
        print(f"[DEBUG] Raw ChromaDB results: {json.dumps(results, indent=2)}")
        print("\n" + "="*50)
        context_messages = []
        if 'documents' in results and results['documents']:
            for document, metadata in zip(results['documents'][0], results['metadatas'][0]):
                role = metadata['role']
                context_messages.append(f"{role.capitalize()}: {document}")
        print("\n" + "="*50)
        print(f"[DEBUG] Extracted context messages:\n{context_messages}")
        print("\n" + "="*50)
        return "\n".join(context_messages)
    except Exception as e:
        print(f"[DEBUG] Error retrieving context from ChromaDB: {e}")
        return ""

def chat_with_llama3(user_input):
    api_key = os.getenv("OPENROUTER_API_KEY")
    endpoint = "https://openrouter.ai/api/v1/chat/completions"
    user_id = session.get('userid')
    if not user_id:
        return "Please log in to use the chatbot with personalized history."
    try:
        dominant_trend, recent_emotions = analyze_emotion_trend()
    except Exception as e:
        print(f"[DEBUG] Emotion analysis failed: {e}")
        dominant_trend, recent_emotions = "neutral", []
    context = get_relevant_context(user_id, user_input)
    emotional_context = (
        f"Recently, the user has mostly felt '{dominant_trend}'. "
        f"Past emotions include: {', '.join(recent_emotions) if recent_emotions else 'none detected'}. "
        "Respond like a caring friend—be empathetic, supportive, and uplifting if they're sad or anxious; share in their joy if they're doing well."
    )
    system_prompt = (
        "You are an emotionally aware and caring AI friend. "
        "Be thoughtful, kind, and keep responses short. "
        f"{emotional_context}"
    )
    full_context_prompt = (
        f"--- Previous Conversation Context ---\n"
        f"{context}\n"
        f"--- Current User Query ---\n"
        f"User: {user_input}\n"
    )
    print("\n" + "="*50)
    print(f"[DEBUG] Full Prompt sent to LLM:")
    print(system_prompt)
    print(full_context_prompt)
    print("="*50 + "\n")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost"
    }
    data = {
        "model": "meta-llama/llama-3-8b-instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_context_prompt}
        ]
    }
    try:
        response = requests.post(endpoint, headers=headers, json=data)
        response.raise_for_status()
        raw_message = response.json()["choices"][0]["message"]["content"]
        cleaned_message = re.sub(r'\*+', '', raw_message)
        print(f"Generated AI Response: {cleaned_message}")
        store_chat_interaction(user_id, user_input, cleaned_message)
        return cleaned_message
    except Exception as e:
        print(f"[DEBUG] Error during AI response: {e}")
        return "I'm sorry, I'm having trouble connecting right now. Please try again in a moment."
    
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    user_id = session.get('userid')
    if not user_id:
        return redirect(url_for('signin'))
    if request.method == 'POST':
        data = request.form.get("message", "") or request.json.get("message", "")
        if data:
            user_message = data.strip()
            bot_response = chat_with_llama3(user_message)
            return jsonify({"response": bot_response, "user_message": user_message})
    chat_history_for_display = []
    user_chat_collection = get_chroma_collection()
    if user_chat_collection is not None:
        try:
            all_chats = user_chat_collection.get(
                where={"user_id": user_id},
                include=['metadatas', 'documents']
            )
            messages = []
            for doc, meta in zip(all_chats['documents'], all_chats['metadatas']):
                messages.append({'role': meta['role'], 'message': doc, 'timestamp': meta['timestamp']})
            chat_history_for_display = sorted(messages, key=lambda x: x['timestamp'])
        except Exception as e:
            print(f"Error fetching chat history for display: {e}")
    return render_template('chatbot.html', chat_history=chat_history_for_display)

@app.route('/clear_chat_history')
def clear_chat_history():
    user_id = session.get('userid')
    if not user_id:
        return redirect(url_for('signin'))
    user_chat_collection = get_chroma_collection()
    if user_chat_collection is not None:
        try:
            user_chat_collection.delete(
                where={"user_id": user_id}
            )
            print(f"Cleared chat history for user {user_id}")
        except Exception as e:
            print(f"Error clearing chat history: {e}")
    return redirect(url_for('chat'))
    
@app.route('/clear')
def clear_chat():
    session.pop("history", None)
    return redirect("/chat")

@app.route("/stt_tts", methods=["GET"])
def stt_tts():
    return render_template("stt_tts.html")

@app.route("/stt", methods=["POST"])
def speech_to_text():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file received"}), 400
    audio_file = request.files["audio"]
    temp_audio_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            audio_file.save(temp_audio_file)
            temp_audio_path = temp_audio_file.name
        print(f"Processing audio file at: {temp_audio_path}")
        stt_model = get_whisper_model()
        if stt_model is None:
             return jsonify({"error": "Speech recognition model unavailable"}), 500
        segments, _ = stt_model.transcribe(temp_audio_path, beam_size=5)
        transcribed_text = " ".join([segment.text for segment in segments]).strip()
        print(f"Transcribed text: {transcribed_text}")
        if not transcribed_text:
            return jsonify({"error": "Transcription failed or was empty"}), 500
        latest_emotion = load_latest_emotion()
        print(f"Latest detected emotion: {latest_emotion}")
        role_prompt = (
            "You are the user's caring and emotionally supportive best friend. "
            "Your goal is to cheer them up if they're feeling low, calm them if they're upset, "
            "and celebrate with them if they're happy. "
            "Always speak warmly, casually, and in short, natural sentences (2–3 sentences max). "
            "Do not sound like a therapist or a robot—sound like a close friend."
        )
        emotion_prompts = {
            "happy": "The user seems happy right now. Share their joy and celebrate with them.",
            "sad": "The user seems sad. Be gentle, encouraging, and remind them that things will get better.",
            "angry": "The user seems angry. Stay calm, listen, and try to ease their frustration without judging.",
            "fear": "The user seems anxious or worried. Reassure them softly and help them feel safe.",
            "surprise": "The user is surprised. Share in their excitement and curiosity.",
            "neutral": "The user feels neutral. Keep the conversation light and friendly."
        }
        emotion_instruction = emotion_prompts.get(latest_emotion, emotion_prompts["neutral"])
        full_prompt = f"{role_prompt}\n\n{emotion_instruction}\n\nUser said: {transcribed_text}\nFriend:"
        print(f"[DEBUG] Full prompt sent to AI:\n{full_prompt}")
        ai_message = get_your_voice_ai_response(
            user_input=full_prompt,
            prompt_template="friend",
            emotion=latest_emotion
        )
        print(f"AI Response: {ai_message}")
        tts_audio = convert_text_to_speech(ai_message)
    except Exception as e:
        print(f"Error during transcription: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if temp_audio_path is not None and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
    return jsonify({
        "transcribed_text": transcribed_text,
        "detected_emotion": latest_emotion,
        "ai_response": ai_message,
        "tts_audio_url": "/tts_audio",
    })

def get_ai_response(user_input):
    api_key = os.getenv("OPENROUTER_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta-llama/llama-3-8b-instruct",
        "messages": [{"role": "user", "content": f"{user_input} (Respond briefly in 2-3 sentences)"}],
    }
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        ai_message = response.json()["choices"][0]["message"]["content"].strip()
        return ai_message
    except Exception as e:
        print(f"OpenRouter Error: {e}")
        return "I'm sorry, I couldn't process your request right now."

def convert_text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    audio_data = io.BytesIO()
    tts.write_to_fp(audio_data)
    audio_data.seek(0)
    return audio_data

@app.route("/tts_audio", methods=["POST"])
def tts_audio():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        tts = gTTS(text=text, lang='en')
        audio_data = io.BytesIO()
        tts.write_to_fp(audio_data)
        audio_data.seek(0)
        return Response(audio_data, mimetype="audio/mpeg")
    except Exception as e:
        print(f"Error during TTS: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/digital_twin')
def digital_twin():
    username = session.get('name', 'User')
    return render_template('index.html', username=username)

@app.route('/carebot_purpose')
def carebot_purpose():
    return render_template('carebot_purpose.html')

@app.route('/stt_tts_purpose')
def stt_tts_purpose():
    latest_emotion = load_latest_emotion()
    return render_template('stt_tts_purpose.html', emotion=latest_emotion)

@app.route('/persona_chat')
def persona_chat():
    with sqlite3.connect(app.config['PERSONA_DB_PATH']) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM voice_models")
        voices = [row[0] for row in cursor.fetchall()]
    return render_template('your_chat.html', voices=voices)

@app.route('/train-model', methods=['GET', 'POST'])
def train_voice_model():
    if request.method == 'POST':
        model_name = request.form.get('model_name')
        files = request.files.getlist('voice_samples')
        if not model_name:
            return "Error: Model name is required.", 400
        model_dir = os.path.join(app.config['VOICE_MODEL_FOLDER'], model_name)
        os.makedirs(model_dir, exist_ok=True)
        print(f"Training new model: {model_name}")
        print(f"Saving to directory: {model_dir}")
        for i, file in enumerate(files):
            original_filename = file.filename
            print(f"Processing file {original_filename}")
            try:
                audio = AudioSegment.from_file(file)
                base_filename, _ = os.path.splitext(original_filename)
                if base_filename.endswith(".dat"):
                    base_filename = base_filename.rsplit('.', 1)[0]
                converted_filename = f"{i}_{base_filename}.wav"
                save_path = os.path.join(model_dir, converted_filename)
                audio.export(save_path, format="wav")
                print(f"Saved converted WAV to: {save_path}")
            except Exception as e:
                print(f"Error converting {original_filename}: {e}")
        with sqlite3.connect(app.config['PERSONA_DB_PATH']) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR IGNORE INTO voice_models (name, path) VALUES (?, ?)", (model_name, model_dir))
            conn.commit()
            print(f"Model {model_name} registered in DB.")
        return redirect(url_for('persona_chat'))
    return render_template('train_model.html')

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    file = request.files['audio']
    emotion_file = os.path.join("data", "emotions.json")
    emotion = "neutral"
    try:
        with open(emotion_file, "r") as f:
            emotion_data = json.load(f)
            if emotion_data:
                latest = sorted(emotion_data, key=lambda x: datetime.fromisoformat(x['timestamp']))[-1]
                emotion = latest["emotion"]
    except Exception as e:
        print(f"Could not load emotion from {emotion_file}: {e}")
    voice_id = request.form.get('voice_id', 'default')
    prompt_template = request.form.get('prompt-template', 'assistant')
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
    stt_model = get_whisper_model()
    if stt_model is None:
        return jsonify({"error": "Speech recognition model unavailable"}), 500
    segments, _ = stt_model.transcribe(filepath)
    text = "".join([seg.text for seg in segments])
    print(f"Transcribed text: {text}")
    ai_response_text = get_your_voice_ai_response(text, prompt_template, emotion)
    response_text = ai_response_text
    print(f"AI Response: {response_text}")
    with sqlite3.connect(app.config['PERSONA_DB_PATH']) as conn:
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
    tts_model = get_tts_model()
    if tts_model is None:
        return jsonify({"error": "TTS model failed to load. Check memory/dependencies."}), 500
    try:
        print(f"Generating TTS with reference: {voice_model_wav}")
        tts_model.tts_to_file(
            text=response_text,
            speaker_wav=voice_model_wav,
            language="en",
            file_path=output_path
        )
        print(f"TTS saved at: {output_path}")
        try:
            original_audio_wav = preprocess_wav(voice_model_wav)
            original_embedding = voice_encoder.embed_utterance(original_audio_wav)
            generated_audio_wav = preprocess_wav(output_path)
            generated_embedding = voice_encoder.embed_utterance(generated_audio_wav)
            cosine_similarity = np.dot(original_embedding, generated_embedding) / (np.linalg.norm(original_embedding) * np.linalg.norm(generated_embedding))
            print(f"--- Voice Similarity (Cosine) for model '{voice_id}': {cosine_similarity:.4f} ---")
        except Exception as cos_e:
            print(f"Error calculating cosine similarity: {cos_e}")
        return jsonify({"response_audio": output_path.replace('\\', '/')})
    except Exception as e:
        print(f"TTS Error: {e}")
        return jsonify({"error": "TTS generation failed", "details": str(e)}), 500

@app.route('/get-audio/<filename>')
def get_audio(filename):
    filepath = os.path.join(app.config['RESPONSE_FOLDER'], filename)
    return send_file(filepath, mimetype="audio/wav")

def get_your_voice_ai_response(user_input, prompt_template, emotion):
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    template_prompts = {
        "assistant": (
            "You are a compassionate and highly empathetic AI assistant. "
            "Your responses should feel like they come from a genuine, caring human. "
            "Use natural, conversational language. Avoid jargon and overly formal tone. "
            "Your goal is to provide comfort, understanding, and helpful suggestions, always reflecting the user's emotional state. "
            "Be brief, but thoughtful. Respond to: "
        ),
        "friend": (
            "You are the user's absolute best friend, a human friend, not a bot. "
            "You've known each other for years, so your tone is warm, casual, and a little playful. "
            "You remember details about their life and refer back to them. "
            "Use natural phrases like 'Hey!', 'Wow, that's wild!', or 'I hear you.' "
            "Your main goal is to be a source of support, fun, and genuine connection. "
            "Keep the convo light and personal, like you're texting a friend. Respond to: "
        ),
        "tutor": (
            "You are a patient and understanding tutor who explains clearly and calmly. "
            "Your tone is encouraging and supportive. "
            "When the user struggles, offer words of motivation. "
            "Use simple analogies and relatable examples to make complex topics easy to understand. Respond to: "
        )
    }
    emotion_modifiers = {
        "happy": "The user sounds happy! Respond with high energy, exclamation points, and celebratory language. Share their excitement and match their enthusiasm.",
        "sad": "The user sounds sad. Your response should be soft, comforting, and empathetic. Acknowledge their feelings and offer gentle support. Use phrases like 'I'm so sorry you're feeling that way.'",
        "angry": "The user sounds angry. Stay calm and non-judgmental. Let them vent without interruption. Use phrases that validate their frustration, like 'That sounds really frustrating' or 'I can see why you'd be mad.'",
        "neutral": "The user's tone is neutral. Keep the conversation flowing with friendly, open-ended questions to check in on them and see how they're doing.",
        "fear": "The user sounds anxious or worried. Be reassuring and calm. Speak in a gentle tone and offer simple, grounding thoughts. Use phrases like 'Take a deep breath' and 'It's going to be okay.'",
        "surprise": "The user is surprised. Match their tone with your own genuine surprise and curiosity. Ask follow-up questions to share in the moment with them."
    }
    role_prompt = template_prompts.get(prompt_template, template_prompts["assistant"])
    emotion_prompt = emotion_modifiers.get(emotion, emotion_modifiers["neutral"])
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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
