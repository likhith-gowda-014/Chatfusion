# ChatFusion

**ChatFusion** is an emotionally-aware chatbot that utilizes advanced AI features to create a more personalized and empathetic user experience. The project is built on the Flask web framework and integrates various AI models for speech-to-text, text-to-speech, and natural language processing.

## 🌟 Features

* **Emotional Chatbot**: The application uses **DeepFace** to continuously capture the user's emotion by analyzing their facial expressions from a webcam feed. The detected emotion (e.g., happy, sad, angry) is stored in a `emotions.json` file. The latest emotion is then used to dynamically adjust the AI's response to provide a more empathetic and contextually relevant conversation.

* **Speech-to-Text (STT) and Text-to-Speech (TTS)**: This feature allows for seamless audio-based interaction.
    * **STT**: The **`faster_whisper`** model transcribes user-uploaded audio into text. This transcription is then fed to the AI for a response.
    * **TTS**: The AI's text response is converted back into an audio file using either **gTTS** (for basic functionality) or the more advanced **XTTSv2** model for voice cloning.

* **Personalized Voice Model Training**: This is a key feature that allows users to create and interact with a digital twin of their own voice.
    * **Zero-Shot Learning**: The **XTTSv2** model is a zero-shot learning model, which means it can clone a voice and generate speech in that voice using a short audio reference clip without extensive training.
    * **Model Training Flow**: Users upload a few audio samples, which the application saves and registers in an SQLite database.
    * **Voice Generation**: When a user selects their trained model, the application uses one of the stored audio samples as a **reference voice** to generate the AI's response in that specific voice.

## 🧠 Workflow

### 1. User Interaction and Emotion Detection

The user's journey begins on the `home.html` page. Once the user logs in, they can access the `dashboard` and the chat features. A separate thread runs the `capture_emotion()` function, which uses OpenCV and DeepFace to analyze webcam frames every 3 seconds. The detected dominant emotion is stored in `data/emotions.json` with a timestamp.

### 2. Audio Processing and AI Response Generation

When a user speaks, the audio is sent to the `/upload-audio` endpoint.

1.  The audio file is saved temporarily, converted to a WAV format, and then transcribed into text using `faster_whisper`.
2.  The application loads the latest emotion from the `emotions.json` file.
3.  The `get_ai_response_unified()` function crafts a system prompt that combines a chosen **role** (assistant, friend, or tutor) with an **emotional modifier** based on the detected emotion.
4.  This tailored prompt and the user's transcribed text are sent to the **OpenRouter API** to get a response from an LLM like Llama-3.
5.  The AI's text response is then passed to the **XTTSv2** model. The model uses a reference audio clip from the user's previously trained voice model to generate an audio response that mimics their voice.
6.  The final audio file is saved and returned to the user.

### 3. Future Roadmap

The project aims to integrate **ChromaDB** with a **vector embedding model**. This will allow the chatbot to store and retrieve relevant conversation history, providing the LLM with richer emotional context for more consistent and coherent responses over time.
