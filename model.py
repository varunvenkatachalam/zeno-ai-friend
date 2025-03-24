import os
import tempfile
import requests
import pyaudio
import wave
import json
import pyttsx3
import numpy as np
from vosk import Model, KaldiRecognizer
import queue
import assemblyai as aai  # Import AssemblyAI

class Zeno:
    def __init__(self) -> None:
        """Initialize the AI assistant with AssemblyAI for speech recognition and Groq API for responses."""
        # Set up AssemblyAI
        aai.settings.api_key = "ddad5326519041a3a1b63877c7c95e83"  # Replace with your actual API key
        self.transcriber = aai.Transcriber()

        # Initialize text-to-speech
        self.tts_engine = pyttsx3.init()

        # API Details
        self.api_key = "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"  # Replace with your actual Groq API Key
        self.url = "https://api.groq.com/openai/v1/chat/completions"

        self.full_transcript = [
            {"role": "system", "content": "You are Zeno, an AI assistant. Answer only based on the user's input and maintain a continuous conversation."}
        ]
    
    def record_audio(self, filename="temp.wav", silence_threshold=500, silence_duration=3):
        """Records audio from the microphone until there is a 3-second gap of silence."""
        print("\n🎙 Recording... Speak now!")  
        
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        CHUNK = 1024

        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        
        frames = []
        silent_frames = 0
        recording = True

        while recording:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

            # Convert audio data to numpy array to analyze volume
            audio_data = np.frombuffer(data, dtype=np.int16)
            volume = np.abs(audio_data).mean()

            # Detect silence
            if volume < silence_threshold:
                silent_frames += 1
            else:
                silent_frames = 0

            # Stop recording if silence exceeds the specified duration
            if silent_frames > (RATE / CHUNK * silence_duration):
                recording = False

        stream.stop_stream()
        stream.close()
        audio.terminate()

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
    
    def transcribe_audio(self, filename="temp.wav"):
        """Transcribes audio using AssemblyAI."""
        try:
            transcript = self.transcriber.transcribe(filename)
            return transcript.text.strip()
        except Exception as e:
            print(f"Error in transcription: {e}")
            return ""
    
    def generate_ai_response(self, text):
        """Generate response using Groq API and play the response."""
        if not text:
            print("No text detected. Please speak again.")
            return

        self.full_transcript.append({"role": "user", "content": text})
        print(f"👤 You: {text}")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama3-8b-8192",
            "messages": self.full_transcript,
            "temperature": 0.7,
            "max_tokens": 100
        }
        response = requests.post(self.url, headers=headers, json=payload)
        
        if response.status_code == 200:
            full_text = response.json()["choices"][0]["message"]["content"].strip()
        else:
            full_text = "Sorry, I couldn't process your request."

        self.full_transcript.append({"role": "assistant", "content": full_text})
        print(f"🤖 Zeno: {full_text}")
        self.speak(full_text)
    
    def speak(self, text):
        """Convert AI text response to speech and play it."""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def start(self):
        """Runs the assistant in a continuous loop until stopped."""
        print("\n🟢 Zeno is ready! Speak to start the conversation. (Press Ctrl + C to exit)")
        
        while True:
            try:
                self.record_audio()
                text = self.transcribe_audio()
                if text:
                    self.generate_ai_response(text)
            except KeyboardInterrupt:
                print("\n🔴 Conversation ended.")
                break
            except Exception as e:
                print(f"Error: {e}")

# Run Zeno (continuous mode)
zeno = Zeno()
zeno.start()
