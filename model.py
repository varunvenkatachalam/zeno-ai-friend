import os
import tempfile
import pyaudio
import wave
import ollama
from gtts import gTTS
import assemblyai as aai  # AssemblyAI for speech-to-text
import numpy as np
import random

class Zeno:
    def __init__(self) -> None:
        """Initialize the AI assistant with AssemblyAI for speech recognition and Llama 3 for responses."""
        # Set up AssemblyAI
        aai.settings.api_key = "ddad5326519041a3a1b63877c7c95e83"  # Replace with your AssemblyAI API key
        self.full_transcript = [
            {"role": "system", "content": "You are Zeno, an AI assistant. Answer only based on the user's input and maintain a continuous conversation."}
        ]

    def record_audio(self, filename="temp.wav", silence_threshold=500, silence_duration=2):
        """Records audio from the microphone until there is a 2-second gap of silence."""
        print("\n🎙️ Recording... Speak now!")  
        
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
            data = stream.read(CHUNK)
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
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(filename)
        return transcript.text.strip()

    def generate_ai_response(self, text):
        """Generate response based on conversation history and make it sound natural."""
        if not text:
            print("No text detected. Please speak again.")
            return

        self.full_transcript.append({"role": "user", "content": text})
        print(f"👤 You: {text}")

        ollama_response = ollama.chat(
            model="llama3",
            messages=self.full_transcript,  # Keeps past messages for context
            stream=False,
        )

        full_text = ollama_response["message"]["content"].strip()

        # Make the response sound more natural
        natural_text = self.make_it_sound_real(full_text)

        self.full_transcript.append({"role": "assistant", "content": natural_text})

        print(f"🤖 Zeno: {natural_text}")
        self.speak(natural_text)

    def make_it_sound_real(self, text):
        """Modify AI response to make it sound like a real person talking."""
        fillers = ["Well,", "You know,", "Honestly,", "I mean,", "So,", "Let's see..."]
        pauses = ["... Hmm.", "... Let me think.", "... That's a good one."]

        # Insert a filler at the start sometimes
        if random.random() < 0.3:  # 30% chance
            text = random.choice(fillers) + " " + text

        # Insert pauses in longer responses
        if len(text.split()) > 10 and random.random() < 0.4:  # 40% chance
            parts = text.split(", ")
            if len(parts) > 1:
                index = random.randint(0, len(parts) - 1)
                parts.insert(index, random.choice(pauses))
                text = ", ".join(parts)

        # Add contractions for more natural speech
        contractions = {
            "I am": "I'm",
            "you are": "you're",
            "do not": "don't",
            "is not": "isn't",
            "let us": "let's",
            "it is": "it's",
            "that is": "that's",
            "cannot": "can't",
            "would not": "wouldn't",
            "should not": "shouldn't",
            "will not": "won't"
        }
        for full, contracted in contractions.items():
            text = text.replace(full, contracted)

        return text

    def speak(self, text):
        """Convert AI text response to speech and play it naturally."""
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as temp_audio:
            tts = gTTS(text=text, lang="en", slow=False)  # Slow=False makes it sound more natural
            tts.save(temp_audio.name)
            os.system(f"afplay {temp_audio.name}")  # macOS (Use 'mpg321' for Linux)

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

zeno = Zeno()
zeno.start()
