import sounddevice as sd
from pvporcupine import Porcupine
from llama_cpp import Llama
from TTS.api import TTS

# Load Models
llama = Llama(model_path="./llama-7b.ggmlv3.q4_0.bin")  # Path to LLaMA model
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
porcupine = Porcupine(
    access_key="+b/VvsTn8hkmOXbtwUqHcw6L+6DhoZzoc0og+pzqHrou7A5v6nY40w==", keyword_paths=["Hello-Joe_en_mac_v3_0_0.ppn"]
)

# Wake-word detection
def detect_wake_word():
    with sd.InputStream(channels=1, samplerate=porcupine.sample_rate) as stream:
        print("Listening for wake word...")
        while True:
            audio_frame = stream.read(porcupine.frame_length)[0]
            keyword_index = porcupine.process(audio_frame)
            if keyword_index >= 0:
                print("Wake word detected!")
                return

# Process user input
def chat():
    user_input = input("You: ")
    response = llama(user_input)["choices"][0]["text"]
    print(f"Assistant: {response}")
    tts.tts_to_file(response, speaker="multispeaker", file_path="response.wav")
    sd.playfile("response.wav")

# Main Loop
try:
    while True:
        detect_wake_word()
        chat()
except KeyboardInterrupt:
    print("Shutting down...")
finally:
    porcupine.delete()
