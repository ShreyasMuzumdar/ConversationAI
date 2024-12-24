import time
import pyttsx3
import speech_recognition as sr
from transformers import pipeline

# Setup speech recognition
def listen_for_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    try:
        command = recognizer.recognize_google(audio)
        print(f"Command received: {command}")
        return command
    except sr.UnknownValueError:
        print("Sorry, I didn't catch that.")
    except sr.RequestError:
        print("Sorry, the service is unavailable.")

# Setup text-to-speech
def speak_response(response_text):
    engine = pyttsx3.init()
    engine.say(response_text)
    engine.runAndWait()

# Setup conversational AI
def get_ai_response(input_text):
    generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")
    response = generator(input_text, max_length=100, num_return_sequences=1)
    return response[0]['generated_text']

# Main conversational loop
def conversational_ai():
    while True:
        command = listen_for_command()

        if command:
            ai_response = get_ai_response(command)
            speak_response(ai_response)

        time.sleep(1)

# Run the conversational AI
conversational_ai()
