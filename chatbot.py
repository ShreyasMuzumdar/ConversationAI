import pyttsx3
from transformers import pipeline

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load the GPT-J model from Hugging Face
chatbot = pipeline("text-generation", model="EleutherAI/gpt-j-6B", device=-1)

# Function to make the AI speak
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Start a conversation loop
print("Hello! I am your chatbot. Type 'exit' to end the conversation.")
while True:
    # Get user input
    user_input = input("You: ")
    
    # Exit condition
    if user_input.lower() == "exit":
        speak("Goodbye!")
        print("Goodbye!")
        break
    
    # Generate response from GPT-J
    response = chatbot(user_input, max_length=150, num_return_sequences=1)
    
    # Extract the generated text
    ai_response = response[0]['generated_text']
    
    # Print and speak the response
    print(f"AI: {ai_response}")
    speak(ai_response)
