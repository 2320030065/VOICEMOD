import speech_recognition as sr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import re

# Load Hugging Face model
print("🔄 Loading model (first time may take a while)...")
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-multi")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-multi")
device = 0 if torch.cuda.is_available() else -1
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

def detect_language_from_input(text):
    langs = ["C", "C++", "Java", "Python", "JavaScript", "Go", "Rust"]
    
    # Use regex to detect exact words
    for lang in langs:
        if re.search(rf"\b{lang}\b", text, re.IGNORECASE):
            print(f"✅ Detected language: {lang}")  # Debugging print
            return lang

    print("❌ Language not detected. Defaulting to C.")
    return "C"

def format_prompt(text, lang):
    prompt = f"// Write a simple and correct {lang} program to {text.strip()}\n"
    
    if lang == "C":
        prompt += "#include <stdio.h>\nint main() {\n"
    elif lang == "C++":
        prompt += "#include <iostream>\nusing namespace std;\nint main() {\n"
    elif lang == "Java":
        prompt += "public class Main {\n    public static void main(String[] args) {\n"
    elif lang == "Python":
        prompt += ""
    elif lang == "JavaScript":
        prompt += "function main() {\n"
    elif lang == "Go":
        prompt += "package main\nimport \"fmt\"\nfunc main() {\n"
    elif lang == "Rust":
        prompt += "fn main() {\n"
    
    return prompt

def clean_output(code, lang):
    # Ensure we correctly extract Java code
    if lang == "Java":
        start = code.find("public class")
        end = code.rfind("}")  # last closing brace
        if start != -1 and end != -1:
            return code[start:end + 1]

    return code.strip()

def generate_code(prompt, lang):
    result = generator(prompt, max_length=256, do_sample=True, temperature=0.5, truncation=True)
    raw_code = result[0]["generated_text"]
    
    print("🚀 Raw Generated Code:\n", raw_code)  # Debugging print
    
    return clean_output(raw_code, lang)

def listen_and_generate_code():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Speak your coding task...")
        audio = recognizer.listen(source)

        try:
            voice_input = recognizer.recognize_google(audio)
            print(f"🗣️ You said: {voice_input}")
            print("🧠 Generating code...\n")

            lang = detect_language_from_input(voice_input)
            prompt = format_prompt(voice_input, lang)
            code = generate_code(prompt, lang)

            print("💻 Generated Code:\n")
            print(code)

        except sr.UnknownValueError:
            print("❌ Could not understand your speech.")
        except sr.RequestError as e:
            print(f"⚠️ Could not request results; {e}")

if __name__ == "__main__":
    listen_and_generate_code()
