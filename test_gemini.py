import os
import google.generativeai as genai

# Load the API key from environment variable
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("No GEMINI_API_KEY found. Please set the GEMINI_API_KEY environment variable.")

# Configure the API
genai.configure(api_key=api_key)

# Choose a model
model = genai.GenerativeModel('gemini-1.5-flash')

# Define the prompt
prompt = "What's the weather like today in Dubrovnik?"

# Generate the response
response = model.generate_content(prompt)

# Print the response
print(response.text)
