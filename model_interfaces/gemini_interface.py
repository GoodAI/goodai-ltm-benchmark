import os
import textwrap
import google.generativeai as genai


GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)


model = genai.GenerativeModel('gemini-pro')
response = model.generate_content("tell me a joke")
print(response.text)
exit(0)

raw_response = response.text  # This has the raw response
chat = model.start_chat(history=[])
response = chat.send_message("Okay, how about a more detailed e...")

# model.count_tokens("What is the meaning of life?")
# model.count_tokens(chat.history)

for message in chat.history:
    print(f'**{message.role}**: {message.parts[0].text}')
