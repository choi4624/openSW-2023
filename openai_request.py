import os
from openai import OpenAI
  
with open(".key","r") as key_file:
        key = key_file.read()
client = OpenAI(api_key=key)

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "Response as korean language, even request message is english. way is not limited. It can be enable translate English to Korean"},
    {"role": "user", "content": "why my sofa foam is getting flush?"}
  ]
)

print(completion.choices[0].message)