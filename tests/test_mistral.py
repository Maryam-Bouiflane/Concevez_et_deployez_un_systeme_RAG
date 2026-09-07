from dotenv import load_dotenv
from mistralai.client import Mistral
import os

load_dotenv()
client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

response = client.chat.complete(
    model="mistral-small-2603",
    messages=[
        {
            "role": "user",
            "content": "Réponds simplement : OK",
        }
    ],
)

print(response.choices[0].message.content)