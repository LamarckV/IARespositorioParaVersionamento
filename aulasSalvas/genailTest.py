from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# for m in client.models.list():
#     print(m.name)

try:
    response = client.models.generate_content(
        model= "gemini-3-flash-preview",
        contents="se ari é meu pai, Bruno é meu irmão, Carolina é oque minha?",
        config=genai.types.GenerateContentConfig(
            system_instruction="Inditifique padrões nos nomes, procure por coisas que combinam, analises de letras e coisas do tipo",
            temperature=0.7,
            top_p=0.95,
            # max_output_tokens=500
            # stop_sequences=["\n\n"]
        )
    )
    print(response.text)

except Exception as e:
    print("erro ao consumir api", e)
