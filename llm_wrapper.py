
import os
import requests
from dotenv import load_dotenv
load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M")
TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "512"))

def query_ollama(prompt: str, stream=False,
                 temperature=TEMPERATURE, num_predict=NUM_PREDICT) -> str:
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": stream,
        "temperature": temperature,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "num_predict": num_predict,
    }
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    return r.json().get("response", "")

SYSTEM_COMPLIANCE = (
    "You are a compliance assistant. Answer ONLY using the provided context. "
    "Cite doc and page where possible. If evidence is missing, reply 'Insufficient evidence.'"
)

def build_prompt(question: str, context: str = "") -> str:
    parts = [f"System:\\n{SYSTEM_COMPLIANCE}\\n"]
    if context:
        parts.append(f"Context:\\n{context}\\n")
    parts.append(f"User:\\n{question}\\nAssistant:")
    return "\\n".join(parts)


