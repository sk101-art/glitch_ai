import os, json, requests, httpx
from typing import Dict, Iterable, Optional
from dotenv import load_dotenv
load_dotenv()

OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
MODEL        = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M")
TEMPERATURE  = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
NUM_PREDICT  = int(os.getenv("OLLAMA_NUM_PREDICT", "512"))

def generate_sync(prompt: str,
                  model: Optional[str] = None,
                  temperature: float = TEMPERATURE,
                  num_predict: int = NUM_PREDICT,
                  **kw) -> str:
    payload: Dict = {
        "model": model or MODEL,
        "prompt": prompt,
        "stream": False,
        "temperature": temperature,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "num_predict": num_predict
    }
    payload.update(kw)
    url = f"{OLLAMA_URL}/api/generate"
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    return r.json().get("response", "")

SYSTEM_COMPLIANCE = (
    "You are a compliance assistant. Write clearly, cite facts, "
    "and avoid speculation. If you lack evidence, say 'Insufficient evidence.'"
)

def build_prompt(system: str, user: str, context: str = "") -> str:
    parts = []
    if system:
        parts.append(f"System:\n{system}\n")
    if context:
        parts.append(f"Context:\n{context}\n")
    parts.append(f"User:\n{user}\nAssistant:")
    return "\n".join(parts)
