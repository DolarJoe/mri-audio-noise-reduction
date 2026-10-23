"""Direct chat call to the E-infra endpoint (replaces headless pi calls)."""
import json
import time
from pathlib import Path

import requests

API_URL = "https://llm.ai.e-infra.cz/v1/chat/completions"


def api_key() -> str:
    auth = json.loads((Path.home() / ".pi/agent/auth.json").read_text())
    return auth.get("E-infra", auth.get("einfra"))["key"]


def api_call(prompt: str, model: str = "deepseek", retries: int = 3) -> str:
    key = api_key()
    err = ""
    for attempt in range(retries):
        try:
            r = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {key}"},
                json={"model": model,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=180)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
            err = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:  # noqa: BLE001
            err = str(e)
        time.sleep(3 * (attempt + 1))
    return f"CALL_FAILED: {err[:200]}"
