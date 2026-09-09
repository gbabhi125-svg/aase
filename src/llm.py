"""
src/llm.py
Single LLM entry point for every AASE agent.
Switch provider by changing LLM_PROVIDER in .env — no agent file needs editing.

.env:
  LLM_PROVIDER=gemini        # gemini | groq
  GEMINI_API_KEY=...
  GEMINI_MODEL=gemini-2.0-flash
  GROQ_API_KEY=...
  GROQ_MODEL=qwen/qwen3.8-27b
"""

import os, time
from dotenv import load_dotenv

load_dotenv()

PROVIDER     = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()
GROQ_MODEL   = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b").strip()

# Free-tier friendly pacing (seconds between calls). 0 disables.
MIN_GAP = float(os.getenv("LLM_MIN_GAP", "0"))
_last_call = [0.0]


class LLMError(Exception):
    def __init__(self, tag, message):
        self.tag = tag
        self.message = message
        super().__init__(f"{tag}: {message}")


def _classify(e):
    m = str(e)
    low = m.lower()
    if ("429" in m or "rate limit" in low or "rate_limit" in low
            or "quota" in low or "resource_exhausted" in low or "exhausted" in low):
        return "RATE_LIMIT"
    if "413" in m or "too large" in low or "payload" in low:
        return "PAYLOAD_TOO_LARGE"
    if "404" in m or "not found" in low or "does not exist" in low or "decommission" in low:
        return "MODEL_UNAVAILABLE"
    if "401" in m or "403" in m or "api key" in low or "permission denied" in low:
        return "AUTH_ERROR"
    if "deadline" in low or "timeout" in low or "504" in m:
        return "TIMEOUT"
    return "API_ERROR"


def _pace():
    if MIN_GAP <= 0:
        return
    wait = MIN_GAP - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()


# ── Gemini (new google-genai SDK, REST) ───────────────────────────
_gclient = None
def _gemini_client():
    global _gclient
    if _gclient is None:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise LLMError("AUTH_ERROR", "GEMINI_API_KEY missing from .env")
        try:
            from google import genai
            from google.genai import types  # noqa
        except ImportError:
            raise LLMError("API_ERROR",
                           "google-genai not installed. Run: pip install google-genai")
        _gclient = genai.Client(api_key=key,
                                http_options={"api_version": "v1beta"})
    return _gclient


def _call_gemini(system, user, max_tokens):
    from google.genai import types
    c = _gemini_client()
    r = c.models.generate_content(
        model=GEMINI_MODEL,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=0.3,
        ),
    )
    txt = getattr(r, "text", None)
    if txt:
        return txt
    try:
        return "".join(p.text for p in r.candidates[0].content.parts if getattr(p, "text", None))
    except Exception:
        raise LLMError("API_ERROR", "empty response from Gemini")


# ── Groq ──────────────────────────────────────────────────────────
_qclient = None
def _groq_client():
    global _qclient
    if _qclient is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise LLMError("AUTH_ERROR", "GROQ_API_KEY missing from .env")
        from groq import Groq
        _qclient = Groq(api_key=key)
    return _qclient


def _call_groq(system, user, max_tokens):
    c = _groq_client()
    r = c.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
    )
    return r.choices[0].message.content


# ── Public API ────────────────────────────────────────────────────
def complete(system: str, user: str, max_tokens: int = 400) -> str:
    """Single call. Raises LLMError with a .tag on failure."""
    _pace()
    try:
        if PROVIDER == "groq":
            return _call_groq(system, user, max_tokens)
        return _call_gemini(system, user, max_tokens)
    except LLMError:
        raise
    except Exception as e:
        raise LLMError(_classify(e), str(e))


def safe_complete(system: str, user: str, max_tokens: int = 400, who: str = "agent") -> str:
    """Never raises. On failure returns a string starting __TAG__ so callers can detect it."""
    try:
        return complete(system, user, max_tokens)
    except LLMError as e:
        return f"__{e.tag}__ {who} did not run.\nError: {e.message[:400]}"


def model_name():
    return GROQ_MODEL if PROVIDER == "groq" else GEMINI_MODEL


def provider():
    return PROVIDER


def selftest():
    """Quick check that the provider works. Run: python -m src.llm"""
    print(f"provider = {provider()}")
    print(f"model    = {model_name()}")
    t0 = time.time()
    try:
        out = complete("Reply with the single word OK.", "ping", 20)
        print(f"result   = OK ({time.time()-t0:.1f}s) → {out.strip()[:60]}")
    except LLMError as e:
        print(f"result   = {e.tag}")
        print(f"detail   = {e.message[:400]}")


if __name__ == "__main__":
    selftest()