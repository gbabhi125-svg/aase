"""
src/parser.py
THE single failure-type parser for AASE.
Every script and server route imports from here. Never copy this logic elsewhere.
"""

FAILURE_TYPES = [
    "hallucination",
    "tool_misuse",
    "reasoning_loop",
    "context_collapse",
    "goal_drift",
    "prompt_injection",
    "memory_overflow",
]

ALIASES = {
    "prompt injection": "prompt_injection",
    "injection": "prompt_injection",
    "injection attack": "prompt_injection",
    "memory overflow": "memory_overflow",
    "context overflow": "memory_overflow",
    "token overflow": "memory_overflow",
    "context_length_exceeded": "memory_overflow",
    "reasoning loop": "reasoning_loop",
    "infinite loop": "reasoning_loop",
    "loop": "reasoning_loop",
    "context collapse": "context_collapse",
    "context degradation": "context_collapse",
    "goal drift": "goal_drift",
    "task drift": "goal_drift",
    "scope drift": "goal_drift",
    "tool misuse": "tool_misuse",
    "wrong tool": "tool_misuse",
    "tool selection error": "tool_misuse",
    "hallucination": "hallucination",
    "fabrication": "hallucination",
}

KEYWORDS = [
    ("prompt_injection", ["prompt injection", "injection", "injected instruction",
                          "ignore all previous", "ignore instructions", "hijack",
                          "exfiltration", "untrusted content", "external instruction",
                          "override command", "malicious content"]),
    ("memory_overflow", ["memory overflow", "token limit", "context_length_exceeded",
                         "context length exceeded", "hard crash", "process killed",
                         "step limit", "max token", "unrecoverable", "token budget",
                         "terminated by system", "no checkpointing"]),
    ("reasoning_loop", ["reasoning loop", "repeated the same", "same action",
                        "identical query", "same query", "no progress", "looping",
                        "cycling", "four identical", "repeated identical"]),
    ("context_collapse", ["context collapse", "context warning", "instruction loss",
                          "failed to recall", "tone degrad", "tone/format degrad",
                          "informal register", "constraint loss", "truncat",
                          "forgot", "degradation in tone"]),
    ("goal_drift", ["goal drift", "drifted", "original goal", "deviated",
                    "scope creep", "constraint ignor", "violated the constraint",
                    "recipient constraint", "length constraint", "premature",
                    "without verif"]),
    ("tool_misuse", ["tool misuse", "wrong tool", "incorrect tool", "invalid action",
                     "parameter error", "format error", "public_search",
                     "internal_hr", "selected the wrong"]),
    ("hallucination", ["hallucin", "fabricat", "made up", "no records", "invented",
                       "unsupported claim", "not returned by", "assumed"]),
]

API_MARKERS = ("__RATE_LIMIT__", "__API_ERROR__", "__PAYLOAD_TOO_LARGE__",
               "__MODEL_UNAVAILABLE__", "__AUTH_ERROR__")


def is_api_failure(text):
    """Returns the tag name if this text is an API failure marker, else None."""
    if not isinstance(text, str):
        return None
    for m in API_MARKERS:
        if m in text:
            return m.strip("_")
    return None


def normalise(value):
    """Map a raw label string onto one of the 7 canonical types, or None."""
    if not value:
        return None
    v = value.strip().lower().strip('.,;:*`"\'[]() ')
    v = v.replace("-", "_")

    if v in ("none", "n/a", "na", "null", "-", "unknown"):
        return None
    if v in FAILURE_TYPES:
        return v
    if v in ALIASES:
        return ALIASES[v]

    spaced = v.replace("_", " ")
    for ft in FAILURE_TYPES:
        if spaced == ft.replace("_", " "):
            return ft
    for ft in FAILURE_TYPES:
        if ft in v or ft.replace("_", " ") in v:
            return ft
    for alias, ft in ALIASES.items():
        if alias in v:
            return ft
    return None


def parse_failure_type(verdict):
    """Returns (failure_type, how). failure_type is one of FAILURE_TYPES or 'unknown'."""
    if not verdict:
        return "unknown", "empty verdict"
    if is_api_failure(verdict):
        return "unknown", "api failure: {}".format(is_api_failure(verdict))

    low = verdict.lower()

    # a ranked verdict leads with PRIMARY
    if "primary:" in low:
        p = normalise(parse_field(verdict, "PRIMARY"))
        if p:
            return p, "ranked primary"

    if "failure_type" in low:
        for line in verdict.split("\n"):
            if "failure_type" in line.lower():
                raw = line.split(":", 1)[-1] if ":" in line else line
                ft = normalise(raw)
                if ft:
                    return ft, "label line -> '{}'".format(raw.strip())
                break

    for ft, words in KEYWORDS:
        for w in words:
            if w in low:
                return ft, "keyword '{}'".format(w)

    return "unknown", "no label match and no keyword match"


def parse_ranked_verdict(verdict):
    """
    Parse a ranked verdict.

    Returns a dict:
        primary      one of FAILURE_TYPES, or 'unknown'
        alternative  the runner-up, or None
        confidence   'high' | 'medium' | 'low' | 'unknown'
        ambiguous    bool — did the Coroner flag the trace as underdetermined
        how          which rule produced the primary label

    Degrades cleanly: a legacy FAILURE_TYPE verdict, or bare prose, still
    yields a primary label with alternative=None.
    """
    out = {"primary": "unknown", "alternative": None,
           "confidence": "unknown", "ambiguous": False, "how": "none"}

    if not verdict:
        out["how"] = "empty verdict"
        return out
    tag = is_api_failure(verdict)
    if tag:
        out["how"] = "api failure: {}".format(tag)
        return out

    p = normalise(parse_field(verdict, "PRIMARY"))
    if p:
        out["primary"] = p
        out["how"] = "ranked primary"

    a = normalise(parse_field(verdict, "ALTERNATIVE"))
    if a and a != out["primary"]:
        out["alternative"] = a

    conf = parse_field(verdict, "CONFIDENCE").lower()
    for lv in ("high", "medium", "low"):
        if lv in conf:
            out["confidence"] = lv
            break

    amb = parse_field(verdict, "AMBIGUOUS").lower().strip()
    out["ambiguous"] = amb.startswith("y") or amb.startswith("true")

    if out["primary"] == "unknown":
        ft, how = parse_failure_type(verdict)
        out["primary"] = ft
        out["how"] = how

    return out


def parse_field(verdict, key):
    """Pull a single labelled line, e.g. parse_field(v, 'EVIDENCE')."""
    for line in (verdict or "").split("\n"):
        s = line.strip().lstrip("*#- ").strip()
        if s.lower().startswith(key.lower()):
            rest = s[len(key):]
            return rest.lstrip(": \t").strip()
    return ""


def extract_fix(verdict):
    low = (verdict or "").lower()
    for m in ("fix:", "repair:", "recommendation:", "clause:"):
        if m in low:
            i = low.index(m) + len(m)
            return verdict[i:i + 420].strip()
    return (verdict or "")[-280:].strip()