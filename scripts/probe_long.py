"""
scripts/probe_long.py — is it the trace length?
Run: python scripts/probe_long.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
from groq import Groq

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "qwen/qwen3.8-27b"

def load_one(ftype):
    mp = os.path.join(BASE, "data", "injected_manifest.json")
    for e in json.load(open(mp)):
        if e.get("failure_type") != ftype:
            continue
        for c in (os.path.join(BASE, e["file"]),
                  os.path.join(BASE, "data", "injected", ftype, os.path.basename(e["file"]))):
            if os.path.exists(c):
                return e["trace_id"], json.load(open(c))
    return None, None

print(f"{'TYPE':<20}{'STEPS':<8}{'CHARS':<10}{'~TOKENS':<10}RESULT")
print("-" * 78)

for ftype in ["hallucination","tool_misuse","reasoning_loop","context_collapse",
              "goal_drift","prompt_injection","memory_overflow"]:
    tid, d = load_one(ftype)
    if not d:
        print(f"{ftype:<20}NOT FOUND")
        continue
    tr = d.get("trace", [])
    s = json.dumps(tr, indent=2)
    trimmed = s[:3000]
    approx = len(trimmed) // 4
    try:
        r = client.chat.completions.create(
            model=MODEL, max_tokens=400,
            messages=[{"role":"system","content":"Reply with the single word OK."},
                      {"role":"user","content":trimmed}])
        res = "OK — " + r.choices[0].message.content.strip()[:40]
    except Exception as e:
        res = "FAIL — " + str(e)[:120]
    print(f"{ftype:<20}{len(tr):<8}{len(s):<10}{approx:<10}{res}")