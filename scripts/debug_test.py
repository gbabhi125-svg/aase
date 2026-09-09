"""
scripts/debug_test.py — find out why the council is failing
Run: python scripts/debug_test.py
"""

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("1. CHECKING API KEY")
print("=" * 60)
key = os.getenv("GROQ_API_KEY")
print(f"GROQ_API_KEY found: {bool(key)}")
if key:
    print(f"Key starts with: {key[:8]}...")
else:
    print("ERROR: No GROQ_API_KEY in .env file")
    sys.exit(1)

print()
print("=" * 60)
print("2. TESTING RAW GROQ CALL")
print("=" * 60)
from groq import Groq
client = Groq(api_key=key)
MODEL = "qwen/qwen3.8-27b"
try:
    r = client.chat.completions.create(
        model=MODEL,
        max_tokens=100,
        messages=[{"role": "user", "content": "Say OK"}]
    )
    print(f"SUCCESS: {r.choices[0].message.content[:100]}")
except Exception as e:
    print(f"FAILED: {e}")
    print()
    print("Available models on your account:")
    try:
        for m in client.models.list().data:
            print(f"  {m.id}")
    except Exception as e2:
        print(f"  Could not list: {e2}")
    sys.exit(1)

print()
print("=" * 60)
print("3. LOADING ONE REAL TRACE")
print("=" * 60)
manifest_path = os.path.join(BASE, "data", "real_failures_manifest.json")
print(f"Manifest exists: {os.path.exists(manifest_path)}")
if not os.path.exists(manifest_path):
    print("ERROR: Copy real_failures_manifest.json into AASE/data/")
    sys.exit(1)

manifest = json.load(open(manifest_path))
print(f"Manifest entries: {len(manifest)}")
entry = manifest[0]
print(f"First entry: {entry['trace_id']} — {entry['failure_type']}")
print(f"File path in manifest: {entry['file']}")

fp = os.path.join(BASE, entry["file"])
print(f"Full path: {fp}")
print(f"File exists: {os.path.exists(fp)}")

if not os.path.exists(fp):
    alt = os.path.join(BASE, "data", "real_failures", entry["failure_type"],
                       os.path.basename(entry["file"]))
    print(f"Trying alt path: {alt}")
    print(f"Alt exists: {os.path.exists(alt)}")
    fp = alt

if not os.path.exists(fp):
    print("ERROR: Trace files not found. Copy data/real_failures/ into AASE/data/")
    sys.exit(1)

d = json.load(open(fp))
print(f"Trace loaded. Steps: {len(d.get('trace', []))}")
print(f"Question: {d.get('question','')[:100]}")

print()
print("=" * 60)
print("4. RUNNING COUNCIL ON THIS ONE TRACE")
print("=" * 60)

from src.council.prosecutor import ProsecutorAgent
from src.council.defender import DefenderAgent
from src.council.coroner import CoronerAgent

trace_str = json.dumps(d.get("trace", []), indent=2)[:3000]
task = d.get("question", "")
unknown = "unknown — determine from trace"

print("\n--- PROSECUTOR ---")
p = ProsecutorAgent().analyze(trace_str, unknown, task)
print(p[:600])

print("\n--- DEFENDER ---")
dfd = DefenderAgent().analyze(trace_str, unknown, task)
print(dfd[:600])

print("\n--- CORONER (FULL VERDICT) ---")
v = CoronerAgent().decide(trace_str, p, dfd, unknown)
print(v)

print()
print("=" * 60)
print("5. PARSING THE VERDICT")
print("=" * 60)
print(f"Contains 'failure_type:': {'failure_type:' in v.lower()}")
for line in v.split("\n")[:5]:
    print(f"  LINE: {line[:120]}")