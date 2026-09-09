"""
scripts/test_gemini.py — find out exactly why Gemini is rejecting
Run: python scripts/test_gemini.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

key = os.getenv("GEMINI_API_KEY")
print("=" * 70)
print("1. KEY CHECK")
print("=" * 70)
print(f"GEMINI_API_KEY present : {bool(key)}")
if key:
    print(f"Starts with            : {key[:10]}...")
    print(f"Length                 : {len(key)}")
    print(f"Looks like Google key  : {key.startswith('AIza')}")
else:
    print("NO KEY. Add GEMINI_API_KEY to .env")
    sys.exit(1)

print()
print("=" * 70)
print("2. LIST MODELS YOUR KEY CAN ACTUALLY USE")
print("=" * 70)
import google.generativeai as genai
genai.configure(api_key=key)
try:
    usable = []
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            usable.append(m.name)
            print(f"  {m.name}")
    if not usable:
        print("  NONE — key has no generateContent access")
except Exception as e:
    print(f"  FAILED to list: {e}")
    usable = []

print()
print("=" * 70)
print("3. TINY TEST CALL")
print("=" * 70)
for candidate in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-pro"]:
    try:
        model = genai.GenerativeModel(candidate)
        r = model.generate_content("Say OK")
        print(f"  {candidate:<28} WORKS → {r.text.strip()[:40]}")
    except Exception as e:
        msg = str(e)
        short = msg[:150].replace("\n", " ")
        print(f"  {candidate:<28} FAIL → {short}")

print()
print("=" * 70)
print("4. VERDICT")
print("=" * 70)
print("If ALL models 429 on the very first call → billing/region block, not usage quota.")
print("If some model WORKS → put that exact name in .env as GEMINI_MODEL")