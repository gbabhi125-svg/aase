from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))
readme = ""
if os.path.exists(os.path.join(here, "README.md")):
    readme = open(os.path.join(here, "README.md"), encoding="utf-8").read()

setup(
    name="aase",
    version="0.1.0",
    description="Agent Autopsy & Self-Evolution Engine — diagnose and repair failing LLM agents",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="G B Abhilash",
    url="https://github.com/gbabhi125-svg/aase",
    packages=find_packages(include=["aase", "aase.*", "src", "src.*"]),
    python_requires=">=3.9",
    install_requires=[
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "groq":   ["groq>=0.9.0"],
        "gemini": ["google-genai>=1.0.0"],
        "ledger": ["chromadb>=0.4.0", "sentence-transformers>=2.2.0"],
        "all":    ["groq>=0.9.0", "google-genai>=1.0.0",
                   "chromadb>=0.4.0", "sentence-transformers>=2.2.0",
                   "rich>=13.0.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Quality Assurance",
        "License :: OSI Approved :: MIT License",
    ],
)