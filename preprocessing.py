"""
Text preprocessing utilities for the Comment Toxicity project.
"""
import re

def clean_text(text: str) -> str:
    """Lowercase, strip URLs/IPs/newlines/special chars, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)           # URLs
    text = re.sub(r"\d{1,3}(?:\.\d{1,3}){3}", " ", text)         # IP addresses
    text = re.sub(r"\n", " ", text)                              # newlines
    text = re.sub(r"[^a-z0-9'!?., ]+", " ", text)                # keep basic punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text
