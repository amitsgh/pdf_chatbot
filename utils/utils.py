import re

def preprocess_content(text: str) -> str:
    if not text:
        return ""

    # Lowercase for normalization (optional)
    text = text.lower()

    # Remove divider lines or excessive symbols
    text = re.sub(r"([_\-=*~#]{3,})", " ", text)

    # Remove emails, URLs, and page numbering
    text = re.sub(r"page\s*\d+\s*(of)?\s*\d*", " ", text)

    # Remove lines with only punctuation/symbols (non-alphanumeric)
    text = "\n".join([line for line in text.splitlines() if re.search(r"\w", line)])

    # Remove timestamps/dates (if needed)
    text = re.sub(r"\b\d{1,2}[:.]\d{2}(?:[:.]\d{2})?\s*(am|pm)?\b", " ", text)
    text = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", " ", text)

    # Normalize whitespace and punctuation
    text = re.sub(r"[^\w\s\-.,]", " ", text)  # preserve -, . and ,
    text = re.sub(r"\s+", " ", text).strip()

    return text