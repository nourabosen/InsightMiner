import re


def clean_text(text):
    """Clean text but preserve quotes."""
    if not text or not isinstance(text, str):
        return ""

    if "## Quotes" in text:
        return text

    text = re.sub(r"background-color:: \w+", "", text)
    text = re.sub(r"==|[\*\_\[\]\(\)#-]", "", text)
    text = re.sub(r'(\w) "(\w)', r'\1 "\2', text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
