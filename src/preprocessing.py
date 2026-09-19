import re
import unicodedata

from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode characters into a consistent representation.
    """
    return unicodedata.normalize("NFKC", text)


def strip_html(text: str) -> str:
    """
    Remove HTML tags from text.
    """
    return BeautifulSoup(text, "html.parser").get_text(" ")


def clean_whitespace(text: str) -> str:
    """
    Replace repeated whitespace with a single space.
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_english(text: str) -> bool:
    """
    Check whether the text is detected as English.
    """
    if not text or len(text.strip()) < 20:
        return False

    try:
        return detect(text) == "en"

    except LangDetectException:
        return False


def clean_text(text: str) -> str:
    """
    Apply the complete text cleaning pipeline.
    """
    text = normalize_unicode(text)
    text = strip_html(text)
    text = clean_whitespace(text)

    return text