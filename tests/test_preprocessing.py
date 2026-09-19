from src.preprocessing import (
    normalize_unicode,
    strip_html,
    clean_whitespace,
    is_english,
    clean_text,
)


def test_unicode_normalization():

    result = normalize_unicode("café")

    assert result == "café"


def test_strip_html():

    result = strip_html(
        "<p>Hello <b>world</b></p>"
    )

    assert "Hello" in result
    assert "world" in result
    assert "<p>" not in result


def test_clean_whitespace():

    result = clean_whitespace(
        "Hello     world\n\nmachine learning"
    )

    assert result == "Hello world machine learning"


def test_english_detection():

    result = is_english(
        "Machine learning is a field of artificial intelligence."
    )

    assert result is True


def test_non_english_detection():

    result = is_english(
        "Bonjour tout le monde."
    )

    assert result is False


def test_clean_text():

    result = clean_text(
        "<p>Hello    world</p>"
    )

    assert result == "Hello world"