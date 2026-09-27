import pytest

from src.chunking import (
    recursive_split,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)


def test_empty_text_returns_empty_list():
    assert recursive_split("") == []


def test_whitespace_only_text_returns_empty_list():
    assert recursive_split("     \n\n   ") == []


def test_short_text_remains_single_chunk():
    text = "Machine learning is useful."

    chunks = recursive_split(
        text,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) == 1
    assert chunks[0] == text


def test_long_text_creates_multiple_chunks():
    text = "Machine learning is useful. " * 100

    chunks = recursive_split(
        text,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) > 1


def test_chunks_respect_maximum_size():
    text = "Artificial intelligence and machine learning. " * 100

    chunks = recursive_split(
        text,
        chunk_size=200,
        chunk_overlap=30,
    )

    assert all(len(chunk) <= 200 for chunk in chunks)


def test_chunks_are_not_empty():
    text = "Artificial intelligence. " * 50

    chunks = recursive_split(
        text,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert all(chunk.strip() for chunk in chunks)


def test_overlap_preserves_context():
    text = (
        "Machine learning models learn patterns from data. "
        "Neural networks are widely used in deep learning. "
        "Transformers are important for modern natural language processing."
    )

    chunks = recursive_split(
        text,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) > 1

    # Neighboring chunks should share some contextual text
    # when overlap is enabled.
    overlap_found = False

    for first, second in zip(chunks, chunks[1:]):
        first_words = first.split()
        second_words = second.split()

        if first_words and second_words:
            if any(
                word in second_words
                for word in first_words[-3:]
            ):
                overlap_found = True
                break

    assert overlap_found


def test_zero_overlap_is_supported():
    text = "Machine learning. " * 50

    chunks = recursive_split(
        text,
        chunk_size=100,
        chunk_overlap=0,
    )

    assert len(chunks) > 1


def test_invalid_chunk_size_raises_error():
    with pytest.raises(ValueError):
        recursive_split(
            "some text",
            chunk_size=0,
            chunk_overlap=0,
        )


def test_negative_overlap_raises_error():
    with pytest.raises(ValueError):
        recursive_split(
            "some text",
            chunk_size=100,
            chunk_overlap=-1,
        )


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        recursive_split(
            "some text",
            chunk_size=100,
            chunk_overlap=100,
        )


def test_non_string_input_raises_error():
    with pytest.raises(TypeError):
        recursive_split(
            None,
            chunk_size=100,
            chunk_overlap=20,
        )


def test_default_configuration_is_valid():
    assert DEFAULT_CHUNK_SIZE > DEFAULT_CHUNK_OVERLAP
    assert DEFAULT_CHUNK_OVERLAP >= 0