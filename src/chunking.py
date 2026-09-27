from typing import List


DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 75


def _tail_by_characters(text: str, max_characters: int) -> str:
    """
    Return a whitespace-aware suffix of text whose length does not
    exceed max_characters.
    """

    if max_characters <= 0:
        return ""

    words = text.split()

    if not words:
        return ""

    selected = []
    total_length = 0

    for word in reversed(words):
        additional_length = len(word)

        if selected:
            additional_length += 1

        if total_length + additional_length > max_characters:
            break

        selected.append(word)
        total_length += additional_length

    return " ".join(reversed(selected))


def _recursive_split(
    text: str,
    chunk_size: int,
    separators: List[str],
    separator_index: int = 0,
) -> List[str]:
    """
    Recursively split text using increasingly fine-grained separators.
    """

    text = text.strip()

    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    if separator_index >= len(separators):
        step = chunk_size

        return [
            text[i:i + chunk_size].strip()
            for i in range(0, len(text), step)
            if text[i:i + chunk_size].strip()
        ]

    separator = separators[separator_index]

    if separator == "":
        return _recursive_split(
            text,
            chunk_size,
            separators,
            separator_index + 1,
        )

    parts = [part.strip() for part in text.split(separator) if part.strip()]

    if len(parts) <= 1:
        return _recursive_split(
            text,
            chunk_size,
            separators,
            separator_index + 1,
        )

    chunks = []
    current = ""

    for part in parts:

        candidate = (
            part
            if not current
            else current + separator + part
        )

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())
            current = ""

        if len(part) > chunk_size:
            smaller_chunks = _recursive_split(
                part,
                chunk_size,
                separators,
                separator_index + 1,
            )

            chunks.extend(smaller_chunks)
        else:
            current = part

    if current:
        chunks.append(current.strip())

    return chunks


def recursive_split(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """
    Recursively split text into overlapping chunks.

    Parameters
    ----------
    text:
        Input document text.

    chunk_size:
        Maximum size of a final chunk in characters.

    chunk_overlap:
        Approximate number of overlapping characters between
        neighboring chunks.

    Returns
    -------
    List[str]
        A list of cleaned text chunks.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    text = text.strip()

    # Natural-to-fine-grained splitting hierarchy.
    separators = [
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ]

    # Reserve space for overlap so the final chunk remains bounded
    # by chunk_size.
    content_size = chunk_size - chunk_overlap

    base_chunks = _recursive_split(
        text,
        content_size,
        separators,
    )

    if not base_chunks:
        return []

    if chunk_overlap == 0:
        return base_chunks

    final_chunks = [base_chunks[0]]

    for index in range(1, len(base_chunks)):
        previous_chunk = base_chunks[index - 1]
        current_chunk = base_chunks[index]

        overlap = _tail_by_characters(
            previous_chunk,
            chunk_overlap,
        )

        if not overlap:
            final_chunks.append(current_chunk)
            continue

        combined = f"{overlap} {current_chunk}".strip()

        # Safety guarantee.
        if len(combined) > chunk_size:
            combined = combined[-chunk_size:]

        final_chunks.append(combined)

    return final_chunks