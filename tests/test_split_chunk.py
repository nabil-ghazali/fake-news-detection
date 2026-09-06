import pytest

from function_chunk.split_chunk import chunk_text


@pytest.fixture
def sample_text():
    """Texte reutilisable pour chaque test (12 mots)."""
    return "one two three four five six seven eight nine ten eleven twelve"


def test_chunk_basic(sample_text):
    # chunk_text renvoie une liste de CHAINES (les mots du chunk joints par un espace).
    result = chunk_text(sample_text, start=0, step=3, overlap=4)
    # idx = 0 -> mots[0:4] ; idx = 3 -> mots[3:7] ; idx = 6 -> mots[6:10] ; idx = 9 -> mots[9:13]
    expected = [
        "one two three four",
        "four five six seven",
        "seven eight nine ten",
        "ten eleven twelve",
    ]
    assert result == expected


def test_chunk_offset_and_step(sample_text):
    result = chunk_text(sample_text, start=2, step=4, overlap=3)
    # idx = 2 -> mots[2:5] ; idx = 6 -> mots[6:9] ; idx = 10 -> mots[10:13] (2 mots restants)
    expected = ["three four five", "seven eight nine", "eleven twelve"]
    assert result == expected


def test_chunk_last_chunk_is_partial(sample_text):
    # Le dernier chunk peut contenir moins de `overlap` mots.
    result = chunk_text(sample_text, start=0, step=12, overlap=5)
    assert result == ["one two three four five"]


def test_chunk_empty_text():
    assert chunk_text("", start=0, step=3, overlap=4) == []


def test_chunk_start_defaults_to_zero(sample_text):
    assert chunk_text(sample_text, step=3, overlap=4) == chunk_text(
        sample_text, start=0, step=3, overlap=4
    )
