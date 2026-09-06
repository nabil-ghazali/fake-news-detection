import pandas as pd

from data_handler.text_cleaning import text_cleaning


def create_mock_dataset():
    data = {
        "title": ["Title 1", "Title 2", "Title 3"],
        "text": [
            "Fake News : Aliens landing on Earth !",
            "Donald Trump is a first super genius !",
            "Alibaba and the 40 thieves !",
        ],
        "label": ["fake", "fake", "true"],
    }
    return pd.DataFrame(data)


def test_mock_dataset_structure():
    df = create_mock_dataset()
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"title", "text", "label"}
    assert len(df) == 3


def test_mock_dataset_labels_are_valid():
    df = create_mock_dataset()
    assert set(df["label"].unique()) <= {"fake", "true"}


def test_text_cleaning_removes_html_and_urls():
    raw = 'Visit <a href="http://example.com">this link</a> &amp; more at www.test.org !'
    cleaned = text_cleaning(raw)
    assert "<" not in cleaned and ">" not in cleaned
    assert "http" not in cleaned and "www" not in cleaned
    assert "&amp;" not in cleaned


def test_text_cleaning_lowercases_and_strips_specials():
    cleaned = text_cleaning("  Hello,  WORLD!!!  #Fake @News  ")
    assert cleaned == "hello world fake news"


def test_text_cleaning_normalizes_whitespace():
    assert text_cleaning("a\n\nb\t c   d") == "a b c d"
