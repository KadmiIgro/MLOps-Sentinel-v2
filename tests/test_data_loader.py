import pytest
import yaml

from src.data_loader import TextDataLoader


@pytest.fixture
def config():
    with open("config.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def test_load_data(config):
    loader = TextDataLoader(config)
    df = loader.load_data()

    assert len(df) > 0
    assert "text" in df.columns
    assert "target" in df.columns


def test_split_data(config):
    loader = TextDataLoader(config)
    loader.load_data()

    x_train, x_test, _, _ = loader.split_data()

    assert len(x_train) > 0
    assert len(x_test) > 0
    assert len(x_train) + len(x_test) == len(loader.df)


def test_get_data(config):
    loader = TextDataLoader(config)
    x_train, _, y_train, _ = loader.get_data()

    assert x_train is not None
    assert y_train is not None
