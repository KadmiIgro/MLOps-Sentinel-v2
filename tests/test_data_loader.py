import pytest
import yaml
from src.data_loader import TextDataLoader

@pytest.fixture
def config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def test_load_data(config):
    loader = TextDataLoader(config)
    df = loader.load_data()
    # Проверяем, что данные загружены (не важно сколько)
    assert len(df) > 0
    assert 'text' in df.columns
    assert 'target' in df.columns

def test_split_data(config):
    loader = TextDataLoader(config)
    loader.load_data()
    X_train, X_test, y_train, y_test = loader.split_data()
    # Проверяем, что разбиение работает
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) + len(X_test) == len(loader.df)

def test_get_data(config):
    loader = TextDataLoader(config)
    X_train, X_test, y_train, y_test = loader.get_data()
    assert X_train is not None
    assert y_train is not None