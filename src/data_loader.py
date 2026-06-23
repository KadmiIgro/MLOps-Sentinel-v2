import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


class TextDataLoader:
    def __init__(self, config):
        self.config = config
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        """Загрузка данных из CSV"""
        path = Path(self.config["data"]["path"])
        self.df = pd.read_csv(path)

        text_col = self.config["data"]["text_column"]
        target_col = self.config["data"]["target_column"]

        # Проверка наличия колонок
        if text_col not in self.df.columns:
            raise ValueError(
                f"Колонка '{text_col}' не найдена. Доступны: {self.df.columns}"
            )
        if target_col not in self.df.columns:
            raise ValueError(
                f"Колонка '{target_col}' не найдена. Доступны: {self.df.columns}"
            )

        print(f"✅ Загружено {len(self.df)} строк")
        print(f"📊 Колонки: {list(self.df.columns)}")
        print(
            f"🎯 Распределение классов:\n{self.df[target_col].value_counts()}"
        )

        return self.df

    def split_data(self):
        """Разбиение на train/test"""
        if self.df is None:
            self.load_data()

        text_col = self.config["data"]["text_column"]
        target_col = self.config["data"]["target_column"]

        X = self.df[text_col]
        y = self.df[target_col]

        test_size = self.config["data"].get("test_size", 0.2)
        random_state = self.config["data"].get("random_state", 42)

        self.X_train, self.X_test, self.y_train, self.y_test = (
            train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )
        )

        print(f"✅ Train: {len(self.X_train)} строк")
        print(f"✅ Test: {len(self.X_test)} строк")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_data(self):
        """Получить готовые данные для обучения"""
        if self.X_train is None:
            self.split_data()
        return self.X_train, self.X_test, self.y_train, self.y_test
