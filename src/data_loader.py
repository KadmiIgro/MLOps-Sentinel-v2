from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


class TextDataLoader:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self) -> pd.DataFrame:
        """Load training data from CSV."""
        path = Path(self.config["data"]["path"])
        self.df = pd.read_csv(path)

        text_col = self.config["data"]["text_column"]
        target_col = self.config["data"]["target_column"]

        if text_col not in self.df.columns:
            raise ValueError(
                f"Column '{text_col}' was not found. Available: {self.df.columns}"
            )
        if target_col not in self.df.columns:
            raise ValueError(
                f"Column '{target_col}' was not found. Available: {self.df.columns}"
            )

        print(f"Loaded {len(self.df)} rows")
        print(f"Columns: {list(self.df.columns)}")
        print(f"Class distribution:\n{self.df[target_col].value_counts()}")
        return self.df

    def split_data(self):
        """Split data into train and test parts."""
        if self.df is None:
            self.load_data()

        text_col = self.config["data"]["text_column"]
        target_col = self.config["data"]["target_column"]

        x_values = self.df[text_col]
        y_values = self.df[target_col]
        test_size = self.config["data"].get("test_size", 0.2)
        random_state = self.config["data"].get("random_state", 42)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            x_values,
            y_values,
            test_size=test_size,
            random_state=random_state,
            stratify=y_values,
        )

        print(f"Train rows: {len(self.X_train)}")
        print(f"Test rows: {len(self.X_test)}")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_data(self):
        """Return train/test data for model training."""
        if self.X_train is None:
            self.split_data()
        return self.X_train, self.X_test, self.y_train, self.y_test
