import os
import pickle
import time
import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


HERE = os.path.dirname(__file__)
DATA_PATH = os.path.abspath(os.path.join(HERE, "../data/Titanic.csv"))

NUMERIC_FEATURES = ["Age", "Pclass", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Sex", "Embarked"]


def load_data():
    """Titanic データを DataFrame で返す（必要ならダウンロード）"""
    if not os.path.exists(DATA_PATH):
        from sklearn.datasets import fetch_openml

        titanic = fetch_openml("titanic", version=1, as_frame=True)
        df = titanic.data
        df["Survived"] = titanic.target
        df = df[
            ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]
        ]
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    return pd.read_csv(DATA_PATH)


def build_model():
    """前処理 + RandomForest を組み合わせた Pipeline を返す"""
    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )
    return model


def train_current_model():
    """現行 PR のコードでモデルを学習し、評価用データを返す"""
    df = load_data()
    X = df.drop("Survived", axis=1)
    y = df["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = build_model()
    model.fit(X_train, y_train)
    return model, X_test, y_test



CUR_MODEL, X_test, y_test = train_current_model()
cur_pred = CUR_MODEL.predict(X_test)
cur_acc = accuracy_score(y_test, cur_pred)

t0 = time.time()
CUR_MODEL.predict(X_test)
cur_inf_time = time.time() - t0


HAS_BASELINE = os.getenv("HAS_BASELINE", "false").lower() == "true"
BASELINE_MODEL = os.getenv("BASELINE_MODEL")

if not HAS_BASELINE or not BASELINE_MODEL or not os.path.exists(BASELINE_MODEL):
    pytest.skip("main ブランチのベースラインモデルが無いのでスキップ")

with open(BASELINE_MODEL, "rb") as f:
    BASE_MODEL = pickle.load(f)

base_pred = BASE_MODEL.predict(X_test)
base_acc = accuracy_score(y_test, base_pred)

t0 = time.time()
BASE_MODEL.predict(X_test)
base_inf_time = time.time() - t0



def test_accuracy_not_degraded():
    """精度が 1pt 以上落ちていないか"""
    assert cur_acc + 0.01 >= base_acc, (
        f"精度が劣化しています: current={cur_acc:.3f} < baseline={base_acc:.3f}"
    )


def test_inference_time_not_slower():
    """推論時間が 20% 以上遅くなっていないか"""
    assert cur_inf_time <= base_inf_time * 1.2, (
        f"推論時間が遅くなりました: current={cur_inf_time:.4f}s "
        f"> baseline={base_inf_time:.4f}s"
    )