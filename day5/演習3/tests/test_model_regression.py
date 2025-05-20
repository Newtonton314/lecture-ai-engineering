import os, pickle, time, pytest
import numpy as np
from sklearn.metrics import accuracy_score

import day5.演習3.src.train_model as tm  # 既存の学習モジュールを想定

# ────────────────────────────────────────────────
# 1. 現行 PR のモデルと評価データを取得
# ────────────────────────────────────────────────
CUR_MODEL, X_test, y_test = tm.train_model()  # 既存 fixture を関数化して再利用
cur_pred = CUR_MODEL.predict(X_test)
cur_acc = accuracy_score(y_test, cur_pred)

start = time.time()
CUR_MODEL.predict(X_test)
cur_inf_time = time.time() - start


# ────────────────────────────────────────────────
# 2. ベースライン（main）モデルをロード
# ────────────────────────────────────────────────
HAS_BASELINE = os.getenv("HAS_BASELINE", "false") == "true"
BASELINE_MODEL = os.getenv("BASELINE_MODEL")

if not HAS_BASELINE:
    pytest.skip("main ブランチにベースラインモデルが無いのでスキップ")

with open(BASELINE_MODEL, "rb") as f:
    BASE_MODEL = pickle.load(f)

base_pred = BASE_MODEL.predict(X_test)
base_acc = accuracy_score(y_test, base_pred)

start = time.time()
BASE_MODEL.predict(X_test)
base_inf_time = time.time() - start


# ────────────────────────────────────────────────
# 3. 性能劣化がないか検証
# ────────────────────────────────────────────────
#   ▼許容幅は ±0.01 ＝ 1pt 以内の低下なら OK とする例
def test_accuracy_not_degraded():
    assert (
        cur_acc + 0.01 >= base_acc
    ), f"精度が劣化しています: current={cur_acc:.3f} < baseline={base_acc:.3f}"

#   ▼推論時間は +20% 以内なら OK
def test_inference_time_not_slower():
    assert cur_inf_time <= base_inf_time * 1.2, (
        f"推論時間が遅くなりました: current={cur_inf_time:.4f}s "
        f"> baseline={base_inf_time:.4f}s"
    )