"""基于分层抽样 + LASSO 的特征选择脚本。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


DATA_PATH = Path("data1.csv")
TARGET = "pulmonary_infection"
TEST_SIZE = 0.3
RANDOM_STATE = 42
CORR_THRESHOLD = 0.6

OUTPUT_DIR = Path("lasso_results")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError("未找到 data1.csv，请先运行 data_clean1.py")
    return pd.read_csv(DATA_PATH)


def select_numeric_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if TARGET not in df.columns:
        raise KeyError(f"数据中缺少因变量列：{TARGET}")
    y = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)
    X = df.drop(columns=[TARGET])
    numeric_columns = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
    X = X[numeric_columns].copy()
    X = X.fillna(X.median(numeric_only=True))
    return X, y


def stratified_split(X: pd.DataFrame, y: pd.Series):
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def fit_lasso(X_train: pd.DataFrame, y_train: pd.Series) -> pd.Series:
    model = LogisticRegressionCV(
        Cs=20,
        cv=10,
        penalty="l1",
        solver="liblinear",
        scoring="roc_auc",
        max_iter=5000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    model.fit(X_train, y_train)
    coef = pd.Series(model.coef_.ravel(), index=X_train.columns, name="lasso_coefficient")
    coef.sort_values(key=np.abs, ascending=False).to_csv(OUTPUT_DIR / "lasso_all_coefficients.csv", encoding="utf-8-sig")
    return coef


def correlation_filter(X_train: pd.DataFrame, selected_coef: pd.Series) -> Tuple[List[str], pd.DataFrame]:
    selected_features = selected_coef[selected_coef != 0].index.tolist()
    if not selected_features:
        raise ValueError("LASSO 未筛选出非零系数变量。")

    selected_train = X_train[selected_features].copy()
    corr = selected_train.corr(method="spearman")

    plt.figure(figsize=(max(8, len(selected_features) * 0.6), max(6, len(selected_features) * 0.6)))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, annot=False, square=True)
    plt.title("LASSO筛选变量的Spearman相关性热力图")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lasso_spearman_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    to_remove = set()
    cols = corr.columns.tolist()
    for i in tqdm(range(len(cols)), desc="相关性筛选"):
        for j in range(i + 1, len(cols)):
            val = abs(corr.iloc[i, j])
            if val > CORR_THRESHOLD:
                left, right = cols[i], cols[j]
                if abs(selected_coef[left]) <= abs(selected_coef[right]):
                    to_remove.add(left)
                else:
                    to_remove.add(right)

    final_features = [feat for feat in selected_features if feat not in to_remove]
    removal_df = pd.DataFrame(
        [{"removed_feature": feat, "lasso_coefficient": float(selected_coef[feat])} for feat in sorted(to_remove)]
    )
    removal_df.to_csv(OUTPUT_DIR / "correlation_removed_features.csv", index=False, encoding="utf-8-sig")
    return final_features, corr


def main() -> None:
    print("[INFO] 加载数据...")
    data = load_data()
    X, y = select_numeric_features(data)

    print("[INFO] 执行按肺部感染分层的 7:3 训练/测试集划分...")
    X_train, X_test, y_train, y_test = stratified_split(X, y)

    split_summary = pd.DataFrame(
        {
            "dataset": ["original", "train", "test"],
            "sample_size": [len(y), len(y_train), len(y_test)],
            "positive_rate": [y.mean(), y_train.mean(), y_test.mean()],
        }
    )
    split_summary.to_csv(OUTPUT_DIR / "split_summary.csv", index=False, encoding="utf-8-sig")

    print("[INFO] 在训练集上执行 LASSO 特征筛选...")
    coef = fit_lasso(X_train, y_train)

    selected = coef[coef != 0].sort_values(key=np.abs, ascending=False)
    selected.to_csv(OUTPUT_DIR / "lasso_selected_coefficients.csv", encoding="utf-8-sig")
    print("[INFO] LASSO 非零系数变量：")
    print(selected)

    print("[INFO] 对 LASSO 筛选特征做 Spearman 相关性分析...")
    final_features, corr = correlation_filter(X_train, coef)

    final_df = pd.DataFrame(
        {
            "feature": final_features,
            "lasso_coefficient": [float(coef[f]) for f in final_features],
        }
    ).sort_values("lasso_coefficient", key=np.abs, ascending=False)
    final_df.to_csv(OUTPUT_DIR / "final_selected_features.csv", index=False, encoding="utf-8-sig")

    payload: Dict[str, object] = {
        "target": TARGET,
        "train_features": final_features,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }
    (OUTPUT_DIR / "selected_features.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[INFO] 最终保留变量：")
    print(final_df)
    print(f"[INFO] 结果目录：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
