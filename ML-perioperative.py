"""围手术期机器学习建模脚本。

构建 6 种模型：GLM、RF、SVM、NNET、GBM、XGBoost。
- 使用 AUC、准确度、灵敏度、特异度、F1 评估模型；
- AUC 为 100 次重复 10 折交叉验证均值；
- 采用 Bootstrap(1000 次) 计算测试集 AUC 的置信区间；
- 所有模型进行 SHAP 解释；
- 所有长耗时流程均增加进度条。
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


DATA_PATH = Path("data1.csv")
FEATURE_JSON_CANDIDATES = [Path("mul_logistic_selected_features.json"), Path("lasso_results/selected_features.json")]
OUTPUT_DIR = Path("ml_results")
OUTPUT_DIR.mkdir(exist_ok=True)
TARGET = "pulmonary_infection"
TEST_SIZE = 0.3
RANDOM_STATE = 42
N_REPEATS = 100
N_SPLITS = 10
N_BOOTSTRAP = 1000


def load_feature_config() -> Tuple[str, list[str]]:
    for path in FEATURE_JSON_CANDIDATES:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            features = payload.get("features") or payload.get("train_features") or []
            if features:
                return payload.get("target", TARGET), features
    raise FileNotFoundError("未找到可用特征配置，请先运行 lasso.py 和 mul_logistic.py")


def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "auc": roc_auc_score(y_true, y_prob),
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "sensitivity": tp / (tp + fn) if (tp + fn) else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, y_pred),
    }


def bootstrap_auc_ci(model, X_train, y_train, X_test, y_test, n_bootstrap: int = N_BOOTSTRAP) -> Tuple[float, float]:
    auc_scores = []
    rng = np.random.default_rng(RANDOM_STATE)
    for _ in tqdm(range(n_bootstrap), desc="Bootstrap 内部验证", leave=False):
        idx = rng.integers(0, len(X_train), len(X_train))
        X_boot = X_train.iloc[idx]
        y_boot = y_train.iloc[idx]
        boot_model = clone(model)
        boot_model.fit(X_boot, y_boot)
        y_prob = boot_model.predict_proba(X_test)[:, 1]
        auc_scores.append(roc_auc_score(y_test, y_prob))
    return float(np.percentile(auc_scores, 2.5)), float(np.percentile(auc_scores, 97.5))


def save_shap_outputs(model, model_name: str, X_background: pd.DataFrame, X_explain: pd.DataFrame) -> None:
    try:
        estimator = model.named_steps.get("model", model) if isinstance(model, Pipeline) else model
        transformed_bg = model[:-1].transform(X_background) if isinstance(model, Pipeline) and len(model.steps) > 1 else X_background
        transformed_exp = model[:-1].transform(X_explain) if isinstance(model, Pipeline) and len(model.steps) > 1 else X_explain

        feature_names = list(X_background.columns)
        transformed_bg = pd.DataFrame(transformed_bg, columns=feature_names)
        transformed_exp = pd.DataFrame(transformed_exp, columns=feature_names)

        if model_name in {"RF", "GBM", "XGBoost"}:
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(transformed_exp)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        elif model_name == "GLM":
            explainer = shap.LinearExplainer(estimator, transformed_bg)
            shap_values = explainer.shap_values(transformed_exp)
        else:
            explainer = shap.KernelExplainer(estimator.predict_proba, transformed_bg)
            shap_values = explainer.shap_values(transformed_exp, nsamples=100)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

        plt.figure()
        shap.summary_plot(shap_values, transformed_exp, show=False)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"shap_summary_{model_name}.png", dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure()
        shap.summary_plot(shap_values, transformed_exp, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"shap_bar_{model_name}.png", dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        print(f"[WARN] {model_name} 的 SHAP 解释失败：{exc}")


def build_models() -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {
        "GLM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, solver="liblinear", random_state=RANDOM_STATE)),
        ]),
        "RF": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, class_weight="balanced")),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "NNET": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=3000, random_state=RANDOM_STATE)),
        ]),
        "GBM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBClassifier(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
            )),
        ])
    else:
        print("[WARN] 当前环境缺少 xgboost，XGBoost 模型将被跳过。")
    return models


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError("未找到 data1.csv，请先运行 data_clean1.py")

    target, features = load_feature_config()
    data = pd.read_csv(DATA_PATH)

    X = data[features].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    y = pd.to_numeric(data[target], errors="coerce").fillna(0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models = build_models()
    performance_rows = []

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)

    for model_name, model in tqdm(models.items(), desc="训练模型总进度"):
        print(f"\n[INFO] 正在训练模型：{model_name}")

        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        )
        mean_cv_auc = float(np.mean(cv_scores))

        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = calculate_metrics(y_test.to_numpy(), y_prob)
        auc_ci_low, auc_ci_high = bootstrap_auc_ci(model, X_train, y_train, X_test, y_test)

        performance_rows.append(
            {
                "model": model_name,
                "cv_auc_mean": mean_cv_auc,
                "test_auc": metrics["auc"],
                "auc_ci_low": auc_ci_low,
                "auc_ci_high": auc_ci_high,
                "accuracy": metrics["accuracy"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "f1": metrics["f1"],
            }
        )

        bg = X_train.sample(min(100, len(X_train)), random_state=RANDOM_STATE)
        ex = X_test.sample(min(50, len(X_test)), random_state=RANDOM_STATE)
        save_shap_outputs(model, model_name, bg, ex)

    performance_df = pd.DataFrame(performance_rows).sort_values("cv_auc_mean", ascending=False)
    performance_df.to_csv(OUTPUT_DIR / "model_performance_summary.csv", index=False, encoding="utf-8-sig")
    print("\n[INFO] 模型评估结果：")
    print(performance_df)


if __name__ == "__main__":
    main()
