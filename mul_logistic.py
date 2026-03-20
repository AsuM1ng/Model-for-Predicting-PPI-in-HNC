"""基于 LASSO 后特征的多因素 Logistic 回归。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm.auto import tqdm


DATA_PATH = Path("data1.csv")
LASSO_JSON = Path("lasso_results/selected_features.json")
OUTPUT_PATH = Path("multivariate_logistic_results.csv")
OUTPUT_JSON = Path("mul_logistic_selected_features.json")
TARGET = "pulmonary_infection"
P_THRESHOLD = 0.05


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError("未找到 data1.csv，请先运行 data_clean1.py")
    if not LASSO_JSON.exists():
        raise FileNotFoundError("未找到 lasso_results/selected_features.json，请先运行 lasso.py")

    data = pd.read_csv(DATA_PATH)
    config = json.loads(LASSO_JSON.read_text(encoding="utf-8"))
    selected_features = config.get("train_features", [])

    if TARGET not in data.columns:
        raise KeyError(f"数据中缺少因变量列：{TARGET}")
    if not selected_features:
        raise ValueError("LASSO 输出中没有可用于 Logistic 的变量。")

    y = pd.to_numeric(data[TARGET], errors="coerce").fillna(0).astype(int)
    X = data[selected_features].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    X = sm.add_constant(X, has_constant="add")

    print("[INFO] 开始多因素 Logistic 回归...")
    model = sm.Logit(y, X)
    result = model.fit(disp=False, maxiter=200)

    rows = []
    conf_int = result.conf_int()
    for column in tqdm(X.columns, desc="整理回归结果"):
        if column == "const":
            continue
        coef = float(result.params[column])
        p_value = float(result.pvalues[column])
        ci_low, ci_high = conf_int.loc[column]
        rows.append(
            {
                "feature": column,
                "coef": coef,
                "odds_ratio": float(np.exp(coef)),
                "p_value": p_value,
                "or_ci_low": float(np.exp(ci_low)),
                "or_ci_high": float(np.exp(ci_high)),
                "is_independent_predictor": int(p_value < P_THRESHOLD),
            }
        )

    results_df = pd.DataFrame(rows).sort_values(["is_independent_predictor", "p_value"], ascending=[False, True])
    results_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    independent_features = results_df.loc[results_df["is_independent_predictor"] == 1, "feature"].tolist()
    OUTPUT_JSON.write_text(
        json.dumps({"target": TARGET, "features": independent_features}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n[INFO] 多因素 Logistic 结果：")
    print(results_df)
    print("\n[INFO] 独立预测因素（P < 0.05）：")
    print(independent_features)


if __name__ == "__main__":
    main()
