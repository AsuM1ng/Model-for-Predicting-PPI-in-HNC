from sklearn.exceptions import ConvergenceWarning

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV

import xgboost as xgb
import shap

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# =================== 数据 ===================
if not os.path.exists("data1.csv"):
    raise FileNotFoundError("未找到 data1.csv")

data = pd.read_csv("data1.csv")

features = [
    'pre_op_palb', 'pre_op_alb', 'pre_op_hgb', 'pre_op_oropharyngeal_swab',
    'pre_op_antibiotics', 'flap_type',
    'post_op_pathology', 'ptnm_stage', 'multiple_primary',
    'anastomotic_leak'
]

X = data[features].copy()
y = data['wound_infection'].astype(int)

X['flap_type'] = LabelEncoder().fit_transform(X['flap_type'].astype(str))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


# =================== 时间戳文件夹 ===================
time_tag = datetime.now().strftime("%H-%M")
FIG_DIR = f"results_figures_{time_tag}"
os.makedirs(FIG_DIR, exist_ok=True)

# =================== 工具函数 ===================
def calculate_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return (
        roc_auc_score(y_true, y_prob),
        tp / (tp + fn) if (tp + fn) else 0,
        tn / (tn + fp) if (tn + fp) else 0,
        (tp + tn) / (tp + tn + fp + fn),
        f1_score(y_true, y_pred)
    )


def calculate_net_benefit(y_true, y_prob, thresholds):
    y_true = np.array(y_true)
    net_benefits = []
    n = len(y_true)
    for pt in thresholds:
        y_pred = (y_prob >= pt).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        nb = (tp / n) - (fp / n) * (pt / (1 - pt)) if 0 < pt < 1 else 0
        net_benefits.append(nb)
    return net_benefits


def save_shap_summary(shap_values, X, model_name):
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/SHAP_{model_name}_summary.png", dpi=300)
    plt.close()


def save_shap_bar(shap_values, X, model_name):
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/SHAP_{model_name}_bar.png", dpi=300)
    plt.close()


# =================== 中文显示 ===================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False



# =================== 模型 ===================
models = {
    'GLM': LogisticRegression(class_weight='balanced', max_iter=1000),
    'RF': RandomForestClassifier(class_weight='balanced', random_state=42),
    'SVM': CalibratedClassifierCV(
        LinearSVC(class_weight='balanced', dual=False, max_iter=10000),
        cv=5
    ),
    'NNET': MLPClassifier(max_iter=500, random_state=42),
    'NB': GaussianNB()
}

param_grids = {
    'GLM': {'classifier__C': [1]},
    'RF': {'classifier__n_estimators': [100], 'classifier__max_depth': [20]},
    'SVM': {'classifier__estimator__C': [0.1, 1]},
    'NNET': {'classifier__hidden_layer_sizes': [(50,), (100,)]},
    'NB': {}
}

results = {}
model_probs = {}
CV = 5

# =================== 训练 + SHAP ===================
for name, model in models.items():
    print(f"\n>>> 训练模型: {name}")

    pipe = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])

    grid = GridSearchCV(pipe, param_grids[name], cv=CV, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_prob = best_model.predict_proba(X_test)[:, 1]
    model_probs[name] = y_prob

    auc, sens, spec, acc, f1 = calculate_metrics(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    results[name] = {
        'AUC': auc,
        'Sensitivity': sens,
        'Specificity': spec,
        'Accuracy': acc,
        'F1': f1,
        'fpr': fpr,
        'tpr': tpr
    }

    clf = best_model.named_steps['classifier']
    X_bg = X_train.sample(min(100, len(X_train)), random_state=42)
    X_exp = X_test.sample(min(50, len(X_test)), random_state=42)

    try:
        if name == 'RF':
            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(X_exp)[1]
        elif name == 'GLM':
            explainer = shap.LinearExplainer(clf, X_bg)
            sv = explainer.shap_values(X_exp)
        else:
            explainer = shap.KernelExplainer(
                lambda x: clf.predict_proba(pd.DataFrame(x, columns=X.columns))[:, 1],
                X_bg
            )
            sv = explainer.shap_values(X_exp, nsamples=100)

        save_shap_summary(sv, X_exp, name)
        save_shap_bar(sv, X_exp, name)

    except Exception as e:
        print(f"{name} SHAP 失败: {e}")

# =================== XGBoost ===================
print("\n>>> 训练模型: XGBoost")
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos

xgb_model = xgb.XGBClassifier(
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

grid = GridSearchCV(
    xgb_model,
    {'n_estimators': [100], 'max_depth': [3, 5], 'learning_rate': [0.1]},
    cv=CV,
    scoring='roc_auc',
    n_jobs=-1
)
grid.fit(X_train, y_train)

clf = grid.best_estimator_
y_prob = clf.predict_proba(X_test)[:, 1]
model_probs['GBM'] = y_prob

auc, sens, spec, acc, f1 = calculate_metrics(y_test, y_prob)
fpr, tpr, _ = roc_curve(y_test, y_prob)

results['GBM'] = {
    'AUC': auc,
    'Sensitivity': sens,
    'Specificity': spec,
    'Accuracy': acc,
    'F1': f1,
    'fpr': fpr,
    'tpr': tpr
}

explainer = shap.TreeExplainer(clf)
X_shap = X_test.sample(min(50, len(X_test)), random_state=42)
sv = explainer.shap_values(X_shap)
save_shap_summary(sv, X_shap, "GBM")
save_shap_bar(sv, X_shap, "GBM")

# =================== 指标表 ===================
df = pd.DataFrame(results).T[['AUC', 'Sensitivity', 'Specificity', 'Accuracy', 'F1']]
df.to_csv(f"{FIG_DIR}/model_metrics.csv", encoding="utf-8-sig")
print("\n模型性能指标：\n", df.round(4))

# =================== ROC ===================
# plt.figure(figsize=(8, 6))
# for k, v in results.items():
#     plt.plot(v['fpr'], v['tpr'], label=k)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlabel("1 - 特异性")
# plt.ylabel("敏感性")
# plt.title("ROC 曲线")
# plt.legend()
# plt.savefig(f"{FIG_DIR}/ROC_curves.png", dpi=300)
# # plt.show()
# =================== ROC（超圆角平滑版） ===================
plt.figure(figsize=(8, 6))

mean_fpr = np.linspace(0, 1, 3000)

for k, v in results.items():
    tpr_interp = np.interp(mean_fpr, v['fpr'], v['tpr'])

    # 第一次平滑
    window = 81
    kernel = np.ones(window) / window
    tpr_smooth = np.convolve(tpr_interp, kernel, mode='same')

    # 第二次平滑（让转弯更圆）
    tpr_smooth = np.convolve(tpr_smooth, kernel, mode='same')

    # ROC 物理约束
    tpr_smooth = np.maximum.accumulate(tpr_smooth)
    tpr_smooth = np.clip(tpr_smooth, 0, 1)

    plt.plot(
        mean_fpr,
        tpr_smooth,
        label=f"{k} "
    )

plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
plt.xlabel("1 - 特异性")
plt.ylabel("敏感性")
plt.title("ROC 曲线")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/ROC_curves_extra_round.png", dpi=300)
plt.close()

# =================== DCA ===================
thresholds = np.linspace(0.05, 0.6, 50)
plt.figure(figsize=(8, 6))
for name, probs in model_probs.items():
    plt.plot(thresholds, calculate_net_benefit(y_test, probs, thresholds), label=name)
plt.xlabel("阈值概率")
plt.ylabel("净受益")
plt.title("DCA 曲线")
plt.legend()
plt.savefig(f"{FIG_DIR}/DCA_curves.png", dpi=300)
# plt.show()
