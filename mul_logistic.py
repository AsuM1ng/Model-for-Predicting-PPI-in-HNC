import pandas as pd
import statsmodels.api as sm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# 加载数据
try:
    data = pd.read_csv(r"data1.csv")
except FileNotFoundError:
    print("错误：无法找到 'data1.csv' 文件，请检查文件路径！")
    exit()


# 分离自变量和因变量
try:

    selected_columns = ['gender', 'age', 'BMI', 'history_head_neck_surgery', 'coronary heart_disease', 'hypertension',
                   'peripheralvascular_disease', 'immune_disease', 'diabetes', 'hyperlipidemia', 'hormone_use',
                   'smoking_history', 'alcohol_history', 'pre_op_palb', 'pre_op_alb', 'pre_op_hgb',
                   'pre_op_oropharyngeal_swab', 'lesion_site', 'asa_score', 'pre_op_antibiotics', 'surgery_duration',
                   'intraop_transfusion', 'anastomosis_type', 'neck_dissection', 'pre_op_radiotherapy',
                   'pre_op_chemotherapy', 'pre_op_chemoradiotherapy', 'endoscopic_surgery', 'flap_type', 'tracheostomy',
                   'post_op_pathology', 'differentiation', 'ptnm_stage', 'multiple_primary', 'post_op_min_alb',
                   'post_op_palb', 'unplanned_reoperation', 'pulmonary_infection', 'anastomotic_leak',
                   'leak_confirm_day', 'fat_liquefaction', 'multidrug_resistant']
    #新的
    X = data[selected_columns]
    y = data['wound_infection']
except KeyError as e:
    print(f"错误：列 {e} 不存在，请检查列名！")
    exit()

# 确保 y 是二分类变量（0 和 1）
if y.dtype not in [np.int32, np.int64, np.float32, np.float64]:
    print("\n因变量 'Infection status' 非数值型，正在处理...")
    if y.nunique() == 2:
        y = pd.factorize(y)[0]
        print(f"因变量已转换为：{np.unique(y)}")
    else:
        print(f"错误：因变量 'Infection status' 有 {y.nunique()} 个唯一值，不是二分类！")
        exit()

# 处理 X 的非数值列
for column in X.columns:
    if X[column].dtype == 'object' or X[column].dtype.name == 'category':
        print(f"\n检测到非数值列 '{column}'，正在进行编码...")
        X[column] = pd.factorize(X[column])[0]
    elif not np.issubdtype(X[column].dtype, np.number):
        print(f"错误：列 '{column}' 包含非数值数据，请检查！")
        exit()

corr_matrix = X.corr(numeric_only=True)
# 设置热力图
plt.figure(figsize=(10, 8))  # 设置图形大小
sns.heatmap(
    corr_matrix,
    annot=False,  # 显示相关系数数值
    cmap='coolwarm',  # 颜色方案（红蓝渐变，0 为中性）
    vmin=-1, vmax=1,  # 颜色范围固定为 [-1, 1]
    center=0,  # 中心点为 0
    fmt='.2f',  # 保留两位小数
    square=True,  # 每个单元格为正方形
    # cbar_kws={'label': 'Correlation Coefficient'}  # 颜色条标签
)
# 设置坐标标签旋转45度
plt.xticks(rotation=45, ha='right')  # x轴标签旋转45度，右对齐
plt.tight_layout()                   # 自动调整布局，避免标签被裁剪
plt.savefig('correlation_heatmap2.png', dpi=300, bbox_inches='tight')
# 显示热力图
plt.show()
# 检查多重共线性
print("\n检查多重共线性（相关系数 > 0.60）...")
corr_matrix = X.corr(numeric_only=True).abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.6)]
if to_drop:
    print(f"警告：以下列存在高相关性，将移除：{to_drop}")
    X = X.drop(columns=to_drop)

# 添加常数项
X = sm.add_constant(X)

# 多因素 logistic 回归
print("\n=== 多因素 Logistic 回归分析 ===")
model = sm.Logit(y, X)
result = model.fit(disp=1, maxiter=200)  # disp=1 显示迭代过程
print("\n模型概要：")
print(result.summary())

# 提取结果
results = []
for column in X.columns:
    if column != 'const':
        coef = result.params[column]
        conf_int = result.conf_int().loc[column]
        p_value = result.pvalues[column]
        odds_ratio = np.exp(coef)
        results.append({
            '变量': column,
            '回归系数': round(coef, 4),
            '优势比 (OR)': round(odds_ratio, 4),
            'P值': round(p_value, 4),
            '95% 置信区间下限 (OR)': round(np.exp(conf_int[0]), 4),
            '95% 置信区间上限 (OR)': round(np.exp(conf_int[1]), 4)
        })

# 输出结果
results_df = pd.DataFrame(results)
print("\n多因素 Logistic 回归结果：")
print(results_df)

# 筛选独立危险因素
independent_factors = results_df[results_df['P值'] < 0.1]
print("\n=== 独立危险因素 (P < 0.05) ===")
print(independent_factors)
#
# # 保存结果
# results_df.to_csv('multivariate_logistic_afterlasso.csv', index=False, encoding='utf-8')
# print("\n结果已保存至 'multivariate_logistic_afterlasso.csv'")