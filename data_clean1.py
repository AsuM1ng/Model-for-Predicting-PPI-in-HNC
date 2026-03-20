# 本代码用于处理原始数据
# 实现以下功能：
# 1. 将中文标题转换为英文，保留最核心的信息（如性别（1=男，2=女）转换为Sex）
# 2. 缺失值处理：连续变量用中位数填充，分类变量用众数填充（术前前白蛋白PALB、术前白蛋白ALB、术前血红蛋白HGB中的0，最新版pTNM中的5等也为缺失值）
# 3. 连续变量标准化（Z-score）
# 4. 分类变量编码：二分类变量用0/1编码，多分类变量用序列编码.对于分类种类较多的变量，应为新分类的出现作好编码准备
# 5. 能够读取.xlsx文件
# 6. 详细的代码注释，标注每一部分功能

import pandas as pd

read_path = "data0.csv"
data = pd.read_csv(read_path, encoding="utf-8-sig", header=0, low_memory=False)

def get_unique_values(column):
    """返回列中唯一值的列表"""
    return list(column.unique())

# 性别转换
# data['性别'].replace({'男': 1, '女': 0}, inplace=True)

# 愈合等级转换，有缺失值
#data['主要手术愈合等级'].replace({'II/甲': 1, 'II/乙': 2, 'II/丙': 3, 'II/其他': 4, 'III/甲': 5, 'III/乙': 6, 'III/其他': 7}, inplace=True)

# 年龄归一化
def agechange(val):
    try:
        val = float(val)

        # 手动输入
        min_val = 8
        max_val = 89

        if val < min_val:
            val = min_val
        elif val > max_val:
            val = max_val

        # 归一化公式: (x - min) / (max - min)
        normalized = (val - min_val) / (max_val - min_val)
        return normalized

    except (ValueError, TypeError):
        return None  # 处理非数值情况

data['年龄'] = data['年龄'].apply(agechange)

# bmi归一化
def BMIchange(val):
    try:
        val = float(val)

        # 手动输入
        min_val = 11.9
        max_val = 41.4

        if val < min_val:
            val = min_val
        elif val > max_val:
            val = max_val

        # 归一化公式: (x - min) / (max - min)
        normalized = (val - min_val) / (max_val - min_val)
        return normalized

    except (ValueError, TypeError):
        return None  # 处理非数值情况

data['BMI'] = data['BMI'].apply(BMIchange)


# 术前前白蛋白PALB归一化
def classify_pre_palb(val):
    try:
        val = float(val)

        # 手动输入
        min_val = 0
        max_val = 39.6

        if val < min_val:
            val = min_val
        elif val > max_val:
            val = max_val

        # 归一化公式: (x - min) / (max - min)
        normalized = (val - min_val) / (max_val - min_val)
        return normalized

    except (ValueError, TypeError):
        return None  # 处理非数值情况

data['术前前白蛋白PALB'] = data['术前前白蛋白PALB'].apply(classify_pre_palb)

# 术前白蛋白ALB归一化
def classify_pre_alb(val):
    try:
        val = float(val)

        # 手动输入
        min_val = 0
        max_val = 56.9

        if val < min_val:
            val = min_val
        elif val > max_val:
            val = max_val

        # 归一化公式: (x - min) / (max - min)
        normalized = (val - min_val) / (max_val - min_val)
        return normalized

    except (ValueError, TypeError):
        return None  # 处理非数值情况

data['术前白蛋白ALB'] = data['术前白蛋白ALB'].apply(classify_pre_alb)

# 术前血红蛋白HGB归一化
def classify_pre_hgb(val):
    try:
        val = float(val)

        # 手动输入
        min_val = 0
        max_val = 196

        if val < min_val:
            val = min_val
        elif val > max_val:
            val = max_val

        # 归一化公式: (x - min) / (max - min)
        normalized = (val - min_val) / (max_val - min_val)
        return normalized

    except (ValueError, TypeError):
        return None  # 处理非数值情况

data['术前血红蛋白HGB'] = data['术前血红蛋白HGB'].apply(classify_pre_hgb)

# ASA评分
data['ASA评分'].replace({'Ⅰ': 1, 'Ⅱ': 2, 'Ⅲ': 3, 'Ⅳ': 4}, inplace=True)

# 手术时长归一化
def classify_surgery_duration(val):
    try:
        val = float(val)

        # 手动输入
        min_val = 0
        max_val = 872

        if val < min_val:
            val = min_val
        elif val > max_val:
            val = max_val

        # 归一化公式: (x - min) / (max - min)
        normalized = (val - min_val) / (max_val - min_val)
        return normalized

    except (ValueError, TypeError):
        return None  # 处理非数值情况

data['手术时长（min）'] = data['手术时长（min）'].apply(classify_surgery_duration)

# 术后0-3天白蛋白ALB
def classify_aft_alb(val):
    try:
        val = float(val)

        # 手动输入
        min_val = 0
        max_val = 57.9

        if val < min_val:
            val = min_val
        elif val > max_val:
            val = max_val

        # 归一化公式: (x - min) / (max - min)
        normalized = (val - min_val) / (max_val - min_val)
        return normalized

    except (ValueError, TypeError):
        return None  # 处理非数值情况

data['术后0-3天白蛋白ALB'] = data['术后0-3天白蛋白ALB'].apply(classify_aft_alb)

# 术后0-3天前白蛋白PALB
def classify_aft_palb(val):
    try:
        val = float(val)

        # 手动输入
        min_val = 0
        max_val = 59.8

        if val < min_val:
            val = min_val
        elif val > max_val:
            val = max_val

        # 归一化公式: (x - min) / (max - min)
        normalized = (val - min_val) / (max_val - min_val)
        return normalized

    except (ValueError, TypeError):
        return None  # 处理非数值情况

data['术后0-3天前白蛋白PALB'] = data['术后0-3天前白蛋白PALB'].apply(classify_aft_palb)

# 保存处理后的数据
write_path = "data1.csv"
data.to_csv(write_path, index=False, encoding='utf-8-sig')