"""数据清洗脚本。

功能：
1. 自动读取 .csv / .xlsx 原始数据。
2. 将中文列名转换为便于建模的英文列名，同时尽量保留核心含义。
3. 缺失值处理：连续变量用中位数填充；分类变量用众数填充。
   - 对术前前白蛋白 PALB、术前白蛋白 ALB、术前血红蛋白 HGB 中的 0 视为缺失。
   - 对最新版 pTNM 中的 5 视为缺失。
4. 连续变量进行 Z-score 标准化。
5. 分类变量编码：
   - 二分类变量尽量编码为 0/1；
   - 多分类变量使用 LabelEncoder，并保存映射表，便于未来出现新类别时复用。
6. 输出详细中间结果与注释，便于追踪处理流程。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm.auto import tqdm


# =========================
# 基础配置
# =========================
RAW_CANDIDATES = [Path("data0.csv"), Path("原始数据.xlsx"), Path("data0.xlsx")]
OUTPUT_DATA = Path("data1.csv")
OUTPUT_MAPPING = Path("column_mapping.json")
OUTPUT_ENCODING = Path("categorical_encoding_mapping.json")

TARGET_COLUMN_CN = "肺部感染（0=无、1=有）"
TARGET_COLUMN_EN = "pulmonary_infection"


# =========================
# 列名映射
# =========================
COLUMN_MAPPING: Dict[str, str] = {
    "性别\n（1=男，2=女）": "sex",
    "年龄": "age",
    "主要诊断": "main_diagnosis",
    "BMI": "bmi",
    "既往头颈部疾病手术外伤史（0=无，1=有）": "history_head_neck_surgery",
    "冠心病（0=无，1=有）": "coronary_heart_disease",
    "高血压（0=无，1=有）": "hypertension",
    "外周血管病（0=无，1=有）": "peripheral_vascular_disease",
    "免疫性疾病（0=无，1=有）": "immune_disease",
    "糖尿病（0=无，1=有）": "diabetes",
    "高脂血症（0=无，1=有）": "hyperlipidemia",
    "激素类（0=无，1=有）": "hormone_use",
    "吸烟史（0=无，1=有）": "smoking_history",
    "饮酒史（0=无，1=有）": "alcohol_history",
    "术前前白蛋白PALB（0=未测）": "pre_op_palb",
    "术前白蛋白ALB（0=未测）": "pre_op_alb",
    "术前血红蛋白HGB（0=未测）": "pre_op_hgb",
    "术前口咽拭子（未生长=0，阳性=1，未测=2）": "pre_op_oropharyngeal_swab",
    "病变部位（喉=1、鼻=2、扁桃体=3、腮腺=4、咽部=5、唇=6、甲状腺=7、食管=8、舌=9、耳=10 、颌、面部=11、口底=12、气管=13、口腔、牙龈=14、梨状窝=15、上颌窦=16、皮肤肿物=17、腭=18、胸部=19、颈部=20）": "lesion_site",
    "ASA评分": "asa_score",
    "术前抗菌药物：未用=0、头孢曲松钠=1、头孢呋辛钠=2、左奥硝唑氯化钠=3、吗啉硝唑氯化钠=4、甲磺酸左氧氟沙星氯化钠=5、克林霉素磷酸酯=6、盐酸莫西沙星=7、头孢噻肟钠=8、甲硝唑氯化钠=9、头孢米诺钠=10、奥硝唑=11、美洛西林钠舒巴坦钠=12、注射用哌拉西林钠舒巴坦钠=13、头孢哌酮钠舒巴坦钠=14、阿奇霉素=15、美罗培南=16、万古霉素=17": "pre_op_antibiotics",
    "手术时长（min）（不详=0，其他具体写出）": "surgery_duration",
    "术中输血（有=1，无=0）": "intraop_transfusion",
    "颈清扫（0=无，1=有）": "neck_dissection",
    "术前放疗（0=无，1=有）": "pre_op_radiotherapy",
    "术前化疗（0=无，1=有）": "pre_op_chemotherapy",
    "术前同步放化疗（0=无，1:<=60Gy,2:>60Gy，有，但具体剂量不详=3）": "pre_op_chemoradiotherapy",
    "腔镜（0=无，1=有）": "endoscopic_surgery",
    "气管造瘘（无=0，有=1）": "tracheostomy",
    "术后病理无=0、warthin瘤=1、鳞癌=2、乳头状癌=3、多形性腺瘤=4、基底细胞癌=5、腺癌=6、黑色素瘤=7、未见癌=8、肉瘤=9、良性=10、甲状腺髓样癌=11、粘液表皮样癌=12、分化差的癌=13、腺样囊性癌=14、淋巴细胞瘤=15、梭形细胞瘤=16、囊肿=17": "post_op_pathology",
    "分化（无=0、低=1、中=2、高=3、未确定=4）": "differentiation",
    "最新版pTNM（5=5期，1=I期，2=II期，3=III期，4=IV期，5=无）": "ptnm_stage",
    "多原发（0=否、1=是）": "multiple_primary",
    "非计划二次手术(0=否，1=是)": "unplanned_reoperation",
    "肺部感染（0=无、1=有）": "pulmonary_infection",
    "吻合口瘘（0=无、1=有）": "anastomotic_leak",
    "吻合口瘘确认距术后天数（0=未发生，具体已写出）": "leak_confirm_day",
    "脂肪液化（0=无，1=有）": "fat_liquefaction",
    "切口感染(0=无、1=有）": "wound_infection",
    "是否多重耐药（0=否，1＝是）": "multidrug_resistant",
}

BINARY_COLUMNS = {
    "sex",
    "history_head_neck_surgery",
    "coronary_heart_disease",
    "hypertension",
    "peripheral_vascular_disease",
    "immune_disease",
    "diabetes",
    "hyperlipidemia",
    "hormone_use",
    "smoking_history",
    "alcohol_history",
    "intraop_transfusion",
    "neck_dissection",
    "pre_op_radiotherapy",
    "pre_op_chemotherapy",
    "endoscopic_surgery",
    "tracheostomy",
    "multiple_primary",
    "unplanned_reoperation",
    "pulmonary_infection",
    "anastomotic_leak",
    "fat_liquefaction",
    "wound_infection",
    "multidrug_resistant",
}

CONTINUOUS_COLUMNS = {
    "age",
    "bmi",
    "pre_op_palb",
    "pre_op_alb",
    "pre_op_hgb",
    "surgery_duration",
    "leak_confirm_day",
}

SPECIAL_MISSING_RULES = {
    "pre_op_palb": [0],
    "pre_op_alb": [0],
    "pre_op_hgb": [0],
    "ptnm_stage": [5],
}


# =========================
# 工具函数
# =========================
def load_raw_data() -> pd.DataFrame:
    """按优先顺序读取原始数据文件。"""
    for path in RAW_CANDIDATES:
        if path.exists():
            print(f"[INFO] 读取原始数据：{path}")
            if path.suffix.lower() == ".xlsx":
                return pd.read_excel(path)
            return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    raise FileNotFoundError("未找到 data0.csv / data0.xlsx / 原始数据.xlsx")


def sanitize_column_name(name: str) -> str:
    """为未知列生成一个尽量稳定的英文占位名。"""
    cleaned = (
        str(name)
        .replace("\n", " ")
        .replace("（", "_")
        .replace("）", "")
        .replace("(", "_")
        .replace(")", "")
        .replace("=", "_")
        .replace("、", "_")
        .replace("，", "_")
        .replace("：", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in cleaned)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_").lower() or "unknown_column"


def convert_to_numeric_when_possible(df: pd.DataFrame) -> pd.DataFrame:
    """尽可能将列转换为数值型。"""
    for column in tqdm(df.columns, desc="尝试数值化列"):
        df[column] = pd.to_numeric(df[column], errors="ignore")
    return df


def apply_special_missing_rules(df: pd.DataFrame) -> pd.DataFrame:
    """按研究规则将特定取值视为缺失。"""
    for column, missing_values in SPECIAL_MISSING_RULES.items():
        if column in df.columns:
            df[column] = df[column].replace(missing_values, np.nan)
    return df


def encode_sex_column(series: pd.Series) -> pd.Series:
    """性别特殊编码：男->1，女->0；若本来已是 1/2，则映射为 1/0。"""
    mapping = {
        "男": 1,
        "女": 0,
        1: 1,
        2: 0,
        "1": 1,
        "2": 0,
    }
    return series.map(mapping).where(series.notna(), np.nan)


# =========================
# 主流程
# =========================
def main() -> None:
    data = load_raw_data()
    print(f"[INFO] 原始数据维度：{data.shape}")

    # 1. 重命名列
    renamed_columns: Dict[str, str] = {}
    for column in data.columns:
        renamed_columns[column] = COLUMN_MAPPING.get(column, sanitize_column_name(column))
    data = data.rename(columns=renamed_columns)
    OUTPUT_MAPPING.write_text(json.dumps(renamed_columns, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] 已保存列名映射：{OUTPUT_MAPPING}")

    # 2. 性别单独处理
    if "sex" in data.columns:
        data["sex"] = encode_sex_column(data["sex"])

    # 3. 尽可能转为数值型
    data = convert_to_numeric_when_possible(data)

    # 4. 特殊缺失值规则
    data = apply_special_missing_rules(data)

    # 5. 自动识别连续/分类变量
    continuous_columns: List[str] = [col for col in CONTINUOUS_COLUMNS if col in data.columns]
    categorical_columns: List[str] = [
        col for col in data.columns if col not in continuous_columns
    ]

    # 6. 缺失值填充
    if continuous_columns:
        print("[INFO] 连续变量使用中位数填充缺失值")
        cont_imputer = SimpleImputer(strategy="median")
        data[continuous_columns] = cont_imputer.fit_transform(data[continuous_columns])

    if categorical_columns:
        print("[INFO] 分类变量使用众数填充缺失值")
        cat_imputer = SimpleImputer(strategy="most_frequent")
        data[categorical_columns] = cat_imputer.fit_transform(data[categorical_columns])

    # 7. 分类编码
    encoding_mapping: Dict[str, Dict[str, int]] = {}
    for column in tqdm(categorical_columns, desc="编码分类变量"):
        if column not in data.columns:
            continue

        # 跳过已经是 0/1 的二分类列
        unique_values = sorted(pd.Series(data[column]).dropna().unique().tolist())
        if column in BINARY_COLUMNS and set(unique_values).issubset({0, 1}):
            continue

        # 常见 ASA 罗马数字转换
        if column == "asa_score":
            asa_map = {"Ⅰ": 1, "Ⅱ": 2, "Ⅲ": 3, "Ⅳ": 4, "I": 1, "II": 2, "III": 3, "IV": 4}
            data[column] = pd.Series(data[column]).replace(asa_map)
            maybe_numeric = pd.to_numeric(data[column], errors="coerce")
            if maybe_numeric.notna().all():
                data[column] = maybe_numeric.astype(int)
                continue

        # 已为纯数字且无需再次编码时保留
        maybe_numeric = pd.to_numeric(data[column], errors="coerce")
        if maybe_numeric.notna().all():
            data[column] = maybe_numeric
            continue

        encoder = LabelEncoder()
        as_str = data[column].astype(str)
        data[column] = encoder.fit_transform(as_str)
        encoding_mapping[column] = {cls: int(idx) for idx, cls in enumerate(encoder.classes_)}

    # 8. 连续变量标准化（Z-score）
    if continuous_columns:
        print("[INFO] 对连续变量执行 Z-score 标准化")
        scaler = StandardScaler()
        data[continuous_columns] = scaler.fit_transform(data[continuous_columns])

    # 9. 目标变量强制转为 int
    for target_candidate in [TARGET_COLUMN_EN, "wound_infection"]:
        if target_candidate in data.columns:
            data[target_candidate] = pd.to_numeric(data[target_candidate], errors="coerce").fillna(0).astype(int)

    OUTPUT_ENCODING.write_text(json.dumps(encoding_mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    data.to_csv(OUTPUT_DATA, index=False, encoding="utf-8-sig")

    print(f"[INFO] 清洗完成，输出文件：{OUTPUT_DATA}")
    print(f"[INFO] 清洗后数据维度：{data.shape}")
    print(f"[INFO] 因变量列：{TARGET_COLUMN_EN if TARGET_COLUMN_EN in data.columns else '未找到'}")


if __name__ == "__main__":
    main()
