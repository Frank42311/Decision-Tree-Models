'''
这并非是在原始数据上进行的操作
手动对原始数据进行了一些操作
这里只是做简单记录
'''

import pandas as pd

"""
# 读取CSV文件
application_data_path = "E:\\Machine_Learning_projects\\data\\LoanDefaulter\\train\\train.csv"
df = pd.read_csv(application_data_path, low_memory=False)

# ------------------------------------------------------------------------------------------------------
# 处理工作相关的列
# 如果emp_title和emp_length都为空，则填充相应的值
df.loc[(df['emp_title'].isnull()) & (df['emp_length'].isnull()), 'emp_title'] = 'unemployed'
df.loc[(df['emp_title'].isnull()) & (df['emp_length'].isnull()), 'emp_length'] = '0 year'

# 如果emp_title有值，emp_length为空，则填充emp_length的众数
emp_length_mode = df['emp_length'].mode().iloc[0]
df.loc[(df['emp_title'].notnull()) & (df['emp_length'].isnull()), 'emp_length'] = emp_length_mode

# 如果emp_length有值，emp_title为空，则填充emp_title的众数
emp_title_mode = df['emp_title'].mode().iloc[0]
df.loc[(df['emp_length'].notnull()) & (df['emp_title'].isnull()), 'emp_title'] = emp_title_mode

# ------------------------------------------------------------------------------------------------------
# 使用中位数或众数填补剩余列
# 遍历DataFrame的每一列
# 定义要检查的列
columns_to_check = [11, 20, 29, 39, 41, 46, 85]

# 遍历每一列
for col in columns_to_check:
    col_data = df.iloc[:, col]

    # 尝试将列转换为数字，记录无法转换的索引
    invalid_rows = pd.to_numeric(col_data, errors='coerce').isna()

    # 仅保留有效的行
    df = df.loc[~invalid_rows]

# ------------------------------------------------------------------------------------------------------
# 处理存在多种值的列，删掉次要值所在的行
# 遍历DataFrame的每一列
for column in df.columns:
    # 如果列的数据类型是数值型
    if df[column].dtype == 'int64' or df[column].dtype == 'float64':
        # 计算中位数并填充缺失值
        median_value = df[column].median()
        df[column].fillna(median_value, inplace=True)
    # 如果列的数据类型是对象（可能是字符串）
    elif df[column].dtype == 'object':
        # 计算众数并填充缺失值
        mode_value = df[column].mode().iloc[0]
        df[column].fillna(mode_value, inplace=True)
"""



# ------------------------------------------------------------
# 第二轮修改


import pandas as pd
import numpy as np

# 读取数据
application_data_path = "E:\\Machine_Learning_projects\\data\\LoanDefaulter\\train\\train.csv"
data = pd.read_csv(application_data_path, encoding='utf-8', low_memory=False)

# ------------------------------------------------------------------------------------------------------
# 使用中位数或众数填补剩余列
# 遍历DataFrame的每一列
# 定义要检查的列
columns_to_check = [11, 20, 29, 39, 41, 46, 85]

# 遍历每一列
for col in columns_to_check:
    col_data = data.iloc[:, col]

    # 尝试将列转换为数字，记录无法转换的索引
    invalid_rows = pd.to_numeric(col_data, errors='coerce').isna()

    # 仅保留有效的行
    data = data.loc[~invalid_rows]

# ------------------------------------------------------------
# 第二轮修改

# 区分特征数据和目标列
features_data = data.drop(columns=['loan_status'])
target_column = data['loan_status']

# 丢弃方差较小的数值特征
numeric_features = features_data.select_dtypes(include=['number'])
variance_threshold = 0.1
low_variance_features = numeric_features.var()[numeric_features.var() < variance_threshold].index
features_data.drop(columns=low_variance_features, inplace=True)

# 丢弃90%以上值相同的特征
threshold_same_value = 0.9
for column in features_data.columns:
    max_count = features_data[column].value_counts(normalize=True).values[0]
    if max_count > threshold_same_value:
        features_data.drop(columns=[column], inplace=True)

# 丢弃数值特征中高度相关的特征
numeric_features_after_drop = features_data.select_dtypes(include=['number'])
correlation_matrix = numeric_features_after_drop.corr().abs()
upper_triangle_corr = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_triangle_corr.columns if any(upper_triangle_corr[column] > 0.8)]
features_data.drop(columns=to_drop, inplace=True)

# 合并目标列并保存新CSV
new_data = pd.concat([features_data, target_column], axis=1)
new_data.to_csv("E:\\Machine_Learning_projects\\data\\LoanDefaulter\\train\\new_train.csv", index=False)

