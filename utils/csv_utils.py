import pandas as pd
import numpy as np


def get_mean_std(file_path, numeric_features):
    num = pd.read_csv(file_path)[numeric_features].describe()
    MEAN = np.array(num.T['mean'])
    STD = np.array(num.T['std'])
    return MEAN, STD


def get_vocabulary(file_path, categorical_features):
    cat = pd.read_csv(file_path)[categorical_features]
    categories = {}
    for feat in categorical_features:
        categories[feat] = list(cat[feat].unique())
    return categories
