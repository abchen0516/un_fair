import os
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


# def load_oulad(data_path):
#     df = pd.read_csv(os.path.join(data_path, 'oulad_clean.csv'))
#     protected_attribute = 'gender'
#     majority_group_name = "Male"
#     minority_group_name = "Female"
#     class_label = 'final_result'
#
#     # Label gender
#     df['gender'] = ['Male' if v == 'M' else 'Female' for v in df['gender']]
#     # label encode
#     le = preprocessing.LabelEncoder()
#     for i in df.columns:
#         if df[i].dtypes == 'object':
#             df[i] = le.fit_transform(df[i])
#     # Splitting data into train and test
#     length = len(df.columns)
#     X = df.iloc[:, :length - 1]
#     y = df.iloc[:, length - 1]
#     X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y.to_numpy(), test_size=0.3, random_state=42)
#
#     # Get index
#     feature = X.keys().tolist()
#     sa_index = feature.index(protected_attribute)
#     p_Group = 0
#
#     input_dim = X_train.shape[1]
#     output_dim = np.unique(y_train).shape[0]
#
#     return X_train, X_test, y_train, y_test, sa_index, p_Group, protected_attribute, majority_group_name, minority_group_name, input_dim, output_dim


def load_oulad(data_path):
    protected_attribute = 'gender'
    majority_group_name = "Male"
    minority_group_name = "Female"
    # protected_attribute = 'poverty'
    # majority_group_name = "true"
    # minority_group_name = "false"
    # protected_attribute = 'disability'
    # majority_group_name = "true"
    # minority_group_name = "false"
    class_label = 'final_result'

    X_train = pd.read_csv(os.path.join(data_path, 'X_train_stClick_7030.csv'))
    y_train = pd.read_csv(os.path.join(data_path, 'y_train_stClick_7030.csv'))
    X_test = pd.read_csv(os.path.join(data_path, 'X_test_stClick_7030.csv'))
    y_test = pd.read_csv(os.path.join(data_path, 'y_test_stClick_7030.csv'))

    # Get index
    feature = X_train.keys().tolist()
    sa_index = feature.index(protected_attribute)
    p_Group = 0

    X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy().flatten(), X_test.to_numpy(), y_test.to_numpy().flatten()
    input_dim = X_train.shape[1]
    output_dim = np.unique(y_train).shape[0]

    return X_train, X_test, y_train, y_test, sa_index, p_Group, protected_attribute, majority_group_name, minority_group_name, input_dim, output_dim


def load_student_por(data_path):
    df = pd.read_csv(os.path.join(data_path, 'student_por_clean.csv'))
    protected_attribute = 'sex'
    majority_group_name = "Male"
    minority_group_name = "Female"
    class_label = 'class'

    # Label class
    df['class'] = [1 if v == "Pass" else 0 for v in df['class']]
    # Label sex
    df['sex'] = ["Female" if v == "F" else "Male" for v in df['sex']]
    # label encode
    le = preprocessing.LabelEncoder()
    for i in df.columns:
        if df[i].dtypes == 'object':
            df[i] = le.fit_transform(df[i])
    # Splitting data into train and test
    length = len(df.columns)
    X = df.iloc[:, :length - 1]
    y = df.iloc[:, length - 1]
    X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y.to_numpy(), test_size=0.3, random_state=42)

    # Get index
    feature = X.keys().tolist()
    sa_index = feature.index(protected_attribute)
    p_Group = 0

    input_dim = X_train.shape[1]
    output_dim = np.unique(y_train).shape[0]

    return X_train, X_test, y_train, y_test, sa_index, p_Group, protected_attribute, majority_group_name, minority_group_name, input_dim, output_dim


def load_xAPI_Edu(data_path):
    df = pd.read_csv(os.path.join(data_path, 'xAPI-Edu-Data.csv'))
    protected_attribute = 'gender'
    majority_group_name = "Male"
    minority_group_name = "Female"
    class_label = 'Class'

    # Label class
    df['Class'] = [1 if v == "Medium-High" else 0 for v in df['Class']]
    # Label sex
    df['gender'] = ["Female" if v == "F" else "Male" for v in df['gender']]
    # label encode
    le = preprocessing.LabelEncoder()
    for i in df.columns:
        if df[i].dtypes == 'object':
            df[i] = le.fit_transform(df[i])
    # Splitting data into train and test
    length = len(df.columns)
    X = df.iloc[:, :length - 1]
    y = df.iloc[:, length - 1]
    X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y.to_numpy(), test_size=0.3, random_state=42)

    # Get index
    feature = X.keys().tolist()
    sa_index = feature.index(protected_attribute)
    p_Group = 0

    input_dim = X_train.shape[1]
    output_dim = np.unique(y_train).shape[0]

    return X_train, X_test, y_train, y_test, sa_index, p_Group, protected_attribute, majority_group_name, minority_group_name, input_dim, output_dim


def construct_dataset(dataset, data_path):
    if dataset == 'oulad':
        X_train, X_test, y_train, y_test, sa_index, p_Group, protected_attribute, majority_group_name, minority_group_name, input_dim, output_dim = load_oulad(data_path)
    elif dataset == 'student':
        X_train, X_test, y_train, y_test, sa_index, p_Group, protected_attribute, majority_group_name, minority_group_name, input_dim, output_dim = load_student_por(data_path)
    elif dataset == 'xapi':
        X_train, X_test, y_train, y_test, sa_index, p_Group, protected_attribute, majority_group_name, minority_group_name, input_dim, output_dim = load_xAPI_Edu(data_path)
    else:
        raise NotImplementedError('Not support!')

    return X_train, X_test, y_train, y_test, sa_index, p_Group, protected_attribute, majority_group_name, minority_group_name, input_dim, output_dim


class PackData(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], idx