#!/usr/bin/env python3
"""
Titanic Dataset - Clean & Preprocess Pipeline
Author: Abhidha Gupta
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import os
print("Current working directory:", os.getcwd())
print("Files in directory:", os.listdir())

# Load dataset
def load_data(path):
    try:
        df = pd.read_csv(path)
        print("✅ Data loaded. Shape:", df.shape)
        return df
    except FileNotFoundError:
        print("❌ File not found!")
        return None

# Handle missing values
def clean_missing(df):
    df = df.copy()
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    if df['Cabin'].isnull().mean() > 0.7:
        df.drop('Cabin', axis=1, inplace=True)
    else:
        df['Cabin'].fillna('Unknown', inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    return df

# Encode categorical variables
def encode_features(df):
    df = df.copy()
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    df = pd.get_dummies(df, columns=['Embarked', 'Pclass'], drop_first=True)
    drop_cols = ['Name', 'Ticket', 'PassengerId']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)
    return df

# Scale numeric features
def scale_features(df):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    to_scale = [col for col in numeric_cols if col != 'Survived' and df[col].nunique() > 2]
    scaler = StandardScaler()
    df[to_scale] = scaler.fit_transform(df[to_scale])
    return df

# Cap outliers using IQR method
def cap_outliers(df):
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == 'Survived': continue
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        df[col] = np.clip(df[col], lower, upper)
    return df

# Complete pipeline
def preprocess_pipeline(file_path):
    df = load_data(file_path)
    if df is not None:
        df = clean_missing(df)
        df = encode_features(df)
        df = scale_features(df)
        df = cap_outliers(df)
        print("✅ Preprocessing complete. Final shape:", df.shape)
        return df

# Run the pipeline
if __name__ == "__main__":
    final_df = preprocess_pipeline('D:/titanic-data-cleaning/data/Titanic-Dataset.csv')
