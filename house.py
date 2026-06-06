#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#load the dataset
df = pd.read_csv("Housing.csv")
print(df.head())
#check the dataset
print(df.info())
print(df.isnull().sum())
#data cleaning
df = df.dropna(axis=1)
#Convert Categorical Data
df = pd.get_dummies(df, drop_first=True)