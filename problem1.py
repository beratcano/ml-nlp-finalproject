import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

dfcsv = pd.read_csv("csv/Customer_Health_Profile.csv")
dfcsv.index = dfcsv.index+1
df = dfcsv.rename(columns={"Age": "age",
                         "BMI": "bmi",
                         "Smoking_Habit": "smoke",
                         "Exercise_Frequency": "exer",
                         "Blood_Pressure": "bp",
                         "Health_Risk": "risk"})

# print(df.info())
# column - null count - datatype
# age - 0 - int64
# bmi - 350 - float64 // Need to handle missing values in this column
# smoke - 0 - object
# exer - 0 - object
# bp - 0 - int64
# risk - 0 - object

# print(df['smoke'].unique())  // Yes, No
# print(df['exer'].unique())   // Low, Moderate, High
# print(df['risk'].unique())   // Low, Medium, High
# Mapping categorical values to numeric
smoke_mapping = {"No": 0, "Yes": 1}
exer_mapping = {"Low": 1, "Moderate": 2, "High": 3}
risk_mapping = {"Low": 1, "Medium": 2, "High": 3}

df["smoke"] = df["smoke"].map(smoke_mapping)
df["exer"] = df["exer"].map(exer_mapping)
df["risk"] = df["risk"].map(risk_mapping)

# Finding missing values with K-Nearest Neighbour Method
df_kn = df.copy()
df_kn["bmi"] = KNNImputer(n_neighbors=5).fit_transform(df[["bmi"]])

print(df.info())
print(df_kn.info())

# Standartization of columns with numerical values (not categorical values)
num_cols = ["age", "bmi", "bp"]
df_std = df_kn.copy()
df_std[num_cols] = StandardScaler().fit_transform(df_kn[num_cols])

# Splitting the data int training and test sets (70% train - 30% test)
x = df_std[["age", "bmi", "smoke", "exer", "bp"]]
y = df_std["risk"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)
