# To be updated

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

dfcsv = pd.read_csv("csv/Customer_Health_Profile.csv")
dfcsv.index = dfcsv.index+1
df = dfcsv.rename(columns={"Age": "age",
                         "BMI": "bmi",
                         "Smoking_Habit": "smoke",
                         "Exercise_Frequency": "exer",
                         "Blood_Pressure": "bp",
                         "Health_Risk": "risk"})

columns = ["age", "bmi", "smoke", "exer", "bp", "risk"]
num_cols = ["age", "bmi", "bp"]
cat_cols = ["smoke", "exer", "risk"]

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
exer_mapping = {"Low": 0, "Moderate": 1, "High": 2}
risk_mapping = {"Low": 0, "Medium": 1, "High": 2}

df["smoke"] = df["smoke"].map(smoke_mapping)
df["exer"] = df["exer"].map(exer_mapping)
df["risk"] = df["risk"].map(risk_mapping)

# Finding missing values with K-Nearest Neighbour Method
df_kn = df.copy()
df_kn[columns] = KNNImputer(n_neighbors=5).fit_transform(df[columns])

# print(df.info())          // bmi : 3150 data
# print(df_kn.info())       // bmi : 3500 data

# Standartization of columns with numerical values (not categorical values)
df_std = df_kn.copy()
df_std[num_cols] = StandardScaler().fit_transform(df_kn[num_cols])

# Splitting the data int training and test sets (70% train - 30% test)
x = df_std[["age", "bmi", "smoke", "exer", "bp"]]
y = df_std["risk"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)

# Building the model and reporting the accuracy
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy Score: {accuracy*100: .2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Boxplot to visualize the relationship between numerical features and risk
for col in num_cols:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='risk', y=col, data=df, palette='viridis')
    plt.title(f'{col.capitalize()} vs. Health Risk', fontsize=16)
    plt.xlabel('Health Risk', fontsize=12)
    plt.ylabel(col.capitalize(), fontsize=12)
    plt.xticks(ticks=[0, 1, 2], labels=['Low', 'Medium', 'High'])
    plt.show()

# Pairwise relationships
sns.pairplot(df, hue='risk', palette='viridis', height=1.5)
plt.show()

# Crosstab for Smoking Habit vs. Health Risk
smoke_risk_counts = pd.crosstab(df['smoke'], df['risk'])  # Raw counts
smoke_risk_percentage = pd.crosstab(df['smoke'], df['risk'], normalize='index') * 100  # Percentages
exer_risk_counts = pd.crosstab(df['exer'], df['risk'])
exer_risk_percentage = pd.crosstab(df['exer'], df['risk'], normalize='index') * 100

# Smoking Habit vs. Health Risk
fig, ax1 = plt.subplots(figsize=(8, 6))
smoke_bars = smoke_risk_percentage.plot(kind='bar', stacked=True, ax=ax1, colormap='viridis', alpha=0.7)

# Annotate Bars for Smoking Habit
for i, p in enumerate(ax1.patches):
    row = i // len(smoke_risk_counts.columns)  # Row index (smoking category)
    col = i % len(smoke_risk_counts.columns)  # Column index (risk category)
    percent = smoke_risk_percentage.values[row, col]
    count = smoke_risk_counts.values[row, col]
    if percent > 0:  # Avoid annotating zero values
        ax1.text(
            p.get_x() + p.get_width() / 2,
            p.get_y() + p.get_height() / 2,
            f'{count}\n({percent:.1f}%)',
            ha='center', va='center', fontsize=9, color='black'
        )

# Formatting Smoking Plot
ax1.set_title('Smoking Habit vs. Health Risk', fontsize=16)
ax1.set_xlabel('Smoking Habit (0=No, 1=Yes)', fontsize=12)
ax1.set_ylabel('Percentage (%)', fontsize=12)
ax1.legend(title='Health Risk', labels=['Low', 'Medium', 'High'], bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(ticks=[0, 1], labels=['No', 'Yes'], fontsize=10)
plt.tight_layout()
plt.show()

# Exercise Frequency vs. Health Risk
fig, ax2 = plt.subplots(figsize=(8, 6))
exer_bars = exer_risk_percentage.plot(kind='bar', stacked=True, ax=ax2, colormap='viridis', alpha=0.7)

# Annotate Bars for Exercise Frequency
for i, p in enumerate(ax2.patches):
    row = i // len(exer_risk_counts.columns)  # Row index (exercise category)
    col = i % len(exer_risk_counts.columns)  # Column index (risk category)
    percent = exer_risk_percentage.values[row, col]
    count = exer_risk_counts.values[row, col]
    if percent > 0:  # Avoid annotating zero values
        ax2.text(
            p.get_x() + p.get_width() / 2,
            p.get_y() + p.get_height() / 2,
            f'{count}\n({percent:.1f}%)',
            ha='center', va='center', fontsize=9, color='black'
        )

# Formatting Exercise Plot
ax2.set_title('Exercise Frequency vs. Health Risk', fontsize=16)
ax2.set_xlabel('Exercise Frequency (0=Low, 1=Moderate, 2=High)', fontsize=12)
ax2.set_ylabel('Percentage (%)', fontsize=12)
ax2.legend(title='Health Risk', labels=['Low', 'Medium', 'High'], bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(ticks=[0, 1, 2], labels=['Low', 'Moderate', 'High'], fontsize=10)
plt.tight_layout()
plt.show()