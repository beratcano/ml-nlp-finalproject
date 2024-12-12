import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

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
# bmi - 350 - float64               // Need to handle missing values in this column
# smoke - 0 - object
# exer - 0 - object
# bp - 0 - int64
# risk - 0 - object

# print(df['smoke'].unique())       // Yes, No
# print(df['exer'].unique())        // Low, Moderate, High
# print(df['risk'].unique())        // Low, Medium, High

# Mapping categorical values to numeric
df["smoke"] = df["smoke"].map({"No": 0, "Yes": 1})
df["exer"] = df["exer"].map({"Low": 0, "Moderate": 1, "High": 2})
df["risk"] = df["risk"].map({"Low": 0, "Medium": 1, "High": 2})

df_kn = df.copy()
df_kn[columns] = KNNImputer(n_neighbors=5).fit_transform(df[columns])

# print(df.info())                  // bmi : 3150 data
# print(df_kn.info())               // bmi : 3500 data

df_std = df_kn.copy()
df_std[num_cols] = StandardScaler().fit_transform(df_kn[num_cols])

x = df_std[["age", "bmi", "smoke", "exer", "bp"]]
y = df_std["risk"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy Score: {accuracy*100: .2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Classification Report Table
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
label_mapping = {
    "0.0": "Low",
    "1.0": "Medium",
    "2.0": "High"
}
report_df.rename(index=label_mapping, inplace=True)
fig, ax = plt.subplots()
ax.axis("off")
table = plt.table(
    cellText=report_df.round(2).values,
    colLabels=report_df.columns,
    rowLabels=report_df.index,
    loc="center",
    cellLoc="center"
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(report_df.columns))))
plt.title("Classification Report", fontsize=14, pad= 2)
plt.tight_layout()
plt.show()

# Numerical Values
for col in num_cols:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='risk', y=col, data=df_kn, palette='viridis')
    plt.title(f'{col.capitalize()} vs. Health Risk', fontsize=16)
    plt.xlabel('Health Risk', fontsize=12)
    plt.ylabel(col.capitalize(), fontsize=12)
    plt.xticks(ticks=[0, 1, 2], labels=['Low', 'Medium', 'High'])
    plt.show()

sns.pairplot(df, hue='risk', palette='viridis', height=1.5)
plt.show()

# Preparing Categorical Values for plot
smoke_risk_counts = pd.crosstab(df['smoke'], df['risk'])
smoke_risk_percentage = pd.crosstab(df['smoke'], df['risk'], normalize='index') * 100
exer_risk_counts = pd.crosstab(df['exer'], df['risk'])
exer_risk_percentage = pd.crosstab(df['exer'], df['risk'], normalize='index') * 100

# Smoking Habit Plot
fig, ax1 = plt.subplots(figsize=(8, 6))
smoke_bars = smoke_risk_percentage.plot(kind='bar', stacked=True, ax=ax1, colormap='viridis', alpha=0.7)
for i, p in enumerate(ax1.patches):
    row = i // len(smoke_risk_counts.columns)
    col = i % len(smoke_risk_counts.columns)
    percent = smoke_risk_percentage.values[row, col]
    count = smoke_risk_counts.values[row, col]
    if percent > 0:
        ax1.text(
            p.get_x() + p.get_width() / 2,          # type: ignore
            p.get_y() + p.get_height() / 2,         # type: ignore
            f'{count}\n({percent:.1f}%)',
            ha='center', va='center', fontsize=9, color='black'
        )

ax1.set_title('Smoking Habit vs. Health Risk', fontsize=16)
ax1.set_xlabel('Smoking Habit (0=No, 1=Yes)', fontsize=12)
ax1.set_ylabel('Percentage (%)', fontsize=12)
ax1.legend(title='Health Risk', labels=['Low', 'Medium', 'High'], bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(ticks=[0, 1], labels=['No', 'Yes'], fontsize=10)
plt.tight_layout()
plt.show()

# Exercise Frequency Plot
fig, ax2 = plt.subplots(figsize=(8, 6))
exer_bars = exer_risk_percentage.plot(kind='bar', stacked=True, ax=ax2, colormap='viridis', alpha=0.7)
for i, p in enumerate(ax2.patches):
    row = i // len(exer_risk_counts.columns)
    col = i % len(exer_risk_counts.columns)
    percent = exer_risk_percentage.values[row, col]
    count = exer_risk_counts.values[row, col]
    if percent > 0:
        ax2.text(
            p.get_x() + p.get_width() / 2,          # type: ignore
            p.get_y() + p.get_height() / 2,         # type: ignore
            f'{count}\n({percent:.1f}%)',
            ha='center', va='center', fontsize=9, color='black'
        )

ax2.set_title('Exercise Frequency vs. Health Risk', fontsize=16)
ax2.set_xlabel('Exercise Frequency (0=Low, 1=Moderate, 2=High)', fontsize=12)
ax2.set_ylabel('Percentage (%)', fontsize=12)
ax2.legend(title='Health Risk', labels=['Low', 'Medium', 'High'], bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(ticks=[0, 1, 2], labels=['Low', 'Moderate', 'High'], fontsize=10)
plt.tight_layout()
plt.show()

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low", "Medium", "High"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()