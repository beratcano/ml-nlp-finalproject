# **Clustering Analysis: Online Shopping Behavior**

## **Objective**
Analyze online shopping behavior using clustering techniques to group users into meaningful segments based on their activity.

---

## **Steps Followed**

### **1. Data Preparation**
- **Dataset**: *Online_Shopping_Behavior.csv*
- **Renamed Columns**:
  - `Session_Duration` → `dur`
  - `Page_Views` → `view`
  - `Purchase_Amount` → `amount`
  - `Bounce_Rate` → `brate`

#### **Handling Missing Values**
- Missing values in the column `view` were imputed using the **KNNImputer** with `n_neighbors=5`.
- **Result**:
  - Original data: 5400 rows (with missing values).
  - Cleaned data: 6000 rows (missing values handled).

#### **Standardization**
- Used `StandardScaler` to standardize the columns: `dur`, `view`, `amount`, `brate`.

---

### **2. Elbow Method for Optimal K**
- **KMeans Algorithm**:
  - Applied K-means clustering for `k` in range (1, 11).
  - Recorded **WCSS (Within-Cluster Sum of Squares)** for each `k`.

- **Results**:
  - **WCSS Values**:
    ```python
    [24000.0, 19514.24, 16744.67, 14566.60, 12932.88, 11710.69, 10558.95, 9477.00, 8980.54, 8448.16]
    ```
  - **Difference in WCSS**:
    ```python
    [4485.76, 2769.57, 2178.08, 1633.72, 1222.19, 1151.74, 1081.95, 496.46, 532.38]
    ```

#### **Key Observation**:
- The **Elbow Point** (where the rate of WCSS reduction slows) is around `k = 5`.

---

### **3. K-Means Clustering**
- **Selected Clusters**: `k = 5`.
- **Cluster Assignment**:
  - Cluster labels added as a new column: `KN-Cluster`.

- **Cluster Sizes**:
  ```python
  df_kn["KN-Cluster"].value_counts()