import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import torch as tr
import torch.nn as nn
import torch.optim as optim
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

dfcsv = pd.read_csv("csv/Rental_Price_Prediction.csv")
dfcsv.index = dfcsv.index + 1
df = dfcsv.rename(columns={"Property_Area": "area",
                           "Bedrooms": "bed",
                           "Bathrooms": "bath",
                           "Property_Age": "age",
                           "Proximity_to_City_Center": "dist",
                           "Monthly_Rent": "rent"})

columns = ["area", "bed", "bath", "age", "dist", "rent"]

# print(df.info())
# column - null count - datatype 
# area - 1000 - float64     // Need to handle missing values in this colum
# bed - 0 - int64
# bath - 0 - int64
# age - 0 - int64
# dist - 0 - float64
# rent - 0 - float64

df_kn = df.copy()
df_kn[columns] = KNNImputer(n_neighbors=5).fit_transform(df_kn[columns])

print(df.info())          # // area : 9000 values
print(df_kn.info())       # // area : 10000 values

df_std = df_kn.copy()
df_std[columns] = StandardScaler().fit_transform(df_kn[columns])
print(df_std.head())

x = df_std.drop(columns=["rent"]).values
y = df_std["rent"].values
x_tensor = tr.tensor(x, dtype=tr.float32)
y_tensor = tr.tensor(y, dtype=tr.float32).view(-1, 1)
x_train, x_temp, y_train, y_temp = train_test_split(x_tensor, y_tensor, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # Hidden layer 1
        self.fc2 = nn.Linear(64, 32)         # Hidden layer 2
        self.fc3 = nn.Linear(32, 1)          # Output layer
    
    def forward(self, x):
        x = tr.relu(self.fc1(x))
        x = tr.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model
model = RegressionModel(input_dim=x_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 500

for epoch in range(num_epochs):
    model.train()
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 25 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

model.eval()
with tr.no_grad():
    y_val_pred = model(x_val)
    val_loss = criterion(y_val_pred, y_val)
    print(f"Validation Loss: {val_loss.item():.4f}")

with tr.no_grad():
    y_test_pred = model(x_test)
    test_loss = criterion(y_test_pred, y_test)
    print(f"Test Loss: {test_loss.item():.4f}")

# Actual vs Predicted Plot
plt.scatter(y_test, y_test_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel("Actual Rent")
plt.ylabel("Predicted Rent")
plt.title("Actual vs Predicted Rent")
plt.show()

mse = mean_squared_error(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")