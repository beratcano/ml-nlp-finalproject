# Model Training Notes

## Overview
This notebook focuses on training a neural network model to predict rental prices based on features such as property area, number of bedrooms, bathrooms, property age, and proximity to the city center. The dataset has been preprocessed using KNN imputation for missing values, and standard scaling to normalize the data.

## Libraries Used
- **Pandas** for data manipulation
- **NumPy** for numerical operations
- **Matplotlib & Seaborn** for data visualization
- **PyTorch** for building and training the neural network
- **Scikit-learn** for data preprocessing, imputation, and splitting the dataset

## Data Preprocessing
1. **Handling Missing Values**:
   - KNN imputation was used to fill missing values in the dataset.

2. **Feature Scaling**:
   - The dataset was scaled using `StandardScaler` to normalize the features before training.

3. **Train-Test Split**:
   - The dataset was split into training (70%), validation (15%), and test (15%) sets using `train_test_split`.

## Model Architecture
The model is a feed-forward neural network with the following layers:
- **Input layer**: Matches the number of input features.
- **Hidden Layer 1**: 64 neurons with ReLU activation.
- **Hidden Layer 2**: 32 neurons with ReLU activation.
- **Output Layer**: 1 neuron (since it's a regression task).

### Loss Function
- **Mean Squared Error (MSE)** was used as the loss function.

### Optimizer
- **Adam Optimizer** was used with different learning rates during training.

---

## Experimentation with Epochs & Learning Rates

### Epoch 100 Results:
- **Loss**: 0.9921
- **Validation Loss**: 0.9981
- **Test Loss**: 1.0144
- **Mean Squared Error (MSE)**: 1.0144
- **Mean Absolute Error (MAE)**: 0.8721

#### Graph: Actual vs Predicted Rent (Epoch 100)

![Epoch 100](file-UuPMqLGi1rVCTGP4tRQ4Yh)

---

### Epoch 500 Results:
- **Loss**: 0.9108
- **Validation Loss**: 1.0711
- **Test Loss**: 1.0877
- **Mean Squared Error (MSE)**: 1.0877
- **Mean Absolute Error (MAE)**: 0.8896

#### Graph: Actual vs Predicted Rent (Epoch 500)

![Epoch 500](file-CUghzaSGTdXfjQW7gjsKA4)

---

### Epoch 1000 Results:
- **Loss**: 0.8712
- **Validation Loss**: 1.0858
- **Test Loss**: 1.1049
- **Mean Squared Error (MSE)**: 1.1049
- **Mean Absolute Error (MAE)**: 0.8982

#### Graph: Actual vs Predicted Rent (Epoch 1000)

![Epoch 1000](file-FViEx1dKDM8Ax5eYvTGV9i)

---

### Learning Rate Experimentation

#### Constant Learning Rate (0.01):

The constant learning rate was set to `0.01` after initial experiments with decay schedulers.

##### Epoch 500 Results with Learning Rate 0.01:
- **Loss**: 0.8670
- **Validation Loss**: 1.1245
- **Test Loss**: 1.1495
- **Mean Squared Error (MSE)**: 1.1495
- **Mean Absolute Error (MAE)**: 0.9089

#### Graph: Actual vs Predicted Rent (Epoch 500, LR 0.01)

![Epoch 500, LR 0.01](file-FViEx1dKDM8Ax5eYvTGV9i)

---

### Learning Rate Scheduling

We experimented with various learning rate schedules, including:

1. **StepLR Scheduler**: The learning rate was decayed by a factor of `0.1` every 10 epochs. However, this led to the learning rate becoming too small after a few epochs, hindering model training.

2. **Final Learning Rate**: During the last epoch (500th), the learning rate became extremely small, as observed in the output where the learning rate was `1.0000000000000026e-52`. This caused the optimizer to make almost no updates to the model weights, resulting in the loss stagnating.

##### Fixed Learning Rate Strategy:
After experimentation, we settled on using a **constant learning rate** of `0.01` for the remaining epochs, leading to more stable results.

---

### Key Observations:
1. **Learning Rate and Model Convergence**: Using a constant learning rate of `0.01` seemed to provide a more stable training process, although the model's performance (especially test loss) could be further optimized.
   
2. **Epochs and Model Performance**: 
   - **Epoch 100**: Still significant room for improvement (higher loss and error).
   - **Epoch 500**: A reasonable trade-off between model performance and training time.
   - **Epoch 1000**: Similar to epoch 500 in terms of results but with slightly higher loss.

3. **Overfitting Concerns**: Given the decrease in validation/test loss compared to training loss at higher epochs, it seems the model may have started overfitting. Regularization techniques like dropout or early stopping could help mitigate this.