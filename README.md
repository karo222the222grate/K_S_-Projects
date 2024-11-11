
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input

# Step 1: Input Data and Target
csv_file = r'C:\Users\karo\Desktop\datas\Higher res.csv'  # Replace with your file path
data = pd.read_csv(csv_file)

samples = data['sample'].values
thd_values = data['THD'].values

# Step 2: Set the Window Size (Rolling Window Length)
window_size = 20  # Number of previous samples to include in each input

# Step 3: Create Rolling Window Inputs and Targets
def create_rolling_window(X, y, window_size):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:i+window_size])
        ys.append(y[i+window_size])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_rolling_window(samples, thd_values, window_size)

# Step 4: Determine the Number of Input Size, N
N = X_seq.shape[1]  # The number of inputs per sequence

# Step 5: Setting Percentage of Training and Testing Data
split_index = int(0.7 * len(X_seq))
X_train, X_test = X_seq[:split_index], X_seq[split_index:]
y_train, y_test = y_seq[:split_index], y_seq[split_index:]

# Step 6: Scale Data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
X_test_scaled = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1]))
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Step 7: Set the ANN-GMDH Parameters
nMax = 20  # Number of neurons on a layer
maxL = 6   # Maximum number of layers
alpha = 0.0003  # Selection pressure

# Step 8: Create and Train the ANN-GMDH Model
def build_gmdh_ann(input_shape, nMax, maxL, alpha):
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    for layer in range(maxL):
        model.add(Dense(nMax, activation='relu'))
        model.add(Dropout(alpha))
        
        y_pred_train = model.predict(X_train_scaled)
        
        # Ensure the prediction output shape matches the target shape
        if y_pred_train.shape[1] != 1:
            y_pred_train = y_pred_train[:, :1]  # Truncate to single output if needed
            
        rmse_train = np.sqrt(mean_squared_error(y_train_scaled, y_pred_train))

        if rmse_train > alpha:  # Simple elimination based on RMSE threshold
            model.pop()  # Remove last added layer/neurons
    
        if layer + 1 >= maxL or model.output_shape[1] <= 1:
            break
    
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

model = build_gmdh_ann((N,), nMax, maxL, alpha)

# Step 12: Train the Final Model
history = model.fit(X_train_scaled, y_train_scaled, epochs=2000, validation_data=(X_test_scaled, y_test_scaled), verbose=1)

# Step 13: Make Predictions
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# Step 14: Inverse Transform Predictions
y_pred_train_inv = scaler_y.inverse_transform(y_pred_train)
y_pred_test_inv = scaler_y.inverse_transform(y_pred_test)
y_train_inv = scaler_y.inverse_transform(y_train_scaled.reshape(-1, 1))
y_test_inv = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1))

# Step 15: Evaluate the Model
rmse_train = np.sqrt(mean_squared_error(y_train_inv, y_pred_train_inv))
rmse_test = np.sqrt(mean_squared_error(y_test_inv, y_pred_test_inv))
print(f'Training RMSE: {rmse_train}, Testing RMSE: {rmse_test}')

# Step 16: Plot the Results
plt.figure(figsize=(14, 7))
plt.plot(samples[window_size:split_index+window_size], y_train_inv, 'b', label='Actual Train')
plt.plot(samples[window_size:split_index+window_size], y_pred_train_inv, 'r', label='Predicted Train')
plt.xlabel('Samples')
plt.ylabel('THDI [%]')
plt.title('THD Prediction Using ANN-GMDH Model')
plt.legend()
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(samples[split_index+window_size:], y_test_inv, 'g', label='Actual Test')
plt.plot(samples[split_index+window_size:], y_pred_test_inv, 'orange', label='Predicted Test')
plt.xlabel('Samples')
plt.ylabel('THDI [%]')
plt.title('THD Prediction Using ANN-GMDH Model - Test Data')
plt.legend()
plt.show()

# Step 17: Error over Samples Plot
errors_test = y_test_inv.flatten() - y_pred_test_inv.flatten()

plt.figure(figsize=(12, 6))
plt.plot(range(len(errors_test)), errors_test, label='Errors')
plt.title(f'MSE = {mean_squared_error(y_test_inv, y_pred_test_inv):.10f}, RMSE = {np.sqrt(mean_squared_error(y_test_inv, y_pred_test_inv)):.6f}')
plt.xlabel('Samples')
plt.ylabel('Errors')
plt.show()

# Step 18: Train and Test Regression Plots
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_train_inv, y_pred_train_inv, label='Data')
plt.plot([min(y_train_inv), max(y_train_inv)], [min(y_train_inv), max(y_train_inv)], color='r', label='Y = T')
plt.title(f'Train: R={r2_score(y_train_inv, y_pred_train_inv):.5f}')
plt.xlabel('Measured THDI')
plt.ylabel('Predicted THDI')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test_inv, y_pred_test_inv, label='Data')
plt.plot([min(y_test_inv), max(y_test_inv)], [min(y_test_inv), max(y_test_inv)], color='r', label='Y = T')
plt.title(f'Test: R={r2_score(y_test_inv, y_pred_test_inv):.5f}')
plt.xlabel('Measured THDI')
plt.ylabel('Predicted THDI')
plt.legend()

plt.tight_layout()
plt.show()
