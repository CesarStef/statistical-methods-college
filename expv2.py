from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd


dataset = pd.read_csv('College.csv')

# Ensure the dataset is in DataFrame format and correctly split
X = dataset.drop(columns=["Apps",'Private','Unnamed: 0','Accept','Enroll'])  # All predictor variables
y = dataset["Apps"]                 # Target variable

# Equivalent Train/Test split to R
train_index = np.random.choice(len(dataset), size=int(0.8 * len(dataset)), replace=False)
test_index = np.setdiff1d(np.arange(len(dataset)), train_index)

X_train, X_test = X.iloc[train_index], X.iloc[test_index]
y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Define the Random Forest model (matching R parameters)
rf_model = RandomForestRegressor(
    n_estimators=1000,  # Equivalent to ntree=1000
    max_features=11,  # Equivalent to mtry in R (default in R is sqrt(n_features))
    min_samples_leaf=1,  # Equivalent to nodesize=1
    bootstrap=True,  # Equivalent to replace=True
    n_jobs=-1,  # Use all available CPU cores
    random_state=42,  # Ensures reproducibility
    oob_score=False  # OOB proximity not used (R uses oob.prox=FALSE)
)

# Train the model
rf_model.fit(X_train, y_train)

# Predictions on training and test sets
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# Compute Performance Metrics
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred) * 100  # Convert to %
test_r2 = r2_score(y_test, y_test_pred) * 100

# Print results similar to R's output
print("Random Forest Regression Report (Python)")
print("=" * 50)
print(f"Number of trees: {rf_model.n_estimators}")
print(f"Number of variables tried at each split: sqrt(n_features)")
print("\nTraining Performance:")
print(f"  Mean Squared Error: {train_mse:.2f}")
print(f"  % Variance Explained: {train_r2:.2f}%")
print("\nTesting Performance:")
print(f"  Mean Squared Error: {test_mse:.2f}")
print(f"  % Variance Explained: {test_r2:.2f}%")
print("=" * 50)



