import pandas as pd



df = pd.read_csv('College.csv')

#print(df['Unnamed: 0'])

grouped = df.groupby('Private')



dfs = {category: group for category, group in grouped}


dfs['Yes'] = dfs['Yes'].drop(['Unnamed: 0','Private'],axis=1)
dfs['No'] = dfs['No'].drop(['Unnamed: 0','Private'],axis=1) 




from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

rf_model = RandomForestRegressor(
    n_estimators=1000,      # Number of trees
    max_features=11,        # Equivalent to mtry in R
    bootstrap=True,         # Equivalent to replace=True
    min_samples_leaf=1,     # Equivalent to nodesize=1
    n_jobs=-1,              # Use all CPU cores
    random_state=42         # Reproducibility
)


dataset = dfs['Yes'].drop(columns=["Accept","Enroll"])

train_data, test_data = train_test_split(dataset, train_size=0.3, random_state=42)

rf_model.fit(train_data.drop(columns=["Apps"]), train_data["Apps"])

# Predictions
train_predictions = rf_model.predict(train_data.drop(columns=["Apps"]))
test_predictions = rf_model.predict(test_data.drop(columns=["Apps"]))

# Compute Mean Squared Errors (MSE)
train_mse = mean_squared_error(train_data["Apps"], train_predictions)
test_mse = mean_squared_error(test_data["Apps"], test_predictions)

# Compute % Variance Explained
train_r2 = r2_score(train_data["Apps"], train_predictions) * 100  # Convert to %
test_r2 = r2_score(test_data["Apps"], test_predictions) * 100

# Print Report
print("Random Forest Regression Report, Private School: Yes")
print("=" * 40)
print(f"Number of trees: {rf_model.n_estimators}")
print(f"Number of variables tried at each split: {rf_model.max_features}")
print("\nTraining Performance:")
print(f"  Mean Squared Error: {train_mse:.2f}")
print(f"  % Variance Explained: {train_r2:.2f}%")
print("\nTesting Performance:")
print(f"  Mean Squared Error: {test_mse:.2f}")
print(f"  % Variance Explained: {test_r2:.2f}%")
print("=" * 40)





rf_model = RandomForestRegressor(
    n_estimators=1000,      # Number of trees
    max_features=11,        # Equivalent to mtry in R
    bootstrap=True,         # Equivalent to replace=True
    min_samples_leaf=1,     # Equivalent to nodesize=1
    n_jobs=-1,              # Use all CPU cores
    random_state=42         # Reproducibility
)


dataset = dfs['No'].drop(columns=["Accept","Enroll"])


train_data, test_data = train_test_split(dataset, train_size=0.8, random_state=42)

rf_model.fit(train_data.drop(columns=["Apps"]), train_data["Apps"])

# Predictions
train_predictions = rf_model.predict(train_data.drop(columns=["Apps"]))
test_predictions = rf_model.predict(test_data.drop(columns=["Apps"]))

# Compute Mean Squared Errors (MSE)
train_mse = mean_squared_error(train_data["Apps"], train_predictions)
test_mse = mean_squared_error(test_data["Apps"], test_predictions)

# Compute % Variance Explained
train_r2 = r2_score(train_data["Apps"], train_predictions) * 100  # Convert to %
test_r2 = r2_score(test_data["Apps"], test_predictions) * 100

# Print Report
print("Random Forest Regression Report, Private School: No")
print("=" * 40)
print(f"Number of trees: {rf_model.n_estimators}")
print(f"Number of variables tried at each split: {rf_model.max_features}")
print("\nTraining Performance:")
print(f"  Mean Squared Error: {train_mse:.2f}")
print(f"  % Variance Explained: {train_r2:.2f}%")
print("\nTesting Performance:")
print(f"  Mean Squared Error: {test_mse:.2f}")
print(f"  % Variance Explained: {test_r2:.2f}%")
print("=" * 40)


rf_model = RandomForestRegressor(
    n_estimators=1000,      # Number of trees
    max_features=11,        # Equivalent to mtry in R
    bootstrap=True,         # Equivalent to replace=True
    min_samples_leaf=1,     # Equivalent to nodesize=1
    n_jobs=-1,              # Use all CPU cores
    random_state=42         # Reproducibility
)


dataset = df.drop(columns=["Accept","Enroll",'Private','Unnamed: 0'])


train_data, test_data = train_test_split(dataset, train_size=0.2, random_state=42)

rf_model.fit(train_data.drop(columns=["Apps"]), train_data["Apps"])

# Predictions
train_predictions = rf_model.predict(train_data.drop(columns=["Apps"]))
test_predictions = rf_model.predict(test_data.drop(columns=["Apps"]))

# Compute Mean Squared Errors (MSE)
train_mse = mean_squared_error(train_data["Apps"], train_predictions)
test_mse = mean_squared_error(test_data["Apps"], test_predictions)

# Compute % Variance Explained
train_r2 = r2_score(train_data["Apps"], train_predictions) * 100  # Convert to %
test_r2 = r2_score(test_data["Apps"], test_predictions) * 100

# Print Report
print("Random Forest Regression Report, All schools")
print("=" * 40)
print(f"Number of trees: {rf_model.n_estimators}")
print(f"Number of variables tried at each split: {rf_model.max_features}")
print("\nTraining Performance:")
print(f"  Mean Squared Error: {train_mse:.2f}")
print(f"  % Variance Explained: {train_r2:.2f}%")
print("\nTesting Performance:")
print(f"  Mean Squared Error: {test_mse:.2f}")
print(f"  % Variance Explained: {test_r2:.2f}%")
print("=" * 40)

