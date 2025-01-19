import pandas as pd

#This method must load a csv file and load it into ski-learn
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data 

#This method must return the data in a format that can be used by the model
def clean_data(data):
    data = data.dropna()    
    # Remove the name column as it's not relevant for prediction
    data = data.drop(columns=['Unnamed: 0'])
    # Convert the Private column to a binary column
    data['Private'] = data['Private'].map({'Yes': 1, 'No': 0})
    return data

#This method must split the data into a training set and a test set
def split_data(data):
    from sklearn.model_selection import train_test_split
    # Using 'Apps' as the target variable 
    X = data.drop(columns=['Apps'])
    X = X.drop(columns=['Accept'])
    X = X.drop(columns=['Enroll'])
    y = data['Apps']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

#This method must train the model
def train_model(X, y):
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(criterion='absolute_error', random_state=0, n_estimators=500)
    model.fit(X, y)
    return model

#This method must return the accuracy of the model
def model_accuracy(model, X, y):
    return model.score(X, y)

#This method must save the model to disk
def save_model(model, file_path):
    import joblib
    joblib.dump(model, file_path)

#This method must load the model from disk
def load_model(file_path):
    import joblib
    return joblib.load(file_path)

#This method must return a prediction
def predict(model, X, y=None):
    predictions = model.predict(X)
    if y is not None:
        # Create a DataFrame with actual and predicted values side by side
        comparison = pd.DataFrame({
            'Actual': y,
            'Predicted': predictions,
            'Difference': y - predictions
        })
        return comparison
    return predictions

#This method must return the probability of a prediction
def predict_probability(model, X):
    return model.predict_proba(X)

#This method must return the feature importances
def feature_importances(model):
    # Get feature names from the model
    feature_names = model.feature_names_in_
    # Get importance scores
    importances = model.feature_importances_
    # Create a dictionary mapping features to their importance scores
    feature_importance_dict = dict(zip(feature_names, importances))
    # Sort by importance score in descending order
    sorted_features = dict(sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True))
    return sorted_features

#This method must return the confusion matrix
def confusion_matrix(model, X, y):
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y, model.predict(X))

#This method must return the classification report
def classification_report(model, X, y):
    from sklearn.metrics import classification_report
    return classification_report(y, model.predict(X))

#This method must return the ROC curve
def roc_curve(model, X, y):
    from sklearn.metrics import roc_curve
    return roc_curve(y, model.predict(X))

#This method must return the AUC
def auc(model, X, y):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, model.predict(X))

#This method must return the precision-recall curve
def precision_recall_curve(model, X, y):
    from sklearn.metrics import precision_recall_curve
    return precision_recall_curve(y, model.predict(X))

#This method must return the average precision
def average_precision(model, X, y):
    from sklearn.metrics import average_precision_score
    return average_precision_score(y, model.predict(X))

#This method must return the F1 score
def f1_score(model, X, y):
    from sklearn.metrics import f1_score
    return f1_score(y, model.predict(X), average='weighted')

#This method must return the Matthews correlation coefficient
def matthews_corrcoef(model, X, y):
    from sklearn.metrics import matthews_corrcoef
    return matthews_corrcoef(y, model.predict(X))

#This method must return the log loss
def log_loss(model, X, y):
    from sklearn.metrics import log_loss
    return log_loss(y, model.predict(X))

#This method must return the Brier score
def brier_score(model, X, y):
    from sklearn.metrics import brier_score_loss
    return brier_score_loss(y, model.predict(X))

#This method must return the zero one loss
def zero_one_loss(model, X, y):
    from sklearn.metrics import zero_one_loss
    return zero_one_loss(y, model.predict(X))

#This method must return the balanced accuracy
def balanced_accuracy(model, X, y):
    from sklearn.metrics import balanced_accuracy_score
    return balanced_accuracy_score(y, model.predict(X))

#This method must return the Cohen's Kappa
def cohen_kappa(model, X, y):
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(y, model.predict(X))


#Make a method that show in a plot the distance from the real value to the predicted value
def plot_difference(model, X, y):
    import matplotlib.pyplot as plt
    import numpy as np
    predictions = model.predict(X)
    
    # Plot actual values in blue and predicted values in red
    plt.scatter(np.arange(len(y)), y, color='blue', label='Actual', alpha=0.5)
    plt.scatter(np.arange(len(predictions)), predictions, color='red', label='Predicted', alpha=0.5)
    
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    plt.show()

#Write a main method that calls the methods above
if __name__ == "__main__":
    data = load_data('College.csv')
    data = clean_data(data)
    X_train, X_test, y_train, y_test = split_data(data)
    model = train_model(X_train, y_train)
    accuracy = model_accuracy(model, X_test, y_test)
    print(accuracy)
    save_model(model, 'model.pkl')
    model = load_model('model.pkl')
    print(predict(model, X_test, y_test))
    #print(predict_probability(model, X_test))
    print(feature_importances(model))
    print(model.score(X=X_test, y=y_test))
    #plot_difference(model, X_test, y_test)

    # Calculate and plot the RMSE
    from sklearn.metrics import mean_squared_error
    import numpy as np
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(rmse)

    # Calculate r^2
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, predictions)
    print(r2)


    #plot_difference(model, X_test, y_test)
    #print(confusion_matrix(model, X_test, y_test))
    #print(classification_report(model, X_test, y_test))
    #print(roc_curve(model, X_test, y_test))
    #print(auc(model, X_test, y_test))
    #print(precision_recall_curve(model, X_test, y_test))
    #print(average_precision(model, X_test, y_test))
    #print(f1_score(model, X_test, y_test))
    #print(matthews_corrcoef(y_test, model.predict(X_test)))
   #print(log_loss(model, X_test, y_test))
    #print(brier_score(model, X_test, y_test))
    #print(zero_one_loss(X_test, y_test))
   #print(balanced_accuracy(model, X_test, y_test))
    #print(cohen_kappa(y_test, model.predict(X_test)))