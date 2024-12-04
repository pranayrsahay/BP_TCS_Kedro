import random
import logging
import pandas as pd

def predict_digit(classifier, save_X_test: pd.DataFrame, save_y_test: pd.Series):
    """
    Predict the digit for a randomly selected test sample using the provided classifier.

    This function randomly selects an index from the test dataset, retrieves the corresponding 
    features and actual label, and then uses the classifier model to make a prediction. 
    The actual and predicted values are logged.

    Parameters:
    ----------
    classifier : object
        A trained machine learning model (e.g., RandomForestClassifier) that has a `predict` method.
    
    save_X_test : pd.DataFrame
        A pandas DataFrame containing the feature data for the test set. The DataFrame should have
        the same structure and feature names as the data used to train the classifier.
    
    save_y_test : pd.Series
        A pandas Series containing the actual labels (targets) for the test set. The Series should
        have the same index as the corresponding rows in `save_X_test`.

    Returns:
    -------
    None
        This function does not return any value. It logs the actual and predicted values to the logger.

    Logs:
    -----
    The function logs the following information:
        - The index of the sample being predicted.
        - The actual label for the selected sample.
        - The predicted label from the classifier.
"""
    model = classifier
    
    # Get a random index
    predict_num = random.randint(0, len(save_X_test) - 1)  
    logger = logging.getLogger(__name__)
    
    logger.info(f"Predicting for index: {predict_num}")
    
    # Select the sample for prediction
    sample = save_X_test.iloc[[predict_num]]  

    # Making the prediction
    prediction = model.predict(sample)  

    # Extract actual and predicted values cleanly
    actual_value = save_y_test.iloc[predict_num].values[0]  
    predicted_value = prediction[0]  
    
    logger.info(f"The Actual is {actual_value} and prediction is {predicted_value}")  