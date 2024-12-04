"""
This is a boilerplate pipeline 'deployment'
generated using Kedro 0.19.10
"""
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

# This function predicts the output based on user defined test set Index input. 
# Optionally, user can define the index in parameters_deployment.yml
def predict_user_defined(x_test_normalized, y_test, best_trained_model, parameters: dict):
    model = best_trained_model
    #user_defined_index = parameters["index"]
    user_defined_index = int(input("Enter the Index of test set to be predicted:\n"))
    x_test_normalized_w_batch = np.expand_dims(x_test_normalized[user_defined_index], axis = 0)
    prediction = model.predict(x_test_normalized_w_batch)
    predicted_class = np.argmax(prediction)
    print(f"Model output: {predicted_class}")
    print(f"Actual output : {y_test[user_defined_index]}")


