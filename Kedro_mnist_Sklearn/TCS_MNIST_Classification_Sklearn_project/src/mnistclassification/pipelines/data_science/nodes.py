"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.10
"""
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns


def split_data(preprocessed_data: pd.DataFrame) -> tuple:
    """
    Splits the dataset into training and testing sets.

    This function takes a Pandas DataFrame containing the dataset and separates it into
    features (X) and labels (y). It then splits the data into training and testing sets
    using a stratified random split.

    Parameters:
    -----------
    data : pd.DataFrame
        A Pandas DataFrame containing the dataset with a column named 'label' for the target variable.

    Returns:
    --------
    tuple
        A tuple containing four elements:
        - X_train: pd.DataFrame, the training set features.
        - X_test: pd.DataFrame, the testing set features.
        - y_train: pd.Series, the training set labels.
        - y_test: pd.Series, the testing set labels.
    """
    y = preprocessed_data['label']
    X = preprocessed_data.drop(columns = ['label'])
    logger = logging.getLogger(__name__)
    # logger.info(y.head())
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.20, random_state=42)

    return X_train, X_test, X_val, y_train, y_test, y_val

def save_test_data(X_test:pd.DataFrame, y_test:pd.Series) -> tuple:
    """
    Save test data and convert the target labels to a DataFrame.

    This function takes feature data in the form of a DataFrame and target labels in the form of a 
    Series. It converts the target labels into a DataFrame, renaming the column to 'target'. 
    The function returns both the feature DataFrame and the target DataFrame.

    Parameters:
    ----------
    X_test : pd.DataFrame
        A pandas DataFrame containing the test features. Each row corresponds to a sample, and each 
        column corresponds to a feature.

    y_test : pd.Series
        A pandas Series containing the test target labels. Each element corresponds to the label 
        for the respective row in `X_test`.

    Returns:
    -------
    tuple
        A tuple containing:
        - save_X_test (pd.DataFrame): The input feature DataFrame (X_test) unchanged.
        - save_y_test (pd.DataFrame): A DataFrame with a single column named 'target' containing 
          the labels from y_test.
    """

    save_X_test = X_test
    y_test_df = y_test.to_frame(name='target')
    save_y_test = y_test_df
    return save_X_test, save_y_test

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Trains a Random Forest classifier on the training data.

    This function initializes a Random Forest classifier, fits it on the provided training
    features and labels, and returns the trained model.

    Parameters:
    -----------
    X_train : pd.DataFrame
        A Pandas DataFrame containing the training set features.

    y_train : pd.Series
        A Pandas Series containing the training set labels.

    Returns:
    --------
    RandomForestClassifier
        The trained Random Forest classifier.
    """
    classifier = RandomForestClassifier()

    classifier.fit(X_train, y_train)
    return classifier

# def save_model(classifier: RandomForestClassifier) -> None:
#     joblib.dump(classifier, "data/06_models/")

def evaluate_model(classifier: RandomForestClassifier, X_val: pd.DataFrame, y_val: pd.Series):
    """
    Evaluates the performance of a trained model on the validation data.

    This function uses the provided classifier to make predictions on the validation data,
    calculates the accuracy, logs the result, and generates a confusion matrix heatmap.

    Parameters:
    -----------
    classifier : RandomForestClassifier
        The trained Random Forest classifier to evaluate.

    X_val : pd.DataFrame
        A Pandas DataFrame containing the validation set features.

    y_val : pd.Series
        A Pandas Series containing the validation set labels.

    Returns:
    --------
    plt.Figure
        The matplotlib figure containing the confusion matrix heatmap.
    """
    y_pred = classifier.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    logger = logging.getLogger(__name__)
    logger.info(f'Model Performance Metrics:')
    logger.info(f"The Validation Accuracy is {acc}")
    cf_data = {"y_Actual": y_val, "y_Predicted": y_pred}
    df = pd.DataFrame(cf_data, columns=["y_Actual", "y_Predicted"])
    confusion_matrix = pd.crosstab(df["y_Actual"], df["y_Predicted"], rownames=["Actual"], colnames=["Predicted"])
    sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues')
    logger.info(f"Confusion matrix has been generated in folder \"08_reporting\" ")
    precision = precision_score(y_val, y_pred, average='weighted')  
    recall = recall_score(y_val, y_pred, average='weighted')
    f1 = f1_score(y_val, y_pred, average='weighted')
    logger.info(f'Precision: {precision:.4f}')
    logger.info(f'Recall: {recall:.4f}')
    logger.info(f'F1 Score: {f1:.4f}')
    return plt
