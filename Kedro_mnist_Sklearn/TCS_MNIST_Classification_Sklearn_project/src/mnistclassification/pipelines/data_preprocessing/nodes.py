"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.10
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler

# def save_data(X: pd.DataFrame, y: pd.Series, output_dir: str):
#     """Save X and y as CSV files."""
#     # Create output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)

#     # Save X and y as CSV files
#     X.to_csv(os.path.join(output_dir, "X.csv"), index=False)
#     y.to_csv(os.path.join(output_dir, "y.csv"), index=False)

def preprocess_data(mnist_train: pd.DataFrame, mnist_test: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the MNIST training and testing datasets by performing the following steps:
    - Combines the training and testing datasets into a single DataFrame.
    - Separates the features (pixels) from the labels.
    - Removes any rows where the label is NaN.
    - Normalizes the pixel values to a range of [0, 1].
    - Reattaches the labels to the feature DataFrame.

    Parameters:
    -----------
    mnist_train : pd.DataFrame
        A pandas DataFrame containing the MNIST training dataset. 
        It is expected to have a column named 'label' indicating the digit (0-9) for each image.

    mnist_test : pd.DataFrame
        A pandas DataFrame containing the MNIST testing dataset. 
        It is also expected to have a column named 'label' indicating the digit (0-9) for each image.

    Returns:
    --------
    pd.DataFrame
        A pandas DataFrame containing the preprocessed features (pixel values) and labels 
        from both the training and testing datasets. The pixel values are normalized to the 
        range [0, 1], and the DataFrame includes a 'label' column with the corresponding digit 
        for each image.

    Notes:
    -------
    - It is assumed that both input DataFrames have the same structure and contain a 'label' column.
    - The pixel values are expected to be in the range [0, 255] prior to normalization.
    - The function discards any rows where the label is NaN, which may occur if the dataset is incomplete.
    """
    data = pd.concat([mnist_train, mnist_test], ignore_index=True)
    X = data.drop(columns = ['label'])
    y = data['label']
    X = X[~y.isna()]
    y = y[~y.isna()]
    X = X / 255.0
    X['label'] = y
    return X