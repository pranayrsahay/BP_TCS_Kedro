"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.10
"""
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tcs_kedro_mnist.datasets.keras_hdf5_dataset import KerasHDF5Dataset
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from fpdf import FPDF
import os

logger = logging.getLogger(__name__)

# This function is responsible for building the model architecture and saves the uncompiled model at data/06_models in h5 format (Custom Keras dataset)
def build_model(parameters: dict, x_train_normalized):
    logger.info("Building Model")
    random_state = parameters["random_state"]
    filter_1 = parameters["filter_1"]
    filter_2 = parameters["filter_2"]
    filter_3 = parameters["filter_3"]
    kernel_size = tuple(parameters["kernel_size"])
    dense_size = parameters["dense_size"]
    dropout_rate = parameters["dropout_rate"]
    activation = parameters["activation"]

    if random_state is not None:
        np.random.seed(random_state)
        tf.random.set_seed(random_state)

    model_1 = models.Sequential()
    model_1.add(layers.Conv2D(filter_1, kernel_size, activation = activation, input_shape = (28, 28, 1)))
    model_1.add(layers.MaxPooling2D(2, 2))
    model_1.add(layers.Conv2D(filter_2, kernel_size, activation = activation))
    model_1.add(layers.MaxPooling2D(2, 2)) 
    model_1.add(layers.Conv2D(filter_3, kernel_size, activation = activation))
    model_1.add(layers.Flatten())
    model_1.add(layers.Dense(dense_size, activation = activation))
    if dropout_rate > 0:
        model_1.add(layers.Dropout(dropout_rate))
    model_1.add(layers.Dense(10, activation = "softmax"))
    logger.info("Successfully built the Model")

    return model_1

# This function compiles and trains the model and saves the best model based on the least validation loss
def train_model(x_train_normalized: np.ndarray, y_train: np.ndarray, x_test_normalized: np.ndarray, y_test: np.ndarray, model_architecture:KerasHDF5Dataset, parameters: dict):
    logger.info("Training the model with Normalized data")
    epochs = parameters["epochs"]
    loss_function = parameters["loss_function"]
    optimizer_v = parameters["optimizer"]
    model_1 = model_architecture
    model_1.compile(optimizer = optimizer_v, loss = loss_function, metrics = ["accuracy"])
    best_model_save_path = r'data\06_models\best_model.h5'
    checkpoint = ModelCheckpoint(best_model_save_path,  monitor='val_loss',  mode='min',  save_best_only=True, verbose=1)
    model_1.fit(x_train_normalized, y_train, epochs = epochs, batch_size = 64, validation_split = 0.1, callbacks = [checkpoint], verbose = 1)
    logger.info("Successfully trained the Model")
    best_trained_model = load_model(best_model_save_path)

    return best_trained_model

# This function evaluates the model performance on the test dataset and logs it on the console and also creates a test result log file
def log_test_results(x_test_normalized:np.ndarray, y_test:np.ndarray, best_trained_model):

    model = best_trained_model
    logger.info("Model loaded successfully")
    predicted_classes = model.predict(x_test_normalized, verbose=0)
    predictions = np.argmax(predicted_classes, axis = 1)
    test_loss, test_acc = model.evaluate(x_test_normalized, y_test, verbose=0)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)
    cm = confusion_matrix(y_test, predictions)

    logger.info(f"Model Performance:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Classification report:\n {report}")

    with open(f"data\\08_reporting\\test_result.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Classification report:\n {report}")
    
    return predictions, test_acc, test_loss, report, cm

# This function additionally generates a PDF format file with all the performance metrics and confusion matrix image 
def generate_reports(y_test, predictions, test_acc, test_loss, report, cm):
    logger.info("Generating custom report")
    plt.figure(figsize=(10, 8))
    labels = [str(i) for i in range(10)]
    sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('data\\07_model_output\\confusion_matrix.png')

    pdf = FPDF(orientation="L", unit="mm", format=(400, 300))
    pdf.set_auto_page_break(auto = True, margin = 15)
    pdf.add_page()

    pdf.set_font("Arial", size = 14)
    pdf.cell(200, 10, txt = "Model Performance Report", ln = True, align = 'C')

    pdf.ln(10)
    pdf.set_font("Arial", size = 12)
    pdf.cell(0, 10, f"Test Accuracy: {test_acc:.2f}", ln = True)
    pdf.cell(0, 10, f"Test Loss: {test_loss:.2f}", ln = True)

    pdf.ln(10)
    pdf.cell(0, 10,"Confusion Matrix:", ln = True)
    pdf.image('data\\07_model_output\\confusion_matrix.png', x = 10, y = None, w = 180)

    pdf.ln(10)
    pdf.set_font("Arial", size = 12)
    pdf.cell(0, 10, "Classification Report:", ln = True)
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            pdf.cell(0, 10, f"{label}: {metrics}", ln = True)

    output_path = 'data\\08_reporting\\model_performance_report.pdf'
    pdf.output(output_path)

    







    



    