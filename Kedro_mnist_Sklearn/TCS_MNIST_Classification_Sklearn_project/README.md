
# MNIST Classification Project

## Table of Contents
- [Introduction](#intro)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Project](#running-the-project)


## Introduction

This repository contains a Kedro project for classifying handwritten digits using the MNIST dataset. The project is designed to demonstrate the use of Kedro for data science workflows. The primary objective of this project is to build a robust model that can accurately classify images of handwritten digits (0-9) based on pixel values. The dataset consists of 70,000 images of handwritten digits, each represented as a 28x28 pixel grayscale, stored in csv files.

### Key Features

- **Data Preprocessing:** The project includes data preprocessing steps to handle missing values, normalize pixel values, and split the dataset into training and testing sets.
- **Model Training:** The Random Forest Classifier is trained on the training dataset.
- **Model Evaluation:** The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score, ensuring a comprehensive understanding of its predictive capabilities.
- **Pipeline Management:** Using Kedro's pipeline management, the project ensures a clear and reproducible workflow, allowing easy modifications and experimentation with different model parameters or algorithms.



## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.10 
- pip (Python package installer)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/mnistclassification.git
   cd mnistclassification

2. **Create a virtual environment (optional but recommended): You can create a virtual environment using venv:**

    ```bash
    - On macOS/Linux:
        python -m venv .venv
        source .venv/bin/activate

    - On Windows:
        python -m venv .venv
        .venv\Scripts\
        
3. **Install the required packages: Install the project's dependencies using pip:**

    ```bash
        pip install -r requirements.txt
    
4. **Download MNIST data form the following link and store the cvs files in data/01_raw folder
    https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
    

## Running the Project

    To run the Kedro project, use the following commands:
        1. Run the Kedro pipeline: You can run the default pipeline with:
            ``` bash
                kedro run
        2. Run specific nodes or pipelines: If you want to run a specific pipeline or node, you can do so using:
            ``` bash
                kedro run --pipeline <pipeline_name>
        3. To classify handwritten digits with the MNIST dataset:
            - Ensure the dataset is available in the data/01_raw/ directory.
                - dataset names: 
                    - mnist_train
                    - mnist_test

