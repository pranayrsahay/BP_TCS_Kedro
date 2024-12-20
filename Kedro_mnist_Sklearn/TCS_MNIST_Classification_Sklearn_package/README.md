# TCS_Kedro_MNIST

## Overview

This README file provides the steps to install and deploy the project using the `.whl` file generated by the Kedro packaging command. This project is a Kedro pipeline designed for MNIST digit classification.

## Prerequisites

Before installing and running the project, ensure you have the following:

1.  **Project Files:**
    *   A `.whl` file (the Python distribution package for the project).
    *   A `data` directory containing the necessary data (in this case, likely MNIST image data).
    *   A `conf-<package_name>.tar.gz` file containing the packaged configuration.
    *   A `requirements.txt` file listing the project's dependencies.

    **Important:** Create a new, dedicated folder and copy all these files into it or use the downloaded folder as it is.

2.  **Python Environment:**
    *   **Python Version:** The project requires Python 3.10.0.
    *   **Virtual Environment:** It is highly recommended to create a virtual environment to isolate the project's dependencies.

    **Creating a Virtual Environment:**

    *   **Using `venv` (recommended):**
        ```bash
        python -m venv <env_name>
        # Example:
        python -m venv MNIST_Kedro
        ```
    *   **Using `conda`:**
        ```bash
        conda create -n <env_name> python=3.10
        # Example:
        conda create -n MNIST_Kedro python=3.10
        ```

    **Activating the Virtual Environment:**

    *   **`venv`:**
        *   **Windows:**
            ```bash
            <env_name>\Scripts\activate
            # Example:
            MNIST_Kedro\Scripts\activate
            ```
        *   **Linux/macOS:**
            ```bash
            source <env_name>/bin/activate
            # Example:
            source MNIST_Kedro/bin/activate
            ```
    *   **`conda`:**
        ```bash
        conda activate <env_name>
        # Example:
        conda activate MNIST_Kedro
        ```

3.  **Installing Dependencies:**

    Once the virtual environment is activated, install the project's dependencies using the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

    This command will download and install all the necessary Python packages listed in the `requirements.txt` file into your virtual environment.

4. **Download MNIST data form the following link and store the cvs files in data/01_raw folder

   https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

## Installation

    Install the Kedro project from the `.whl` file using `pip`:

    ```bash
    pip install <package_name>.whl
    # Example:
    pip install mnistclassification-0.1-py3-none-any.whl

    ```

    This will install the kedro package.

## Running and Execution
    To execute the entire pipeline, use the following command:

    ```bash
    python -m <project_name>
    e.g. python -m mnistclassification 
```

    To run a specific pipeline, use the following command:

    ```bash
    python -m mnistclassification --pipeline <pipeline_name>
    e.g. python -m tcs_kedro_mnist --pipeline data_preprocessing
    ```
