# TCS_Kedro_MNIST

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Overview

The project is an implementation of a multiclass classification model based on CNN architecture using MNIST dataset. The project is based on Kedro' Python Framework (Generated using `kedro 0.19.10`)

## Project deployment

Please follow the steps to deploy the project on your local environment 

* Download the entire project using git clone (or download it as ZIP file and unzip it)
* The project needs Python 3.10.0 version. Please create a virtual environment using the following command:
	```
	python -m venv <env_name>
	```
	e.g. python -m venv MNIST_Kedro_Project

	Once the environment is created, activate it using the following command:
	```
	activate <env_name> 
	```
	e.g. activate MNIST_Kedro_Project

	Incase of conda, use the following command:

	```
	conda create -n <env_name> python==3.10.0
	```
	e.g. conda create -n MNIST_Kedro_Project python==3.10.0

	```
	conda activate <env_name>
	```
	e.g. conda activate MNIST_Kedro_Project

## How to install dependencies

The required dependencies are listed in `requirements.txt` 

To install them, navigate to the project folder and run within the newly create environment: 

```
pip install -r requirements.txt
```

## Project Overview

The CNN architecture used to train the model has  three 2D convolution layers with 32, 64 and 64 filters respectively with a kernel size of (3x3). Each Convolution layer is followed by 2D max pool layer with an optional dropout layer. The two dense layer consists of 64 and 10 nodes respectively. The loss function used is sparse categorical cross entropy.
Note: All these hyperparameters can be configured using a configuration file located at conf/parameters.yml file

To store and handle the training, test and normalized data, there is a custom numpy dataset which extends Kedro's AbstractDataSet. There is another custom dataset to store and handle keras model files. The dataset extension classes is located at src/tcs_kedro_mnist/datasets

## Project Pipeline Overview

The project implements 3 Kedro pipelines for the overall ML implementation

* Data Processing pipeline (data_processing): This pipeline is responsible for downloading the MNIST dataset and performs the basic Data Pre-Processing. The pipeline implements 2 nodes:
	
	1. load_mnist_data: This node download the MNIST dataset using TensorFlow keras function. The raw data is saved in data/01_raw folder
	2. preprocess: This node does the preprocessing (Normalization) of MNIST dataset. The preprocess node takes x_train and x_test (grayscale images) as input and performs max-min normalization along with reshaping (Adding an extra channel dimension). The normalized data is saved in data/05_model_input folder and test data is saved in data/03_primary folder

* Data Science pipeline (data_science): This pipeline is responsible for building the model, training the model, testing and generating the report. The pipeline implements the following nodes:

	1. build_model: This node is responsible for building the CNN model architecture using the inputs from conf/parameters_data_science.yml. It returns a keras model object, i.e. the uncompiled model. The uncompiled model is saved at data\06_models\model_architecture.h5
	2. train_model: This node takes the uncompiled model as as input along with x_train_normalized and y_train, and compiles the model using the loss fucntion and optimizer information from conf\data_science.yml. The model is trained with a checkpoint which saves the best model considering the minimum validation loss across all epochs. The best model is saved at data\06_models\best_model.h5
	3. log_test_results: This node takes the best trained model and log the performance of the model on the testset, i.e. x_test_normalized and y_test using sckit-learn libraries and tesnorflow's model evaluate. The results are logged on the console as well as in a log file that is store at data\08_reporting\test_result.txt. The confusion matrix plot is saved at data\07_model_output\confusion_matrix.png
	4. generate_reports: This node outputs a comprehensive view of the model performance by logging various model performance metrics in a pdf format which is easy to share and view

* Deployment Pipeline (deployment): This pipeline loads the best trained model and predicts the output based on user provided input index of the test set.

	1. predict_user_defined: This nodes takes the best trained model, x_test_normalized (normalized test data) as inputs. The user is prompted to provide the testset index that is to be predicted. The model does the inference on the given test set index and returns the output of the model on the console. Optionally, a user can define the test index to be predicted in conf\deployment.yml  and proceed with the inference part.

## Project Execution

To run the entire project, execute the following command to run all the 3 pipeline in sequence

```
kedro run
```
To run a specific pipeline, execute the following command:

```
kedro run --pipeline <pipeline_name>
```
e.g. kedro run --pipeline data_science


## Project Packaging 

For making a Kedro package, update the "pyproject.toml" file under the root folder with all the dependencies

Execute the following command"

```
kedro package
```
The command will create the .whl file along with a zipped conf file. The .whl file can be easily installed on any other system by executing the following command:

```
pip install  <package_name>.whl
```
Please refer to the https://www.github.com/pranayrsahay/Kedro_MNIST for more details on environment setting and other dependencies
