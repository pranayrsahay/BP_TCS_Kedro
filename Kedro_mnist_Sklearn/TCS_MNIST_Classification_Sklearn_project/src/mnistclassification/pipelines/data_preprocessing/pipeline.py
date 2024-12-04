"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import preprocess_data


def create_pipeline(**kwargs) -> Pipeline:
    """
    Creates a pipeline for preprocessing the MNIST dataset.

    This pipeline consists of a single node that applies the `preprocess_data` function
    to the MNIST training and testing datasets, which are provided as inputs. The output of 
    this node is a preprocessed dataset ready for further analysis or modeling.

    Parameters:
    -----------
    **kwargs : dict
        Additional keyword arguments that can be passed to the pipeline creation process.
        These may include configuration settings, parameters, or other options required 
        by the pipeline.

    Returns:
    --------
    Pipeline
        An instance of `Pipeline` that defines the data preprocessing workflow.


    Notes:
    -------
    - The `preprocess_data` function is expected to be defined in the `nodes` module and 
      should be capable of handling the inputs specified in the pipeline.
    - The inputs "mnist_train" and "mnist_test" must be available in the Kedro catalog 
      as datasets that can be loaded during the pipeline execution.
    - The output "preprocessed_data" will be stored in the Kedro data catalog for 
      subsequent use in the project.
    """
    return pipeline([
        node(
                func=preprocess_data,
                inputs=["mnist_train", "mnist_test"],
                outputs="preprocessed_data",
                name="data_preprocessing_node",
            ),

    ])
