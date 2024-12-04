"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, pipeline, node
from kedro.io import MemoryDataset
from .nodes import load_mnist_data, preprocess

# The pipleline first executes load_mnist_data node followed by preprocess node
def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=load_mnist_data,
            inputs=[],
            outputs = ["x_train", "y_train", "x_test", "y_test"],
            name = "load_mnist_data",
        ),
        node(
            func=preprocess, 
            inputs = ["x_train", "x_test"],
            outputs = ["x_train_normalized", "x_test_normalized"],
            name = "preprocess"
        )
    ])
