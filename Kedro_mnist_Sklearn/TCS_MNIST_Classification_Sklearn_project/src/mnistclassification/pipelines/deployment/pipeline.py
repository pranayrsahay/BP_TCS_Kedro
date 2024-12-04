"""
This is a boilerplate pipeline 'deployment'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import predict_digit


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=predict_digit,
            inputs=['classifier', 'save_X_test', 'save_y_test'],
            outputs=None,
            name="predict_digit_node",
        ),

    ])
