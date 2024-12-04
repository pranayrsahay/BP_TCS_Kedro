"""
This is a boilerplate pipeline 'deployment'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import predict_user_defined

# This pipeline is meant for predicting the output based on user defined Test input Index
def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=predict_user_defined,
            inputs = ["x_test_normalized", "best_trained_model", "params:user_test_index"],
            outputs=None,
            name = "predict"
        ),
    ])
