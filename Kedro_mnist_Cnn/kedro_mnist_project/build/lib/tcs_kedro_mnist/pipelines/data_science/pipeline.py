"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import build_model, train_model, log_test_results, generate_reports 

# This pipelines sequentially calls all the nodes to build, compile, train and evaluate model performance
def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func= build_model,
            inputs= ["params:model_options", "x_train_normalized"],
            outputs = "model_architecture",
            name = "build_custom_model",
        ),
        node(
            func= train_model,
            inputs= ["x_train_normalized",  "y_train", "x_test_normalized", "y_test", "model_architecture", "params:model_options"],
            outputs = "best_trained_model",
            name = "train_model",
        ),
        node(
            func= log_test_results,
            inputs= ["x_test_normalized", "y_test", "best_trained_model"],
            outputs = ["predictions", "test_acc", "test_loss", "report", "cm"],
            name = "log_test_results"
        ),
        node(
            func= generate_reports,
            inputs= ["y_test", "predictions", "test_acc", "test_loss", "report", "cm"],
            outputs = None,
            name = "generate_reports",
        ),
    ])
