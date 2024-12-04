"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.10
"""


from kedro.pipeline import Pipeline, pipeline, node
from .nodes import evaluate_model, split_data, train_model, save_test_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=split_data,
                inputs=["preprocessed_data"],
                outputs=["X_train", "X_test", "X_val", "y_train", "y_test", "y_val"],
                name="split_data_node",
            ),
        node(
                func=save_test_data,
                inputs=["X_test", "y_test"],
                outputs=["save_X_test", "save_y_test"],
                name="save_test_data_node",
            ),    
        node(
            func=train_model,
            inputs=["X_train", "y_train"],
            outputs="classifier",
            name="train_model_node",
        ),
        # node(
        #     func=save_model,
        #     inputs=["classifier"],
        #     outputs=None,
        #     name="save_model_node"
        # ),
        node(
            func=evaluate_model,
            inputs=["classifier", "X_test", "y_test"],
            outputs = "confusion_matrix",
            name="evaluate_model_node",
        ),
    ])
