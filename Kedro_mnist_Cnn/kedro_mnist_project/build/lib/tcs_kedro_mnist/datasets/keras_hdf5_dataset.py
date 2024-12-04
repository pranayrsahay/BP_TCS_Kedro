import os
from kedro.io import AbstractDataset
from tensorflow.keras.models import load_model
import tensorflow as tf

# This is an extension of Kedro AbstractDataset to support H5 or Keras saved model data formats
class KerasHDF5Dataset(AbstractDataset):
    def __init__(self, filepath: str):
        self._filepath = filepath

    def _load(self):
        return load_model(self._filepath)

    def _save(self, model):
        model.save(self._filepath)

    def _exists(self):
        return os.path.exists(self._filepath)
    
    def _describe(self):
        return dict(param1=self._filepath)