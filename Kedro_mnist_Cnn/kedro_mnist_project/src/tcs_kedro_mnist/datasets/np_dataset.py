from kedro.io import AbstractDataset
import numpy as np

# This is an extension of Kedro AbstractDataset to support numpy data formats
class NumpyDataset(AbstractDataset):
    def __init__(self, filepath: str):
        self._filepath = filepath
    
    def _load(self):
        return np.load(self._filepath)

    def _save(self, data: np.ndarray):
        if not isinstance(data, np.ndarray):
            raise ValueError("Data is not of type numpy.ndarray")
        np.save(self._filepath, data)

    def exists(self):
        try:
            np.load(self._filepath)
            return True
        except FileNotFoundError:
            return False
        
    def _describe(self):
        return dict(param1=self._filepath)
        #return f'NumpyDataset (filepath = {self._filepath})'