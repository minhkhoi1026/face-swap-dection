from abc import ABC, abstractmethod 
import pandas as pd

class BaseDetector(ABC):
    @abstractmethod
    def predict(self, video_bytes: bytes, *args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError()
