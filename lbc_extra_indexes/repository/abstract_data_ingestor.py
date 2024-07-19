from abc import ABC, abstractmethod
import pandas as pd
from lbc_extra_indexes.utils.logger import logger


class DataIngestor(ABC):
    def __init__(self):
        self.__df = None

    @abstractmethod
    def load_data(self):
        pass

    @property
    def df(self) -> pd.DataFrame:
        return self.__df

    @df.setter
    def df(self, new_data: pd.DataFrame):
        new_len = new_data.shape[0]
        logger.info(f'[Data Ingestor]: New DataFrame has {new_len} rows.')
        self.__df = new_data

