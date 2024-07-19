from abc import ABC, abstractmethod
import pandas as pd
from lbc_extra_indexes.utils.logger import logger
from lbc_extra_indexes.utils.values_container import ProjectValues


class AbstractDataProcessor(ABC):
    def __init__(self, df: pd.DataFrame):
        self.values = ProjectValues()
        self.__df = df

    def filter_df(self, column: str, value: list):
        self.df = self.df.loc[self.df[column].isin(value)]

    @staticmethod
    def logger_message(msg: str):
        logger.info(f'[Data Processor]: {msg}')

    @abstractmethod
    def binning_area(self):
        pass

    @abstractmethod
    def select_professionals(self):
        pass

    @abstractmethod
    def select_france_metro(self):
        pass

    @abstractmethod
    def select_houses_and_flats(self):
        pass

    @abstractmethod
    def select_sells(self):
        pass

    @abstractmethod
    def select_columns(self):
        pass

    @abstractmethod
    def remove_price_with_na(self):
        pass

    @abstractmethod
    def remove_properties_with_size_zero(self):
        pass

    @abstractmethod
    def compute_price_m2(self):
        pass

    @abstractmethod
    def remove_points_with_low_quality_coordinates(self):
        pass

    def execute(self):
        self.select_sells()
        self.select_professionals()
        self.select_houses_and_flats()
        self.remove_properties_with_size_zero()
        self.select_france_metro()
        self.remove_price_with_na()
        self.compute_price_m2()
        self.binning_area()
        self.remove_points_with_low_quality_coordinates()
        # df.rename(columns={'number_of_total_rooms': 'rooms', 'postal_code': 'zip_code'}, inplace=True)

    @property
    def df(self):
        return self.__df

    @df.setter
    def df(self, new_data: pd.DataFrame):
        new_len = new_data.shape[0]
        self.logger_message(f'New DataFrame has {new_len} rows.')
        self.__df = new_data
