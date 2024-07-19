import numpy as np
import pandas as pd
from lbc_extra_indexes.utils.values_container import ProjectValues
from lbc_extra_indexes.utils.logger import logger


class PriceIndexExtractor:
    NAME = 'Price Index Extractor'
    NODES_ROOMS = {'1': [0, 1], '2': [1, 2], '3': [2, 3], '4': [3, 4], '5_plus': [4, np.Inf]}
    NODES_SIZE = {'< 75': [0, 75], '75_100': [75, 100], '100_150': [100, 150], '150_plus': [150, np.Inf]}

    def __init__(self, df: pd.DataFrame):
        self.values = ProjectValues()
        self.__df = df
        self.compute_price_ratio()
        self.dict_indexes = dict()

    def compute_price_ratio(self):
        self.index_logger(msg='Computing price ratio.')
        target = self.values.new_columns['price_m2']
        pred_col = self.values.new_columns['prediction']
        ratio_col = self.values.new_columns['price_ratio']
        self.df[ratio_col] = self.df[target]/self.df[pred_col]

    def compute_mean_price(self, df: pd.DataFrame):
        self.index_logger(msg='Computing mean price.')
        ratio_col = self.values.new_columns['price_ratio']
        mean_price = df[ratio_col].median()
        return mean_price

    @staticmethod
    def get_boolean_to_filter_df_from_feature(df: pd.DataFrame, list_feature: list, col_name: str):
        if len(list_feature) == 2:
            boolean_list = (df[col_name] > list_feature[0]) & (df[col_name] <= list_feature[1])
        elif len(list_feature) == 1:
            boolean_list = (df[col_name] == list_feature[0])
        else:
            raise ValueError(f'list feature {",".join(list_feature)} for {col_name} has length from 1 or 2')
        return boolean_list

    def filter_by_size_and_rooms(self, df: pd.DataFrame, list_size: list, list_rooms: list):
        size_col = self.values.input_columns['size']
        rooms_col = self.values.input_columns['rooms']
        bool_size = self.get_boolean_to_filter_df_from_feature(df=df, list_feature=list_size, col_name=size_col)
        bool_rooms = self.get_boolean_to_filter_df_from_feature(df=df, list_feature=list_rooms, col_name=rooms_col)
        df_loc = df[bool_size & bool_rooms]
        return df_loc

    def execute(self):
        for name_rooms, data_rooms in self.NODES_ROOMS.items():
            self.index_logger(f'{name_rooms} bedroom')
            self.dict_indexes[name_rooms] = dict()
            for name_size, data_size in self.NODES_SIZE.items():
                df_loc = self.filter_by_size_and_rooms(df=self.df, list_size=data_size, list_rooms=data_rooms)
                price_loc = self.compute_mean_price(df_loc)
                logger.info(f'  {name_rooms} bedroom - {name_size} size: {price_loc}')
                self.dict_indexes[name_rooms][name_size] = price_loc

    @property
    def df(self):
        return self.__df

    @df.setter
    def df(self, new_df: pd.DataFrame):
        self.__df = new_df

    def index_logger(self, msg: str):
        logger.info(f'{self.NAME}: {msg}')
