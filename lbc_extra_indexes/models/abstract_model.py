from abc import ABC, abstractmethod
import pandas as pd
from lbc_extra_indexes.models.trainer import Trainer
from lbc_extra_indexes.models.predictor import Predictor
from lbc_extra_indexes.utils.values_container import ProjectValues
from lbc_extra_indexes.utils.logger import logger


values = ProjectValues()


class AbstractModel(ABC):
    FEATURES = [values.new_columns['lat_bin'], values.new_columns['lng_bin'], values.input_columns['size'],
                values.input_columns['property_type'], values.input_columns['rooms']]
    TARGET = values.new_columns['price_m2']
    CATEGORICAL_FEATURES = [values.input_columns['property_type']]
    NUMERICAL_FEATURES = [values.new_columns['lat_bin'], values.new_columns['lng_bin'], values.input_columns['size'],
                          values.input_columns['rooms']]

    def __init__(self, df: pd.DataFrame, out_path_model: str, pred_out_path: str):
        selected_columns = self.FEATURES + [self.TARGET]
        self.df = df[selected_columns]
        self.out_path_model = out_path_model
        self.pred_out_path = pred_out_path
        self.input_pre_processor = None
        self.__model = None
        self.__trainer = Trainer()
        self.__predictor = Predictor()

    def train(self):
        self.trainer.create_input_transformer(numerical_features=self.NUMERICAL_FEATURES,
                                              categorical_features=self.CATEGORICAL_FEATURES)
        self.trainer.transform_input(df=self.df, features=self.FEATURES, target=self.TARGET)
        self.trainer.train_model(model=self.model, out_path=self.out_path_model)
        self.input_pre_processor = self.trainer.preprocessor

    def predict(self):
        pred = self.predictor.predict(model=self.model, df=self.df, preprocessor=self.input_pre_processor,
                                      features=self.FEATURES, target=self.TARGET, out_pred_path=self.pred_out_path)
        return pred

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, new_model):
        self.__model = new_model

    @property
    def trainer(self):
        return self.__trainer

    @property
    def predictor(self):
        return self.__predictor

    @staticmethod
    def model_logger(msg: str):
        logger.info(f'[Model Class]: {msg}')

