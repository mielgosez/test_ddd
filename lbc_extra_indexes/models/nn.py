import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from lbc_extra_indexes.models.abstract_model import AbstractModel


class NNModel(AbstractModel):
    def __init__(self, df: pd.DataFrame, out_path_model: str, pred_out_path: str):
        super().__init__(df=df, out_path_model=out_path_model, pred_out_path=pred_out_path)

    def instance_model(self):
        self.model_logger('Instantiating NN Model')
        self.model = Sequential()
        self.model.add(Dense(64, input_dim=self.df.shape[1]-1, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
