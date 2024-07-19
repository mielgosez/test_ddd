import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from lbc_extra_indexes.utils.logger import logger
from tensorflow.keras.models import load_model


class Trainer:
    def __init__(self):
        self.preprocessor = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.loss = None

    def create_input_transformer(self, numerical_features: list, categorical_features: list):
        self.trainer_logger('Creating input transformer')
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(), categorical_features)
            ])

    def transform_input(self, df: pd.DataFrame, features: list, target: str):
        self.trainer_logger('Transform and splitting inputs')
        df.dropna(inplace=True)
        x_preprocessed = self.preprocessor.fit_transform(df[features])
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_preprocessed, df[target],
                                                                                test_size=0.2, random_state=42)

    def train_model(self, model, out_path: str):
        self.trainer_logger('Training the model')
        model.fit(self.x_train, self.y_train, epochs=100, batch_size=32, validation_split=0.2)
        self.loss = model.evaluate(self.x_test, self.y_test)
        # TODO: Parametrize output path
        model.save(out_path)

    @staticmethod
    def trainer_logger(msg: str):
        logger.info(f'Model Trainer: {msg}')
