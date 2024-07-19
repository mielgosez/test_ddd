import pandas as pd
from lbc_extra_indexes.utils.values_container import ProjectValues
from lbc_extra_indexes.utils.logger import logger


class Predictor:
    ID = '[Predictor]'

    def __init__(self):
        self.values = ProjectValues()

    def predict(self, df: pd.DataFrame, model, preprocessor, features: list, target: str, out_pred_path: str):
        self.predictor_logger(msg='Performing Prediction.')
        df_target = df[target]
        df = df[features]
        df.dropna(inplace=True)
        x_preprocessed = preprocessor.transform(df)
        pred = model.predict(x_preprocessed)
        df[self.values.output_columns['prediction']] = pred
        df[target] = df_target
        df.to_csv(out_pred_path, index=False)
        return df

    def predictor_logger(self, msg: str):
        logger.info(f'{self.ID}: {msg}')



