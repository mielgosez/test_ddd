import pandas as pd
from lbc_extra_indexes.repository.data_ingestor import DataIngestorCSV
from lbc_extra_indexes.repository.data_processor import DataProcessor
from lbc_extra_indexes.models.model_factory import ModelFactory
from lbc_extra_indexes.utils.logger import logger
from lbc_extra_indexes.utils.values_container import ProjectValues
from lbc_extra_indexes.use_cases.index_generator import PriceIndexExtractor


values = ProjectValues()


class Orchestrator:
    def __init__(self, input_path: str, model_type: str):
        self.model = None
        self.model_type = model_type
        self.input_path = input_path

    def run_ingestor(self):
        self.orchestrator_logger('Running Ingestor')
        ingestor = DataIngestorCSV(path=self.input_path)
        ingestor.load_data()
        return ingestor.df

    def process_data(self, df: pd.DataFrame):
        self.orchestrator_logger('Processing data')
        processor = DataProcessor(df=df)
        processor.execute()
        return processor.df

    def train_model(self, df: pd.DataFrame):
        self.orchestrator_logger('Training Data')
        model_class, model_out_path, pred_out_path = ModelFactory(self.model_type).instance_model_class()
        self.model = model_class(df=df, out_path_model=model_out_path, pred_out_path=pred_out_path)
        self.model.instance_model()
        self.model.train()

    def computing_index(self, df: pd.DataFrame):
        self.orchestrator_logger('Computing Indexes')
        index_extractor = PriceIndexExtractor(df=df)
        index_extractor.execute()

    def predict(self):
        self.orchestrator_logger('Creating prediction')
        pred = self.model.predict()
        return pred

    def execute(self, train_required: bool = True):
        if train_required:
            df = self.run_ingestor()
            df = self.process_data(df=df)
            self.train_model(df=df)
            df = self.predict()
        else:
            df = pd.read_csv(values.new_values['output_predictions']['nn'])
        self.computing_index(df=df)

    @staticmethod
    def orchestrator_logger(msg: str):
        logger.info(f'[ORCHESTRATOR]: {msg}')
