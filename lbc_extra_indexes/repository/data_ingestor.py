from lbc_extra_indexes.repository.abstract_data_ingestor import DataIngestor
import pandas as pd


class DataIngestorCSV(DataIngestor):
    def __init__(self, path: str):
        self.path = path
        super().__init__()

    def load_data(self):
        df = pd.read_csv(self.path)
        self.df = df
