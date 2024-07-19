import yaml


class ProjectValues:
    def __init__(self):
        self.input_values = self.load_yaml('lbc_extra_indexes/values/input_values.yaml')
        self.input_columns = self.load_yaml('lbc_extra_indexes/values/input_columns.yaml')
        self.new_columns = self.load_yaml('lbc_extra_indexes/values/new_columns.yaml')
        self.new_values = self.load_yaml('lbc_extra_indexes/values/new_values.yaml')
        self.output_columns = self.load_yaml('lbc_extra_indexes/values/output_columns.yaml')

    @staticmethod
    def load_yaml(path: str):
        with open(path) as stream:
            result = yaml.safe_load(stream)
        return result
