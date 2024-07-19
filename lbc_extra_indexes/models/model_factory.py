from lbc_extra_indexes.models.nn import NNModel
from lbc_extra_indexes.utils.values_container import ProjectValues


class ModelFactory:
    MODELS = {'nn': {'model': NNModel,
                     'model_path': ProjectValues().new_values['output_model_files']['nn'],
                     'prediction_path': ProjectValues().new_values['output_predictions']['nn']}}

    def __init__(self, model_type: str):
        self.model_type = model_type

    def instance_model_class(self):
        try:
            return self.MODELS[self.model_type]['model'], \
                   self.MODELS[self.model_type]['model_path'], \
                   self.MODELS[self.model_type]['prediction_path']
        except KeyError as e:
            error_msg = f'{self.model_type} is not a valid model name. Please select from this list: ' \
                        f'{", ".join(self.MODELS.keys())}'
            raise ValueError(error_msg)
