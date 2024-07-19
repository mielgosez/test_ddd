import pandas as pd
from lbc_extra_indexes.repository.abstract_data_processor import AbstractDataProcessor


class DataProcessor(AbstractDataProcessor):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df=df)

    def select_france_metro(self):
        self.logger_message('Selecting Metro France.')
        lat = self.values.input_columns['latitude']
        lon = self.values.input_columns['longitude']
        self.df = self.df.loc[(self.df[lat] >= self.values.new_values['geo']['france_min_lat']) &
                              (self.df[lat] <= self.values.new_values['geo']['france_max_lat']) &
                              (self.df[lon] >= self.values.new_values['geo']['france_min_lng']) &
                              (self.df[lon] <= self.values.new_values['geo']['france_max_lng']), :]

    def select_professionals(self):
        self.logger_message('Selecting Professionals.')
        column = self.values.input_columns['publisher_type']
        value = [self.values.input_values['publisher_type']['professional']]
        self.filter_df(column=column, value=value)

    def select_houses_and_flats(self):
        self.logger_message('Selecting Houses and Flats.')
        column = self.values.input_columns['property_type']
        value = [self.values.input_values['property_type']['houses'],
                 self.values.input_values['property_type']['flats']]
        self.filter_df(column=column, value=value)

    def select_sells(self):
        self.logger_message('Selecting Sells.')
        column = self.values.input_columns['operation']
        value = [self.values.input_values['operation']['sell']]
        self.filter_df(column=column, value=value)

    def select_columns(self):
        self.logger_message('Selecting Columns to Preserve.')
        # @TODO: Decide columns to preserve

    def remove_price_with_na(self):
        self.logger_message('Removing NAN Prices.')
        column = self.values.input_columns['price']
        self.df.dropna(subset=[column], inplace=True)

    def remove_properties_with_size_zero(self):
        self.logger_message('Removing properties with size <= 0.')
        column = self.values.input_columns['size']
        self.df = self.df[self.df[column] > 0]

    def bin_to_float(self, ref_col: str, new_col: str, binning_size: int):
        self.df[new_col] = pd.cut(self.df[ref_col], binning_size)
        bins = self.df[new_col].values.categories.left.values
        midpoints = (bins[:-1] + bins[1:]) / 2
        bin_to_midpoint = {interval: midpoint for interval, midpoint in
                           zip(self.df[new_col].values.categories, midpoints)}
        self.df[new_col] = self.df[ref_col].map(bin_to_midpoint).astype(float)

    def binning_area(self):
        self.logger_message('Binning long and lat.')
        old_lat = self.values.input_columns['latitude']
        old_lng = self.values.input_columns['longitude']
        new_lat = self.values.new_columns['lat_bin']
        new_lng = self.values.new_columns['lng_bin']
        binning_size = self.values.new_values['geo']['binning_size']
        self.bin_to_float(ref_col=old_lat, new_col=new_lat, binning_size=binning_size)
        self.bin_to_float(ref_col=old_lng, new_col=new_lng, binning_size=binning_size)

    def compute_price_m2(self):
        self.logger_message('Create column price_m2')
        new_column = self.values.new_columns['price_m2']
        price_column = self.values.input_columns['price']
        size_column = self.values.input_columns['size']
        self.df[new_column] = self.df[price_column]/self.df[size_column]

    def remove_points_with_low_quality_coordinates(self):
        self.logger_message('Remove duplicated coordinates.')
        n = self.values.new_values['thresholds']['n_duplicated_coords']
        lat = self.values.input_columns['latitude']
        lng = self.values.input_columns['longitude']
        counter = self.df[[lat, lng]].value_counts()
        selected_coords = counter[counter < n].reset_index()
        self.df = self.df.loc[self.df[lat].isin(selected_coords[lat]) & self.df[lng].isin(selected_coords[lng]), :]
