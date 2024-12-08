"""
Preprocessor to unify and make numerical
"""
import glob
import logging
import os
from datetime import datetime

import pandas as pd
from darts import TimeSeries
from tqdm import tqdm

from snow_prediction.preprocessing.metadata import Metadata

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Preprocessor:
    def __init__(self, metadata: Metadata, filter_data: list[str] = None):
        self.filter_data: list[str] = filter_data
        logger.info("Starting preprocessor")
        self.metadata: Metadata = metadata
        self.data: pd.DataFrame = None
        self.past_cov_series_list = None
        self.target_series_list = None

    def process(self):
        logger.info("Starting preprocessor")
        self.data: pd.DataFrame = self._union_groups()
        self.data = self._format_time_column(self.data)
        self.data.set_index(self.metadata.time_col, inplace=True)
        self.data = self.data.groupby(self.metadata.group_cols, as_index=False).resample('30min').ffill()
        self.data.reset_index(inplace=True)
        self.past_cov_series_list = self._to_timeseries(self.metadata.past_cov_cols)
        self.target_series_list = self._to_timeseries(self.metadata.target_cols)

    def _format_time_column(self, df):
        df[self.metadata.time_col] = df[self.metadata.time_col].apply(
            lambda x: datetime.strptime(str(x), self.metadata.format_time))
        return df

    def _to_timeseries(self, value_col: str | list[str]) -> list[TimeSeries]:
        logger.info(f"Converting {value_col} to timeseries")
        return TimeSeries.from_group_dataframe(
            df=self.data,
            group_cols=self.metadata.group_cols,
            time_col=self.metadata.time_col,
            value_cols=value_col,
            static_cols=self.metadata.static_cols,
            fill_missing_dates=True,
            fillna_value=0.0,
            freq=self.metadata.freq,
            drop_group_cols=self.metadata.group_cols
        )

    def _union_groups(self):
        csv_files = self.get_files()
        columns = self.get_unique_columns(csv_files)
        logger.info("Starting to union groups")
        frames = self.process_frames(columns, csv_files)
        return pd.concat(frames)

    def process_frames(self, columns, csv_files):
        if not self.filter_data:
            return [self._process_frame(f, columns) for f in tqdm(csv_files)]
        return [self._process_frame(f, columns) for f in tqdm(csv_files) if os.path.basename(f) in self.filter_data]

    def get_unique_columns(self, csv_files: list[str]) -> list[str]:
        columns = set()
        for file in csv_files:
            columns = columns.union(pd.read_csv(file, nrows=0).columns.to_list())

        return [x.lower() for x in columns]

    def get_files(self) -> list[str]:
        return glob.glob(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'data', '*.csv')))

    def _get_station_df(self):
        metadata_file = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'metadata', 'stations.csv'))
        return pd.read_csv(metadata_file)

    def _process_frame(self, file: str, columns: list[str]) -> pd.DataFrame:
        station_df = self._get_station_df()
        try:
            df = pd.read_csv(file)
            df.columns = map(str.lower, df.columns)
            df = df.reindex(columns=columns)
            df.ffill(inplace=True)
            df.dropna(subset=self.metadata.target_cols, inplace=True)
            df = pd.merge(df, station_df, on='station_code', how='left')
            return df
        except Exception as inst:
            inst.add_note(f"file {file} could not be processed")
            raise inst
