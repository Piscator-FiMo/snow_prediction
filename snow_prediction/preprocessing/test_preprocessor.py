from unittest import TestCase

from snow_prediction.preprocessing.metadata import snow_metadata
from snow_prediction.preprocessing.preprocessor import Preprocessor


class TestPreprocessor(TestCase):
    def test__get_unique_columns(self):
        metadata = snow_metadata
        preprocessor = Preprocessor(metadata)
        columns: list[str] = preprocessor.get_unique_columns(preprocessor.get_files())
        self.assertEqual(columns,
                         {'DW_30MIN_MEAN', 'DW_30MIN_SD', 'HS', 'RH_30MIN_MEAN', 'RSWR_30MIN_MEAN', 'TA_30MIN_MEAN',
                          'TS0_30MIN_MEAN', 'TS100_30MIN_MEAN', 'TS25_30MIN_MEAN', 'TS50_30MIN_MEAN', 'TSS_30MIN_MEAN',
                          'VW_30MIN_MAX', 'VW_30MIN_MEAN', 'active', 'elevation', 'hyear', 'label', 'lat', 'lon',
                          'measure_date', 'network', 'station_code', 'type'})
        self.assertEqual(23, len(columns))


    def test__process_frame(self):
        metadata = snow_metadata
        preprocessor = Preprocessor(metadata)
        columns = preprocessor.get_unique_columns(preprocessor.get_files())
        first_file = preprocessor.get_files()[0]
        df = preprocessor._process_frame(first_file, list(columns))
        self.assertEqual(df.get('station_code')[0], 'ADE2')
        self.assertEqual(23, len(df.columns.values.tolist()))

    def test__union_group(self):
        metadata = snow_metadata
        preprocessor = Preprocessor(metadata)
        df = preprocessor._union_groups()
        self.assertEqual(23, len(df.columns.values.tolist()))
