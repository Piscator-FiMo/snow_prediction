from dataclasses import dataclass
from typing import Union, Optional


@dataclass
class Metadata:
    # file_dir
    file_dir: str
    # name of metadata file
    metadata_file_name: str | None
    # used to indicate the target
    target_cols: Union[str, list[str]]
    # used to parse the dataset file
    time_col: Optional[str]
    # used to create group series
    group_cols: Union[str, list[str]]
    # used to select past covariates
    past_cov_cols: Union[str, list[str]]
    # used to select static cols
    static_cols: Union[str, list[str]] = None
    # used to convert the string date to pd.Datetime
    # https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
    format_time: Optional[str] = None
    # used to indicate the freq when we already know it
    freq: Optional[str] = None
    # multivariate
    multivariate: Optional[bool] = None
    # cutoff
    training_cutoff: [float] = 0.5


snow_metadata: Metadata = Metadata(
    file_dir="data",
    metadata_file_name="stations.csv",
    target_cols="hs",
    time_col="measure_date",
    group_cols="station_code",
    past_cov_cols=['dw_30min_mean', 'dw_30min_sd', 'rh_30min_mean', 'rswr_30min_mean', 'ta_30min_mean',
                   'ts0_30min_mean', 'ts100_30min_mean', 'ts25_30min_mean', 'ts50_30min_mean', 'tss_30min_mean',
                   'vw_30min_max', 'vw_30min_mean'],
    static_cols=['active', 'elevation', 'label', 'lat', 'lon', 'network', 'type'],
    format_time="%Y-%m-%d %H:%M:%S+00:00",
    freq="30min",
    multivariate=False,
    training_cutoff=0.7,
)
