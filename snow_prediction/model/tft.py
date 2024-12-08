import pandas as pd
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import StaticCovariatesTransformer, Scaler
from darts.models import TFTModel, CatBoostModel
from darts.utils.likelihood_models import QuantileRegression

from snow_prediction.preprocessing.preprocessor import Preprocessor


class Model:
    def __init__(self, preprocessor: Preprocessor, version: str, model_name="TFT"):
        assert model_name == "TFT" or model_name == "CatBoost", 'Use "TFT" or "CatBoost" for model name'

        self.model_name = model_name
        self.preprocessor = preprocessor
        self.version = version
        # train
        self.train_target_transformed = None
        self.val_target_transformed = None
        # validation
        self.train_past_cov_transformed = None
        self.val_past_cov_transformed = None

        self.input_chunk_length = 3
        self.forecast_horizon = 12

        self.train_target_scaler = None
        self.train_past_cov_scaler = None
        self.train_static_transformer = None

        if model_name == "TFT":
            # before starting, we define some constants
            # default quantiles for QuantileRegression
            quantiles = [
                0.01,
                0.05,
                0.1,
                0.15,
                0.2,
                0.25,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.75,
                0.8,
                0.85,
                0.9,
                0.95,
                0.99,
            ]

            self.model = TFTModel(
                input_chunk_length=self.input_chunk_length,
                output_chunk_length=self.forecast_horizon,
                hidden_size=64,
                lstm_layers=1,
                num_attention_heads=4,
                dropout=0.1,
                batch_size=16,
                n_epochs=2,
                add_relative_index=True,
                add_encoders=None,
                likelihood=QuantileRegression(
                    quantiles=quantiles
                ),  # QuantileRegression is set per default
                # loss_fn=MSELoss(),
                random_state=42,
                use_static_covariates=True
            )

        if model_name == "CatBoost":
            self.model = CatBoostModel(
                lags=self.input_chunk_length,
                lags_past_covariates=self.input_chunk_length,
                lags_future_covariates=None,
                output_chunk_length=self.forecast_horizon
            )

    def transform(self):

        target_series_list = self.preprocessor.target_series_list
        past_cov_series_list = self.preprocessor.past_cov_series_list

        # use StaticCovariatesTransformer to encode categorical static covariates into numeric data
        self.train_static_transformer = StaticCovariatesTransformer()
        target_series_list = self.train_static_transformer.fit_transform(target_series_list)
        past_cov_series_list = self.train_static_transformer.fit_transform(past_cov_series_list)

        # Create training and validation sets:
        # target(s)
        target_series_list_split = [x.split_after(self.preprocessor.metadata.training_cutoff) for x in
                                    target_series_list]
        train_target_series_list = [x[0] for x in target_series_list_split]
        val_target_series_list = [x[1] for x in target_series_list_split]

        # past covariates:
        past_cov_series_list_split = [x.split_after(self.preprocessor.metadata.training_cutoff) for x in
                                      past_cov_series_list]
        train_past_cov_series_list = [x[0] for x in past_cov_series_list_split]
        val_past_cov_series_list = [x[1] for x in past_cov_series_list_split]

        # Normalize the time series (note: we avoid fitting the transformer on the validation set)

        self.train_target_scaler = Scaler(verbose=False, n_jobs=-1, name="Target_Scaling")
        self.train_past_cov_scaler = Scaler(verbose=False, n_jobs=-1, name="Past_Cov_Scaling")

        self.train_target_transformed = self.train_target_scaler.fit_transform(train_target_series_list)
        self.val_target_transformed = self.train_target_scaler.transform(val_target_series_list)

        self.train_past_cov_transformed = self.train_past_cov_scaler.fit_transform(train_past_cov_series_list)
        self.val_past_cov_transformed = self.train_past_cov_scaler.fit_transform(val_past_cov_series_list)

    def fit(self):
        self.model.fit(self.train_target_transformed,
                       past_covariates=self.train_past_cov_transformed,
                       val_series=self.val_target_transformed,
                       val_past_covariates=self.val_past_cov_transformed,
                       verbose=True)

    def predict(self, target_series: TimeSeries, past_covariates: TimeSeries):
        forecast = self.model.predict(n=self.forecast_horizon, series=target_series, past_covariates=past_covariates, )
        return forecast

    def validate(self, on_finished_projects):
        backtest_series_transformed = self.model.generate_backtest_series(on_finished_projects=on_finished_projects)

        backtest_series = self.model.train_target_scaler.inverse_transform(backtest_series_transformed)
        if on_finished_projects:
            target_series = self.model.val_target_transformed
            data = self.model.preprocessor.val_data
        else:
            target_series = self.model.test_target_transformed
            data = self.model.preprocessor.test_data
        target_series = self.model.train_target_scaler.inverse_transform(target_series)
        gp = data.groupby(self.model.preprocessor.metadata.group_cols)
        results = []
        for backtest, target, group in zip(backtest_series, target_series, gp.groups.keys()):
            print(f"Group Keys: {group}")
            target_df = target.pd_dataframe()
            backtest_df = concatenate(backtest).pd_dataframe()
            # print(backtest_df.index)

            merge = pd.merge(target_df, backtest_df, how='left', left_index=True, right_index=True,
                             suffixes=('_true', '_forecast'))
            merge['residuals'] = target_df - backtest_df
            merge['integration_id'] = group[0]
            merge['stage'] = group[1]
            merge['milestone'] = group[2]

            results.append(merge)
        result_df = pd.concat(results, axis=0)
        result_df.reset_index(drop=False, inplace=True)
        return result_df

    def save(self, directory: str):
        self.model.save()

    def load(self, directory: str):
        self.model = self.model.load(directory)
