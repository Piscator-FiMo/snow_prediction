import logging
import os
from datetime import datetime

import pandas as pd
from darts import TimeSeries
from darts.explainability import TFTExplainer
from darts.metrics import mape
from matplotlib import pyplot as plt

from snow_prediction.model.tft import Model
from snow_prediction.preprocessing.metadata import snow_metadata
from snow_prediction.preprocessing.preprocessor import Preprocessor

logger = logging.getLogger(__name__)
figsize = (9, 6)
lowest_q, low_q, high_q, highest_q = 0.01, 0.1, 0.9, 0.99
label_q_outer = f"{int(lowest_q * 100)}-{int(highest_q * 100)}th percentiles"
label_q_inner = f"{int(low_q * 100)}-{int(high_q * 100)}th percentiles"


def eval_model(pred_series: TimeSeries, actual_series: TimeSeries):
    # plot actual series
    plt.figure(figsize=figsize)
    actual_series[pred_series.start_time() - pd.Timedelta(48 * 30, unit='h'): pred_series.end_time()].plot(label="actual")

    # plot prediction with quantile ranges
    pred_series.plot(
        low_quantile=lowest_q, high_quantile=highest_q, label=label_q_outer
    )
    pred_series.plot(low_quantile=low_q, high_quantile=high_q, label=label_q_inner)

    plt.title("MAPE: {:.2f}%".format(mape(actual_series[pred_series.start_time():pred_series.end_time()], pred_series)))
    plt.legend()
    plt.show()

def validate(model: Model):
    start = model.train_target_transformed.end_time() + model.train_target_transformed.freq
    backtest_series = model.model.historical_forecasts(
        actual_series=model.actual_target_transformed,
        start=start,
        num_samples=200,
        forecast_horizon=model.forecast_horizon,
        stride=model.forecast_horizon,
        last_points_only=False,
        retrain=False,
        verbose=True,
    )
    plt.figure(figsize=figsize)
    model.actual_target_transformed.plot(label="actual")
    backtest_series.plot(
        low_quantile=lowest_q, high_quantile=highest_q, label=label_q_outer
    )
    backtest_series.plot(low_quantile=low_q, high_quantile=high_q, label=label_q_inner)
    plt.legend()
    plt.title(f"Backtest, starting {start}, {model.forecast_horizon}-30mins horizon")
    print(
        "MAPE: {:.2f}%".format(
            mape(
                model.train_target_scaler.inverse_transform(model.actual_target_transformed),
                model.train_target_scaler.inverse_transform(backtest_series),
            )
        )
    )

def main():
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    logger.info(f"Logging started at {datetime.now()}")
    preproc = Preprocessor(metadata=snow_metadata, filter_data=['ADE2.csv', 'ALB2.csv'])
    preproc.process()
    model = Model(preproc, '1', 'TFT')
    model.transform()
    print(model.train_target_transformed[0].static_covariates)
    model.fit()
    # model = Model(preproc, '1', 'TFT')
    # model.load('TFTModel_2024-12-09_07_09_44.pt')
    # model.transform()
    predictions_list = model.predict(model.train_target_transformed, model.train_past_cov_transformed)

    predictions_list = model.train_target_scaler.inverse_transform(predictions_list)
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'model'))
    model.save(path)

    eval_model(predictions_list[0], model.actual_target_series[0])
    eval_model(predictions_list[1], model.actual_target_series[1])
    prediction_df = pd.concat([prediction.pd_dataframe() for prediction in predictions_list])

    # gp = preproc.data.groupby(snow_metadata.group_cols)
    # groups_df = pd.DataFrame(gp.groups.keys(), index=prediction_df.index)

    # prediction_df = pd.concat([groups_df, prediction_df], axis=1)

    if model.model_name == "TFT":
        explainer = TFTExplainer(model.model)
        results = explainer.explain()
        # plot the results
        explainer.plot_attention(results, plot_type="all")
        explainer.plot_variable_selection(results)


if __name__ == "__main__":
    main()
