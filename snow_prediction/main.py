import logging
import os
from datetime import datetime

import pandas as pd
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


def eval_model(pred_series, actual_series, val_series):
    # plot actual series
    plt.figure(figsize=figsize)
    actual_series[: pred_series.end_time()].plot(label="actual")

    # plot prediction with quantile ranges
    pred_series.plot(
        low_quantile=lowest_q, high_quantile=highest_q, label=label_q_outer
    )
    pred_series.plot(low_quantile=low_q, high_quantile=high_q, label=label_q_inner)

    plt.title("MAPE: {:.2f}%".format(mape(val_series, pred_series)))
    plt.legend()


def main():
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    logger.info(f"Logging started at {datetime.now()}")
    preproc = Preprocessor(metadata=snow_metadata, filter_data=['ADE2.csv', 'ERN1.csv'])
    preproc.process()
    model = Model(preproc, '1', 'TFT')
    model.transform()
    print(model.train_target_transformed[0].static_covariates)
    model.fit()
    # model = (Model(preproc, '1', 'TFT'))
    # model.load('TFTModel_2024-12-08_21_17_19.pt')
    # model.transform()
    predictions_list = model.predict(model.train_target_transformed, model.train_past_cov_transformed)

    predictions_list = model.train_target_scaler.inverse_transform(predictions_list)
    model.save(os.path.abspath(os.path.join(os.path.dirname(__file__), 'model')))

    eval_model(predictions_list[0], predictions_list[1], predictions_list[2])

    prediction_df = pd.concat([prediction.pd_dataframe() for prediction in predictions_list])

    gp = preproc.data.groupby(snow_metadata.group_cols)
    groups_df = pd.DataFrame(gp.groups.keys(), index=prediction_df.index)

    prediction_df = pd.concat([groups_df, prediction_df], axis=1)

    if model.model_name == "TFT":
        explainer = TFTExplainer(model.model)
        results = explainer.explain()
        # plot the results
        explainer.plot_attention(results, plot_type="all")
        explainer.plot_variable_selection(results)


if __name__ == "__main__":
    main()
