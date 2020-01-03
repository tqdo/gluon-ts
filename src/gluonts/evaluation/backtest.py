# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# Standard library imports
import logging
import re
from typing import Dict, Iterator, NamedTuple, Optional, Tuple, Union

# Third-party imports
import pandas as pd

# First-party imports
import gluonts  # noqa
from gluonts import transform
from gluonts.core.serde import load_code
from gluonts.dataset.common import DataEntry, Dataset
from gluonts.dataset.loader import InferenceDataLoader
from gluonts.dataset.stat import (
    DatasetStatistics,
    calculate_dataset_statistics,
)
from gluonts.evaluation import Evaluator
from gluonts.model.estimator import Estimator, GluonEstimator
from gluonts.model.forecast import Forecast
from gluonts.model.predictor import GluonPredictor, Predictor
from gluonts.transform import TransformedDataset


def make_evaluation_predictions(
    dataset: Dataset, predictor: Predictor, num_samples: int
) -> Tuple[Iterator[Forecast], Iterator[pd.Series]]:
    """
    Return predictions on the last portion of predict_length time units of the
    target. Such portion is cut before making predictions, such a function can
    be used in evaluations where accuracy is evaluated on the last portion of
    the target.

    Parameters
    ----------
    dataset
        Dataset where the evaluation will happen. Only the portion excluding
        the prediction_length portion is used when making prediction.
    predictor
        Model used to draw predictions.
    num_samples
        Number of samples to draw on the model when evaluating.

    Returns
    -------
    """

    prediction_length = predictor.prediction_length
    freq = predictor.freq

    def add_ts_dataframe(
        data_iterator: Iterator[DataEntry]
    ) -> Iterator[DataEntry]:
        for data_entry in data_iterator:
            data = data_entry.copy()
            holidays_NYSE=['1998-04-10',
         '1998-05-25',
         '1998-07-03',
         '1998-09-07',
         '1998-11-26',
         '1998-12-25',
         '1999-01-01',
         '1999-01-18',
         '1999-02-15',
         '1999-04-02',
         '1999-05-31',
         '1999-07-05',
         '1999-09-06',
         '1999-11-25',
         '1999-12-24',
         '2000-01-01',
         '2000-01-17',
         '2000-02-21',
         '2000-04-21',
         '2000-05-29',
         '2000-07-04',
         '2000-09-04',
         '2000-11-23',
         '2000-12-25',
         '2001-01-01',
         '2001-01-15',
         '2001-02-19',
         '2001-04-13',
         '2001-05-28',
         '2001-07-04',
         '2001-09-03',
         '2001-09-11',
         '2001-09-12',
         '2001-09-13',
         '2001-09-14',
         '2001-09-15',
         '2001-09-16',
         '2001-11-22',
         '2001-12-25',
         '2002-01-01',
         '2002-01-21',
         '2002-02-18',
         '2002-03-29',
         '2002-05-27',
         '2002-07-04',
         '2002-09-02',
         '2002-11-28',
         '2002-12-25',
         '2003-01-01',
         '2003-01-20',
         '2003-02-17',
         '2003-04-18',
         '2003-05-26',
         '2003-07-04',
         '2003-09-01',
         '2003-11-27',
         '2003-12-25',
         '2004-01-01',
         '2004-01-19',
         '2004-02-16',
         '2004-04-09',
         '2004-05-31',
         '2004-06-11',
         '2004-07-05',
         '2004-09-06',
         '2004-11-25',
         '2004-12-24',
         '2005-01-01',
         '2005-01-17',
         '2005-02-21',
         '2005-03-25',
         '2005-05-30',
         '2005-07-04',
         '2005-09-05',
         '2005-11-24',
         '2005-12-26',
         '2006-01-02',
         '2006-01-16',
         '2006-02-20',
         '2006-04-14',
         '2006-05-29',
         '2006-07-04',
         '2006-09-04',
         '2006-11-23',
         '2006-12-25',
         '2007-01-01',
         '2007-01-02',
         '2007-01-15',
         '2007-02-19',
         '2007-04-06',
         '2007-05-28',
         '2007-07-04',
         '2007-09-03',
         '2007-11-22',
         '2007-12-25',
         '2008-01-01',
         '2008-01-21',
         '2008-02-18',
         '2008-03-21',
         '2008-05-26',
         '2008-07-04',
         '2008-09-01',
         '2008-11-27',
         '2008-12-25',
         '2009-01-01',
         '2009-01-19',
         '2009-02-16',
         '2009-04-10',
         '2009-05-25',
         '2009-07-03',
         '2009-09-07',
         '2009-11-26',
         '2009-12-25',
         '2010-01-01',
         '2010-01-18',
         '2010-02-15',
         '2010-04-02',
         '2010-05-31',
         '2010-07-05',
         '2010-09-06',
         '2010-11-25',
         '2010-12-24',
         '2011-01-01',
         '2011-01-17',
         '2011-02-21',
         '2011-04-22',
         '2011-05-30',
         '2011-07-04',
         '2011-09-05',
         '2011-11-24',
         '2011-12-26',
         '2012-01-02',
         '2012-01-16',
         '2012-02-20',
         '2012-04-06',
         '2012-05-28',
         '2012-07-04',
         '2012-09-03',
         '2012-10-29',
         '2012-10-30',
         '2012-11-22',
         '2012-12-25',
         '2013-01-01',
         '2013-01-21',
         '2013-02-18',
         '2013-03-29',
         '2013-05-27',
         '2013-07-04',
         '2013-09-02',
         '2013-11-28',
         '2013-12-25',
         '2014-01-01',
         '2014-01-20',
         '2014-02-17',
         '2014-04-18',
         '2014-05-26',
         '2014-07-04',
         '2014-09-01',
         '2014-11-27',
         '2014-12-25',
         '2015-01-01',
         '2015-01-19',
         '2015-02-16',
         '2015-04-03',
         '2015-05-25',
         '2015-07-03',
         '2015-09-07',
         '2015-11-26',
         '2015-12-25',
         '2016-01-01',
         '2016-01-18',
         '2016-02-15',
         '2016-03-25',
         '2016-05-30',
         '2016-07-04',
         '2016-09-05',
         '2016-11-24',
         '2016-12-26',
         '2017-01-02',
         '2017-01-16',
         '2017-02-20',
         '2017-04-14',
         '2017-05-29',
         '2017-07-04',
         '2017-09-04',
         '2017-11-23',
         '2017-12-25',
         '2018-01-01',
         '2018-01-15',
         '2018-02-19',
         '2018-03-30',
         '2018-05-28',
         '2018-07-04',
         '2018-09-03',
         '2018-11-22',
         '2018-12-05',
         '2018-12-25',
         '2019-01-01',
         '2019-01-21',
         '2019-02-18',
         '2019-04-19',
         '2019-05-27',
         '2019-07-04',
         '2019-09-02',
         '2019-11-28',
         '2019-12-25',
         '2020-01-01',
         '2020-01-20',
         '2020-02-17',
         '2020-04-10',
         '2020-05-25',
         '2020-07-03',
         '2020-09-07',
         '2020-11-26',
         '2020-12-25']
            
            index = pd.date_range(
                start=data["start"],
                freq=pd.offsets.CustomBusinessHour(end='16:00',holidays=holidays_NYSE),#freq,
                #periods=data["target"].shape[-1],
                end=data["start"]+(data["target"].shape[-1]/7-1)*pd.offsets.CustomBusinessDay(holidays=holidays_NYSE)+pd.offsets.Hour(6)
            )
            data["ts"] = pd.DataFrame(
                index=index, data=data["target"].transpose()
            )
            yield data

    def ts_iter(dataset: Dataset) -> pd.DataFrame:
        for data_entry in add_ts_dataframe(iter(dataset)):
            yield data_entry["ts"]

    def truncate_target(data):
        data = data.copy()
        target = data["target"]
        assert (
            target.shape[-1] >= prediction_length
        )  # handles multivariate case (target_dim, history_length)
        data["target"] = target[..., :-prediction_length]
        return data

    # TODO filter out time series with target shorter than prediction length
    # TODO or fix the evaluator so it supports missing values instead (all
    # TODO the test set may be gone otherwise with such a filtering)

    dataset_trunc = TransformedDataset(
        dataset, transformations=[transform.AdhocTransform(truncate_target)]
    )

    return (
        predictor.predict(dataset_trunc, num_samples=num_samples),
        ts_iter(dataset),
    )


train_dataset_stats_key = "train_dataset_stats"
test_dataset_stats_key = "test_dataset_stats"
estimator_key = "estimator"
agg_metrics_key = "agg_metrics"


def serialize_message(logger, message: str, variable):
    logger.info(f"gluonts[{message}]: {variable}")


def backtest_metrics(
    train_dataset: Optional[Dataset],
    test_dataset: Dataset,
    forecaster: Union[Estimator, Predictor],
    evaluator=Evaluator(
        quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    ),
    num_samples: int = 100,
    logging_file: Optional[str] = None,
    use_symbol_block_predictor: bool = False,
):
    """
    Parameters
    ----------
    train_dataset
        Dataset to use for training.
    test_dataset
        Dataset to use for testing.
    forecaster
        An estimator or a predictor to use for generating predictions.
    evaluator
        Evaluator to use.
    num_samples
        Number of samples to use when generating sample-based forecasts.
    logging_file
        If specified, information of the backtest is redirected to this file.
    use_symbol_block_predictor
        Use a :class:`SymbolBlockPredictor` during testing.

    Returns
    -------
    tuple
        A tuple of aggregate metrics and per-time-series metrics obtained by
        training `forecaster` on `train_dataset` and evaluating the resulting
        `evaluator` provided on the `test_dataset`.
    """

    if logging_file is not None:
        log_formatter = logging.Formatter(
            "[%(asctime)s %(levelname)s %(thread)d] %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
        )
        logger = logging.getLogger(__name__)
        handler = logging.FileHandler(logging_file)
        handler.setFormatter(log_formatter)
        logger.addHandler(handler)
    else:
        logger = logging.getLogger(__name__)

    if train_dataset is not None:
        train_statistics = calculate_dataset_statistics(train_dataset)
        serialize_message(logger, train_dataset_stats_key, train_statistics)

    test_statistics = calculate_dataset_statistics(test_dataset)
    serialize_message(logger, test_dataset_stats_key, test_statistics)

    if isinstance(forecaster, Estimator):
        serialize_message(logger, estimator_key, forecaster)
        assert train_dataset is not None
        predictor = forecaster.train(train_dataset)

        if isinstance(forecaster, GluonEstimator) and isinstance(
            predictor, GluonPredictor
        ):
            inference_data_loader = InferenceDataLoader(
                dataset=test_dataset,
                transform=predictor.input_transform,
                batch_size=forecaster.trainer.batch_size,
                ctx=forecaster.trainer.ctx,
                dtype=forecaster.dtype,
            )

            if forecaster.trainer.hybridize:
                predictor.hybridize(batch=next(iter(inference_data_loader)))

            if use_symbol_block_predictor:
                predictor = predictor.as_symbol_block_predictor(
                    batch=next(iter(inference_data_loader))
                )
    else:
        predictor = forecaster

    forecast_it, ts_it = make_evaluation_predictions(
        test_dataset, predictor=predictor, num_samples=num_samples
    )

    agg_metrics, item_metrics = evaluator(
        ts_it, forecast_it, num_series=len(test_dataset)
    )

    # we only log aggregate metrics for now as item metrics may be very large
    for name, value in agg_metrics.items():
        serialize_message(logger, f"metric-{name}", value)

    if logging_file is not None:
        # Close the file handler to avoid letting the file open.
        # https://stackoverflow.com/questions/24816456/python-logging-wont-shutdown
        logger.removeHandler(handler)
        del logger, handler

    return agg_metrics, item_metrics


class BacktestInformation(NamedTuple):
    train_dataset_stats: DatasetStatistics
    test_dataset_stats: DatasetStatistics
    estimator: Estimator
    agg_metrics: Dict[str, float]

    @staticmethod
    def make_from_log(log_file):
        with open(log_file, "r") as f:
            return BacktestInformation.make_from_log_contents(
                "\n".join(f.readlines())
            )

    @staticmethod
    def make_from_log_contents(log_contents):
        messages = dict(re.findall(r"gluonts\[(.*)\]: (.*)", log_contents))

        # avoid to fail if a key is missing for instance in the case a run did
        # not finish so that we can still get partial information
        try:
            return BacktestInformation(
                train_dataset_stats=eval(
                    messages[train_dataset_stats_key]
                ),  # TODO: use load
                test_dataset_stats=eval(
                    messages[test_dataset_stats_key]
                ),  # TODO: use load
                estimator=load_code(messages[estimator_key]),
                agg_metrics={
                    k: load_code(v)
                    for k, v in messages.items()
                    if k.startswith("metric-") and v != "nan"
                },
            )
        except Exception as error:
            logging.error(error)
            return None
