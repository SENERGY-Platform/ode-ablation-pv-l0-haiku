"""Training, on Ray.

Separate from op.py because the two run in different places: op.py runs in the
operator's own process for every message, while this runs distributed and rarely.

provide_historic_data() is Operator Lib's reader over the platform's timeseries
store. It returns Ray Datasets, so the training below never holds the whole
history in one process.

The model is a seasonal profile of the PV power itself: for each calendar month
and each hour of the day, the median of the hourly means observed in that cell
over the training window. The median is the value that minimises absolute error,
which is what the evaluation scores; a few cloudy days in a cell do not drag the
profile down the way a mean would. A 24 hour ahead forecast is the profile cell
for the hour it is about. It uses no weather, so it follows the season and not the
cloud cover.
"""

import datetime
import statistics
import typing

import pandas as pd
import ray
from mlflow.pyfunc import PythonModel

from operator_lib.util.helpers import provide_historic_data, TrainMlflowLogger


# How much history one training pass reads. A year, so every calendar month is
# represented once in the training window and the test window's season is in it.
TRAINING_WINDOW = datetime.timedelta(days=365)

# Index of a (month, hour) cell in the profile table.
def _cell(month: int, hour: int) -> int:
    return (month - 1) * 24 + hour


class OdeAblationPvL0HaikuModel(PythonModel):
    """The model MLflow registers and op.py later loads.

    predict() receives exactly the payload infer() builds, so the two are one
    contract in two files. Keep it serialisable: MLflow stores this object.

    month_hour is 288 cells, (month - 1) * 24 + hour, None where the training
    window held no data for that cell. hour is 24 cells, the fallback for a
    month-hour cell with no data.
    """

    def __init__(self, month_hour: typing.List[typing.Optional[float]], hour: typing.List[float]) -> None:
        self.month_hour = month_hour
        self.hour = hour

    def predict(self, context, model_input=None, params=None):
        # The pyfunc signature carries a context when MLflow calls it and not when the
        # model is called directly, so the payload is taken from whichever argument
        # holds it.
        payload = model_input if model_input is not None else context
        at = payload["timestamp"]
        value = self.month_hour[_cell(at.month, at.hour)]
        if value is None:
            value = self.hour[at.hour]
        return max(0.0, float(value))


@ray.remote
def _hourly_sums(datasets: typing.List[typing.Any]) -> typing.Dict[str, typing.List[float]]:
    """The distributed part: per (day, hour), the sum and count of samples.

    Summed per hour rather than per sample, so that an irregular sampling
    interval does not weight the busy hours of the day more than the quiet ones.
    Returns a plain dict of [sum, count] keyed by the hour's ISO timestamp, which
    is small enough to come back to the driver whole.
    """
    sums: typing.Dict[str, typing.List[float]] = {}
    for dataset in datasets:
        ds = ray.get(dataset) if isinstance(dataset, ray.ObjectRef) else dataset
        for batch in ds.iter_batches(batch_size=4096, batch_format="pandas"):
            if "time" not in batch or "value" not in batch:
                raise ValueError("history batch carries no time or value column")
            frame = pd.DataFrame({
                "time": pd.to_datetime(batch["time"], utc=True),
                "value": pd.to_numeric(batch["value"], errors="coerce"),
            }).dropna()
            if frame.empty:
                continue
            grouped = frame.groupby(frame["time"].dt.floor("h"))["value"].agg(["sum", "count"])
            for hour, row in grouped.iterrows():
                key = hour.isoformat()
                acc = sums.setdefault(key, [0.0, 0.0])
                acc[0] += float(row["sum"])
                acc[1] += float(row["count"])
    return sums


def _profiles(sums: typing.Dict[str, typing.List[float]]):
    """Hourly means, then the median of those per (month, hour) and per hour alone."""
    cell_values: typing.List[typing.List[float]] = [[] for _ in range(288)]
    hour_values: typing.List[typing.List[float]] = [[] for _ in range(24)]
    hourly: typing.List[float] = []
    for key, (total, count) in sums.items():
        if count <= 0:
            continue
        mean = total / count
        at = pd.Timestamp(key)
        cell_values[_cell(at.month, at.hour)].append(mean)
        hour_values[at.hour].append(mean)
        hourly.append(mean)

    if not hourly:
        return None, None, 0, 0.0

    month_hour = [statistics.median(v) if v else None for v in cell_values]
    hour = [statistics.median(v) if v else 0.0 for v in hour_values]
    return month_hour, hour, len(hourly), statistics.fmean(hourly)


def train_model(logger: TrainMlflowLogger) -> typing.Optional[PythonModel]:
    """Read the history, fit the monthly-hourly profile, and hand back a model for MLflow to register."""
    with logger.trace("read history"):
        datasets = provide_historic_data(TRAINING_WINDOW)
    if not datasets:
        # Explicitly nothing rather than a model fitted on no data: returning None
        # leaves the previously registered model in place.
        return None

    with logger.trace("fit"):
        sums = ray.get(_hourly_sums.remote(datasets))
        month_hour, hour, hourly_count, mean_hourly = _profiles(sums)

    if month_hour is None:
        return None

    filled = sum(1 for cell in month_hour if cell is not None)
    logger.log_params({
        "training_window_days": TRAINING_WINDOW.days,
        "method": "month_hour_profile",
        "statistic": "median",
        "horizon_hours": 24,
    })
    logger.log_metrics({
        "train_hours": float(hourly_count),
        "train_cells_filled": float(filled),
        "train_mean_hourly": mean_hourly,
    })
    return OdeAblationPvL0HaikuModel(month_hour=month_hour, hour=hour)
