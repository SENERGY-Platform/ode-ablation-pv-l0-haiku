"""The operator: what it infers per message, and when it retrains.

MLOperator is the machine-learning half of Operator Lib. It loads the model
registered under this pipeline and operator from MLflow, calls infer() for every
message that matches a selector, and calls train() when there is no model yet or
when need_retraining() says so. Training runs on Ray.

Three methods are yours. Everything else — Kafka, the model registry, the
lifecycle — belongs to the library.
"""

import datetime
import typing

from mlflow.pyfunc import PyFuncModel, PythonModel

from operator_lib.util import Config, MLOperator, Selector
from operator_lib.util.helpers import TrainMlflowLogger

from training import train_model


# The forecast horizon: each result is the hourly mean power for the hour this far
# after the message that produced it.
HORIZON = datetime.timedelta(hours=24)


class CustomConfig(Config):
    """Deployment configuration, typed.

    The base Config already carries mlflow_url, ray_url and ts_conn. Anything
    added here arrives from the operator's deployment config under the same name,
    with this value as the default.
    """

    # Retrain at most this often, in seconds. A day, so a deployment does not
    # spend its life training.
    retrain_after_s = 86400


class Operator(MLOperator):
    configType = CustomConfig

    # Which inputs this operator accepts. "args" are the mapping destinations the
    # pipeline is configured with, and the name is what infer() receives as
    # "selector", so one operator can treat several input shapes differently.
    selectors = [
        Selector({"name": "value", "args": ["value"]}),
    ]

    def init(self, *args, **kwargs):
        # State first: under a data split, super().init() trains and replays the
        # test window before it returns, so anything set after it is missing
        # during the replay and overwrites what train() just recorded.
        self.trained_at: typing.Optional[datetime.datetime] = None
        super().init(*args, **kwargs)

    def infer(
        self,
        model: typing.Optional[PyFuncModel],
        data: typing.Dict[str, typing.Any],
        selector: str,
        device_id: str,
        timestamp: datetime.datetime,
    ) -> typing.Tuple[
        typing.Optional[datetime.datetime], typing.Optional[typing.Any], typing.Optional[PythonModel]
    ]:
        """Called for every message. Returns (result timestamp, result, new model).

        The message is the trigger: its value is present, but the forecast is the
        seasonal profile at the hour HORIZON after it, so the result carries that
        hour as its timestamp. Several messages in one hour answer the same hour,
        and the evaluation averages them within the bucket.
        """
        value = data.get("value")
        if value is None or model is None:
            return None, None, None

        at = timestamp if timestamp.tzinfo is not None else timestamp.replace(tzinfo=datetime.timezone.utc)
        target = at.replace(minute=0, second=0, microsecond=0) + HORIZON

        prediction = model.predict({"timestamp": target, "value": float(value)})
        return target, {"prediction": prediction}, None

    def train(
        self, model: typing.Optional[PyFuncModel], logger: TrainMlflowLogger
    ) -> typing.Optional[PythonModel]:
        """Called when there is no model, or when need_retraining() said so.

        Runs inside a Ray session the library opened, with an MLflow run already
        started — so params and metrics logged through "logger" land on the run
        the resulting model is registered from.
        """
        self.trained_at = datetime.datetime.now(datetime.timezone.utc)
        return train_model(logger)

    def need_retraining(self, model: typing.Optional[PyFuncModel]) -> bool:
        """Called after every inference, so it has to be cheap.

        Time-based to start with. A better policy watches the data — a drift in the
        input distribution, or an error that stops falling — and that is a decision
        to make with the profile in front of you rather than a default to inherit.
        """
        if model is None:
            return True
        if self.trained_at is None:
            return False
        age = datetime.datetime.now(datetime.timezone.utc) - self.trained_at
        return age.total_seconds() >= self.config.retrain_after_s
