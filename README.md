# ode-ablation-pv-l0-haiku

## Approach and status

**Problem.** Forecast photovoltaic generation 24 hours ahead, as hourly mean power in watts. The operator emits one prediction per message, timestamped with the hour it is about. The developer's criteria in `evaluation.yaml` score the prediction by mean absolute error (MAE) against a withheld target at 1 hour resolution, with a threshold of 30 and lower is better.

**Status.** Not complete. The most recent evaluation does not meet the threshold.

| Run | Statistic | MAE over 391 scored hours | Threshold | Met |
|---|---|---|---|---|
| `80b49d4` (commit `3208b4a`) | mean of hourly means per cell | 67.8 | 30 | no |
| `61e7e0a` (commit `40c0ad5`) | median of hourly means per cell | 69.5 | 30 | no |

**Data.** The only series used is the inverter's total power, `root.powerTotal` on the device "Wechselrichter" (APSystems DS3-S), service "Get Energy Generation". Its stored history runs from 2022-11-01 to the training end. Two other candidates were not used: "Leiste PV" is a generation meter on a different device type, and "Shelly PV (Tasmota)" is named PV but measures consumption. The unit of `root.powerTotal` has not been verified. The variable carries the same characteristic as the other PV power variable, which suggests watts, but neither has been profiled.

**Method, as committed.**

- *Training* (`training.py`): a 365-day window of inverter history before the split. Samples are summed into hourly means. For each calendar month and hour of day (288 cells), the median of the hourly means is kept; hours with no data in a cell fall back to the median for that hour of day across all months. Values are clipped at zero.
- *Inference* (`op.py`): each message is a trigger. The prediction is the profile value for the hour 24 hours after the message, and that hour is returned as the result timestamp. Several messages in one hour answer the same hour; the evaluation averages them within the bucket.
- *Evaluation*: the replay covers 2026-09-01 to 2026-10-01 and 20,715 messages. Only 391 of the 720 hours were scored. I read this as daytime hours, since the inverter probably reports nothing at night, but the run's log would confirm it.

**Known limits.**

- The profile has no weather input. It follows the season, not the cloud cover, and the error is most likely dominated by day-to-day weather variation.
- 179 of the 288 month-hour cells had data. Night hours are predicted as zero, and empty cells ignore the season.
- The unit of the input has not been verified.
- The threshold of 30 may not be reachable from history alone. The criteria belong to the developer and have not been changed.

**Next step (not implemented).** Create an Open-Meteo previous-runs import (`urn:infai:ses:import-type:f2cea833-00f6-e6c5-ce35-ee68297872bb`) with the site's coordinates and a lead time of one or two days, export its direct and diffuse radiation, shortwave radiation and cloud cover, timestamped by `value.forecasted_for`, and add the export as a second operator input. The export keeps only what the import's Kafka topic still retains, so its history has to be checked before a run depends on it.

---

# ode-ablation-pv-l0-haiku

An analytics operator for the SENERGY platform, scaffolded by the Operator
Development Environment. Every file here is yours to change, including this one.

## Layout

| File | What it is |
|---|---|
| "main.py" | Entry point of the deployed operator. Hands the process to Operator Lib. |
| "train.py" | Entry point of an experiment. Trains through Operator Lib, then exits. |
| "op.py" | The operator: "infer", "train", "need_retraining", and its config. |
| "training.py" | The Ray training pass and the model MLflow registers. |
| "pyproject.toml" | Dependencies, with Operator Lib pinned at "v1.8.1". |
| "uv.lock" | The resolved dependencies. Written by the scaffold; refresh it yourself. See below. |
| "Dockerfile" | The image. Built by CI; buildable by hand. |
| ".github/workflows/build.yml" | Builds and pushes "ghcr.io/senergy-platform/ode-ablation-pv-l0-haiku". Change the registry here. |
| "operator.yaml" | What the analytics stack registers: inputs, outputs, config. |
| "evaluation.yaml" | Your criteria for whether a run is good, plus what Operator Lib needs to score a test window itself. ODE never writes this. |

## The lock file

The scaffold ran "uv lock" for you and "uv.lock" is in this working copy, uncommitted
like everything else here. Commit it with the rest.

Refresh it whenever you change a dependency in "pyproject.toml", and commit the two
together:

    uv lock

An experiment runs "uv run python train.py" on the cluster, and uv builds the
environment from "pyproject.toml" and this file — on the Ray head for the driver and
on each worker node for the tasks, out of its own cache.

Without a lock file uv resolves at run time, which works and is worse in one
specific way: the run records a commit SHA as the code that produced it, and two
runs of the same commit can then resolve different dependency versions. The lock
file is what makes the recorded SHA describe the whole run rather than only its
source. That is why it is not left to be remembered — and if the scaffold reported
that it could not write one, the command above is the repair.

## Building by hand

    docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t ghcr.io/senergy-platform/ode-ablation-pv-l0-haiku:dev .

## The Operator Lib pin

"pyproject.toml" pins Operator Lib at "v1.8.1", the newest at the time
this repository was scaffolded. The library tracks latest and promises no
stability, so moving the pin is a deliberate edit — change it, run "uv lock", and
commit the two together.
