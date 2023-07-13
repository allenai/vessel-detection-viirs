"""Monitoring module for the vvd service
"""
import os
from typing import Callable

from prometheus_client import Histogram
from prometheus_client.utils import INF
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from prometheus_fastapi_instrumentator.metrics import Info

NAMESPACE = os.environ.get("METRICS_NAMESPACE", "machine_learning")
SUBSYSTEM = os.environ.get("METRICS_SUBSYSTEM", "")

instrumentator = Instrumentator()
buckets = (0, 1, 2, 3, 4, 5, 10, 100, INF)

DETECTION_METRIC = Histogram(
    "number_of_detection",
    "vvd model number of detection",
    buckets=buckets,
    namespace=NAMESPACE,
    subsystem=SUBSYSTEM,
    labelnames=["type", "detection", "operator"],
)


def vvd_model_moonlight() -> Callable[[Info], None]:
    """prometheus instrumentation for the vvd model for moonlight
    Returns
    -------
    Callable[[Info], None]
        _description_
    """

    def instrumentation(info: Info) -> None:
        if info.modified_handler == "/detections":
            n_detections = info.response.headers.get("avg_moonlight")
            if n_detections:
                DETECTION_METRIC\
                    .labels(type="viirs", detection="avg_moonlight", operator="avg")\
                    .observe(float(n_detections))

    return instrumentation


def vvd_model_gas_flare_count() -> Callable[[Info], None]:
    """
    Returns
    -------
    Callable[[Info], None]
        _description_
    """

    def instrumentation(info: Info) -> None:
        if info.modified_handler == "/detections":
            n_detections = info.response.headers.get("gas_flare_count")
            if n_detections:
                DETECTION_METRIC\
                    .labels(type="viirs", detection="gas_flare_count", operator="sum")\
                    .observe(float(n_detections))

    return instrumentation


def vvd_model_lightning_count() -> Callable[[Info], None]:
    """
    Returns
    -------
    Callable[[Info], None]
        _description_
    """

    def instrumentation(info: Info) -> None:
        if info.modified_handler == "/detections":
            n_detections = info.response.headers.get("lightning_count")
            if n_detections:
                DETECTION_METRIC\
                    .labels(type="viirs", detection="lightning_count", operator="sum")\
                    .observe(float(n_detections))

    return instrumentation


def vvd_model_detections_output() -> Callable[[Info], None]:
    """
    Returns
    -------
    Callable[[Info], None]
        _description_
    """

    def instrumentation(info: Info) -> None:
        if info.modified_handler == "/detections":
            n_detections = info.response.headers.get("n_detections")
            if n_detections:
                DETECTION_METRIC\
                    .labels(type="viirs", detection="vessels", operator="sum")\
                    .observe(float(n_detections))

    return instrumentation


# ----- add metrics -----
instrumentator.add(
    metrics.request_size(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)
instrumentator.add(
    metrics.response_size(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)
instrumentator.add(
    metrics.latency(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)
instrumentator.add(
    metrics.requests(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)

instrumentator.add(
    vvd_model_moonlight()
)

instrumentator.add(
    vvd_model_detections_output()
)

instrumentator.add(
    vvd_model_lightning_count()
)

instrumentator.add(
    vvd_model_gas_flare_count()
)
