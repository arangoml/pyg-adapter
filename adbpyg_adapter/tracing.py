from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Iterator, List, Optional, TypeVar, cast

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SpanExporter,
    )
    from opentelemetry.trace import Tracer

    TRACING_ENABLED = True
except ImportError:  # pragma: no cover
    TRACING_ENABLED = False


class TracingManager:
    __tracer: Optional["Tracer"] = None

    @classmethod
    def get_tracer(cls) -> Optional["Tracer"]:
        return cls.__tracer

    @classmethod
    def set_tracer(cls, tracer: "Tracer") -> None:
        cls.__tracer = tracer

    @classmethod
    def set_attributes(self, **attributes: Any) -> None:
        if TRACING_ENABLED and self.__tracer is not None:
            current_span = trace.get_current_span()
            for k, v in attributes.items():
                if isinstance(v, set):
                    v = str(sorted(v))

                elif isinstance(v, dict):
                    v = str(dict(sorted(v.items())))

                # 2D+ List
                elif isinstance(v, list) and any(isinstance(item, list) for item in v):
                    v = str(v)

                current_span.set_attribute(k, v)


T = TypeVar("T", bound=Callable[..., Any])


def with_tracing(method: T) -> T:
    if not TRACING_ENABLED:
        return method  # pragma: no cover

    @wraps(method)
    def decorator(*args: Any, **kwargs: Any) -> Any:
        if tracer := TracingManager.get_tracer():
            with tracer.start_as_current_span(method.__name__):
                return method(*args, **kwargs)

        return method(*args, **kwargs)

    return cast(T, decorator)


@contextmanager
def start_as_current_span(*args: Any, **kwargs: Any) -> Iterator[None]:
    if tracer := TracingManager.get_tracer():
        with tracer.start_as_current_span(*args, **kwargs):
            yield
    else:
        yield


def create_tracer(
    name: str,
    enable_console_tracing: bool = False,
    span_exporters: List["SpanExporter"] = [],
) -> "Tracer":
    """
    Create a tracer instance.

    :param name: The name of the tracer.
    :type name: str
    :param enable_console_tracing: Whether to enable console tracing. Default is False.
    :type enable_console_tracing: bool
    :param span_exporters: A list of SpanExporter instances to use for tracing.
        For example, to export to a local Jaeger instance running via docker, you
        could use `[OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)]`.
    :type span_exporters: List[opentelemetry.sdk.trace.export.SpanExporter]
    :return: A configured tracer instance.
    :rtype: opentelemetry.trace.Tracer
    """
    if not TRACING_ENABLED:  # pragma: no cover
        m = "OpenTelemetry is not installed. Cannot create tracer. Use `pip install adbpyg-adapter[tracing]`"  # noqa: E501
        raise RuntimeError(m)

    resource = Resource(attributes={SERVICE_NAME: name})
    provider = TracerProvider(resource=resource)

    if enable_console_tracing:  # pragma: no cover
        console_processor = BatchSpanProcessor(ConsoleSpanExporter())
        provider.add_span_processor(console_processor)

    for span_exporter in span_exporters:  # pragma: no cover
        provider.add_span_processor(BatchSpanProcessor(span_exporter))

    trace.set_tracer_provider(provider)

    return trace.get_tracer(__name__)
