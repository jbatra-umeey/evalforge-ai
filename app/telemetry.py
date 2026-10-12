"""Optional OTLP traces; endpoint is configured only via trusted environment."""
import os

def configure_telemetry(app):
    endpoint = os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT')
    if not endpoint:
        return
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    provider = TracerProvider(resource=Resource.create({'service.name': 'evalforge-api'}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint.rstrip('/') + '/v1/traces')))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
