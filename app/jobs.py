"""Celery evaluator: broker Redis; result states stored in Redis backend.
Only JSON-serializable strict eval payloads are accepted; no arbitrary code execution.
"""
import os
from celery import Celery
from app.advanced import GraphRequest, execute_graph

broker = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
backend = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1')
celery_app = Celery('evalforge', broker=broker, backend=backend)
celery_app.conf.update(task_serializer='json', accept_content=['json'], result_serializer='json',
                       task_track_started=True, task_time_limit=120, worker_prefetch_multiplier=1,
                       broker_connection_retry_on_startup=True)

@celery_app.task(name='evalforge.graph', bind=True, max_retries=2)
def graph_task(self, payload: dict):
    return execute_graph(GraphRequest.model_validate(payload))
