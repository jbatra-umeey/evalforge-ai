import os
import tempfile
from pathlib import Path

# Isolate test database before app import.
_tmp = tempfile.TemporaryDirectory()
os.environ['DATABASE_URL'] = 'sqlite:///' + str(Path(_tmp.name) / 'tests.db')
from fastapi.testclient import TestClient
from app.main import app


def test_health():
    with TestClient(app) as client:
        assert client.get('/health').json() == {'status': 'ok'}


def test_golden_dataset_and_persistence():
    with TestClient(app) as client:
        data = {'cases': [{'case_id': '1', 'query': 'x', 'answer': 'RAG uses retrieval', 'required_facts': ['retrieval']}]}
        out = client.post('/api/evaluate', json=data)
        assert out.status_code == 200
        assert out.json()['passed'] is True
        rid = out.json()['run_id']
        assert client.get(f'/api/runs/{rid}').json()['result']['passed'] is True
        assert any(r['id'] == rid for r in client.get('/api/runs').json())


def test_fail_closed_without_assertions():
    with TestClient(app) as client:
        out = client.post('/api/evaluate', json={'cases': [{'case_id': 'none', 'query': 'q', 'answer': 'a'}]}).json()
        assert out['passed'] is False
        assert out['score'] == 0


def test_wrong_trajectory_and_banned_term():
    with TestClient(app) as client:
        data = {'cases': [{'case_id': 't', 'query': 'q', 'answer': 'This is guaranteed', 'forbidden_terms': ['guaranteed'], 'expected_tools': ['retrieve'], 'trajectory': [{'tool': 'delete_all'}]}]}
        out = client.post('/api/evaluate', json=data).json()
        assert out['passed'] is False
        assert out['cases'][0]['checks']['trajectory'] is False


def test_configured_auth():
    os.environ['EVALFORGE_API_KEY'] = 'test-secret'
    try:
        with TestClient(app) as client:
            assert client.get('/api/runs').status_code == 401
            assert client.get('/api/runs', headers={'x-api-key': 'test-secret'}).status_code == 200
    finally:
        os.environ.pop('EVALFORGE_API_KEY', None)


def test_missing_judge_configuration():
    with TestClient(app) as client:
        assert client.post('/api/judge', json={'query': 'q', 'answer': 'a'}).status_code == 503
