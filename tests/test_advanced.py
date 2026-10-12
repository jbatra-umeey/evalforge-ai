import asyncio
import os
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from app.advanced import CalibrationRequest, EnsembleRequest, calibrate, ensemble, require_enterprise_auth


def test_calibration_known_scores():
    req = CalibrationRequest(items=[{'human_score': x, 'judge_scores': [x, x, x]} for x in (.2, .5, .9)])
    assert calibrate(req)['calibrated'] is True
    assert calibrate(req)['mae'] == 0


def test_calibration_rejects_misalignment():
    req = CalibrationRequest(items=[{'human_score': 1, 'judge_scores': [0, 0, 0]}] * 3)
    assert calibrate(req)['calibrated'] is False


def test_ensemble_disagreement_requires_review():
    async def fake_judge(system, data, model_override=None):
        return {'score': {'a': .95, 'b': .95, 'c': .1}[model_override], 'reason': 'test'}
    req = EnsembleRequest(query='q', answer='a', models=['a', 'b', 'c'])
    output = asyncio.run(ensemble(req, fake_judge))
    assert output['human_review_required'] is True
    assert output['passed'] is False


def test_ensemble_identical_models_rejected():
    async def fail(*args, **kwargs):
        raise AssertionError('should not call provider')
    with pytest.raises(HTTPException) as exc:
        asyncio.run(ensemble(EnsembleRequest(query='q', answer='a', models=['a', 'a', 'b']), fail))
    assert exc.value.status_code == 422


def test_production_requires_oidc():
    with patch.dict(os.environ, {'APP_ENV': 'production'}, clear=True):
        with pytest.raises(HTTPException) as exc:
            require_enterprise_auth()
        assert exc.value.status_code == 503
