import json
from ..app import app, schedule

def test_missing_fields():
    with app.test_request_context(data=json.dumps({'t1': '2022-01-01 12:00', 'policy': 'dummy'}), method='POST'):
        response = schedule()
        assert response.status_code == 400
        assert json.loads(response.data) == {'status': 'error', 'message': 'Missing t0 field'}

    with app.test_request_context(data=json.dumps({'t0': '2022-01-01 12:00', 'policy': 'dummy'}), method='POST'):
        response = schedule()
        assert response.status_code == 400
        assert json.loads(response.data) == {'status': 'error', 'message': 'Missing t1 field'}

    with app.test_request_context(data=json.dumps({'t0': '2022-01-01 12:00', 't1': '2022-01-01 12:00'}), method='POST'):
        response = schedule()
        assert response.status_code == 400
        assert json.loads(response.data) == {'status': 'error', 'message': 'Missing policy field'}