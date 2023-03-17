"""Create a dummy server for testing purposes."""

import os, yaml
from pathlib import Path
import flask
import flask_cors
from datetime import datetime, timezone

from . import handler

SUPPORTED_POLICIES = ['dummy', 'basic']
POLICY_HANDLERS = {
    'dummy': handler.dummy_policy,
    'basic': handler.basic_policy
}

app = flask.Flask(__name__)

# Allow CORS for all domains.
flask_cors.CORS(app)

@app.route('/api/v1/schedule/', methods=['POST'])
def schedule():
    """return a schedule"""
    data = flask.request.get_json()

    # check for missing field
    for f in ['t0', 't1', 'policy']:
        if f not in data:
            response = flask.jsonify({
                'status': 'error',
                'message': f'Missing {f} field'
            })
            response.status_code = 400
            return response

    t0 = data['t0']
    t1 = data['t1']
    policy = data['policy']

    # check policy is supported
    if policy not in SUPPORTED_POLICIES:
        response = flask.jsonify({
            'status': 'error',
            'message': f'Invalid policy. Supported policies are: {SUPPORTED_POLICIES}'
        })
        response.status_code = 400
        return response

    # parse into datetime objects
    try:
        t0 = datetime.fromisoformat(t0)
        t1 = datetime.fromisoformat(t1)
        # if no timezone is specified, assume UTC
        if t0.tzinfo is None:
            t0 = t0.replace(tzinfo=timezone.utc)
        if t1.tzinfo is None:
            t1 = t1.replace(tzinfo=timezone.utc)
    except ValueError:
        response = flask.jsonify({
            'status': 'error',
            'message': 'Invalid date format, needs to be ISO format'
        })
        response.status_code = 400
        return response

    commands = POLICY_HANDLERS[policy](t0, t1, app.config)

    response = flask.jsonify({
        'status': 'ok',
        'commands': commands,
        'message': 'Success'
    })
    response.status_code = 200
    return response

# load config
default_config = Path(__file__).parent / 'config.yaml'
config_file = os.environ.get('SCHEDULER_CONFIG', default_config)
with open(config_file, 'r') as file:
    config_data = yaml.safe_load(file)
if config_data is None:
    config_data = {}
app.config.update(config_data)