"""Create a dummy server for testing purposes."""

import os
from pathlib import Path
import flask
import flask_cors
from datetime import datetime, timezone
import json
import traceback

from . import handler, utils

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

    # policy is a json string, so parse it now. The json
    # has the form {"policy": "policy_name", "config": {...}}
    try:
        policy_dict = json.loads(policy)
    except json.JSONDecodeError:
        response = flask.jsonify({
            'status': 'error',
            'message': 'Invalid policy, needs to be a json string'
        })
        response.status_code = 400
        return response 

    policy_name = policy_dict.get('policy', 'dummy')
    user_policy_config = policy_dict.get('config', {})

    # check policy is supported
    if policy_name not in SUPPORTED_POLICIES:
        response = flask.jsonify({
            'status': 'error',
            'message': f'Invalid policy. Supported policies are: {SUPPORTED_POLICIES}'
        })
        response.status_code = 400
        return response

    try:
        # load default config for the selected policy
        policy_config_file = app.config['policy_configs'][policy_name]
        policy_config = utils.load_config(policy_config_file)

        # merge user config with default config
        utils.nested_update(policy_config, user_policy_config)
        commands = POLICY_HANDLERS[policy](t0, t1, policy_config, app.config)
    except Exception as e:
        response = flask.jsonify({
            'status': 'error',
            'message': f'Error: {e}'
        })
        response.status_code = 500
        app.logger.error(traceback.format_exc())
        return response

    response = flask.jsonify({
        'status': 'ok',
        'commands': commands,
        'message': 'Success'
    })
    response.status_code = 200
    return response

# load config
default_config_file = Path(__file__).parent / 'config.yaml'
config_file = os.environ.get('SCHEDULER_CONFIG', default_config_file)
config = utils.load_config(config_file)
app.config.update(config)
