"""Create a dummy server for testing purposes."""

import flask
import flask_cors
from datetime import datetime
import random

SUPPORTED_POLICIES = ['dummy']

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
        t0 = datetime.strptime(t0, "%Y-%m-%d %H:%M")
        t1 = datetime.strptime(t1, "%Y-%m-%d %H:%M")
    except ValueError:
        response = flask.jsonify({
            'status': 'error',
            'message': 'Invalid date format'
        })
        response.status_code = 400
        return response

    dt = abs(t1.timestamp() - t0.timestamp())  # too lazy to check for t1<t0 now
    # get current time as a timestamp string
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n = random.randint(1, 10)
    commands = ["import time", f"time.sleep({random.random()*5:.2f})  # {now}"]
    for i in range(n):
        commands.append(f"time.sleep({random.random()*5:.2f})")
    commands = "\n".join(commands)

    response = flask.jsonify({
        'status': 'ok',
        'commands': commands,
        'message': 'Success'
    })
    response.status_code = 200
    return response
