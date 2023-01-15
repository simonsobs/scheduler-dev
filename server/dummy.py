"""Create a dummy server for testing purposes."""

import flask
import flask_cors

app = flask.Flask(__name__)

# Allow CORS for all domains.
flask_cors.CORS(app)

@app.route('/api/v1/schedule/', methods=['POST'])
def schedule():
    """return a schedule"""
    data = flask.request.get_json()
    t0 = data['t0']
    t1 = data['t1']
    policy = data['policy']

    # parse into datetime objects
    t0 = datetime.strptime(t0, "%Y-%m-%d %H:%M")
    t1 = datetime.strptime(t1, "%Y-%m-%d %H:%M")

    commands = ["import time"]
    # add a random number of sleep commands betweeen 1 and 10 seconds
    for i in range(random.randint(1, 10)):
        commands.append("time.sleep(1)")

    commands = "\n".join(commands)

    response = flask.jsonify({
        'status': 'ok',
        'commands': commands,
        'message': 'Success'
    })
    response.status_code = 200
    return response