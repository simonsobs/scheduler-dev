"""basic backend service for serving any frontend"""

from datetime import datetime
import requests, os

from sosched import config
from sosched.schedule import Schedule, get_site
from sosched.scans import ScanSet
from sosched.commands import CommandWriter
from sosched.procedures import add_calibration_targets, add_soft_targets, resolve_overlap, sun_avoidance

from flask import Flask, request
from flask_cors import CORS

# preload schedule
config.init("../config.yaml")
master_format = config.get("master_format", "toast")
master_file = config.get("master")
if master_format == 'toast':
    sched = Schedule.from_toast_txt(master_file)
elif master_format == 'act':
    sched = Schedule.from_act_txt(master_file)
else:
    raise ValueError(f"Unknown master format: {master_format}") 
sset = sched.to_scanset()
site = get_site(config.get("site"))

# web app
app = Flask(__name__)
CORS(app)

@app.route('/schedule/', methods=['POST'])
def schedule():
    """return a schedule"""
    data = request.get_json()
    t0 = data['t0']
    t1 = data['t1']
    # parse into datetime objects
    t0 = datetime.strptime(t0, "%Y-%m-%d %H:%M")
    t1 = datetime.strptime(t1, "%Y-%m-%d %H:%M")
    # convert to unix time
    t0 = int(datetime.strftime(t0, "%s"))
    t1 = int(datetime.strftime(t1, "%s"))
    scanset = sset.get_tslice(t0, t1)
    # process scanset
    if data['sun_avoidance']:
        scanset = sun_avoidance(scanset, site)
    if data['add_calibration_targets']:
        scanset = add_calibration_targets(scanset, site)
    if data['soft_targets']:
        scanset = add_soft_targets(scanset, site)
    scanset = resolve_overlap(scanset)
    response = scanset.to_json()
    return response

@app.route('/commands/', methods=['POST'])
def commands():
    """return sequencer commands"""
    data = request.get_json()
    scanset = ScanSet.from_json(data['sset'])
    writer = CommandWriter(scanset)
    return str(writer.get_commands())

@app.route('/sunavoidance/', methods=['POST'])
def sunavoidance():
    """remove overlap in scanset and return it"""
    data = request.get_json()
    scanset = ScanSet.from_json(data['sset'])
    scanset = sun_avoidance(scanset, site)
    return scanset.to_json()

@app.route('/calibration/', methods=['POST'])
def calibration():
    """add calibration targets in scanset and return it"""
    data = request.get_json()
    scanset = ScanSet.from_json(data['sset'])
    scanset = add_calibration_targets(scanset, site)
    return scanset.to_json()

@app.route('/soft/', methods=['POST'])
def soft():
    """add calibration targets in scanset and return it"""
    data = request.get_json()
    scanset = ScanSet.from_json(data['sset'])
    scanset = add_soft_targets(scanset, site)
    return scanset.to_json()

@app.route('/overlap/', methods=['POST'])
def overlap():
    """remove overlap in scanset and return it"""
    data = request.get_json()
    scanset = ScanSet.from_json(data['sset'])
    scanset = resolve_overlap(scanset)
    return scanset.to_json()

@app.route('/inject/', methods=['POST'])
def inject():
    """remove overlap in scanset and return it"""
    data = request.get_json()
    commands = data['commands']
    query = {
        "query": 'mutation { reset(statement: %s) }' % repr(commands)
    }
    token = os.environ.get("NEXTLINE_TOKEN")
    if not token:  raise ValueError("NEXTLINE_TOKEN not defined")
    header = {"Authorization" : f"Basic {token}"}
    response = requests.post("https://grumpy.physics.yale.edu/site/nextline2/api/", json=query, headers=header)
    print(response.text)
    return response.text