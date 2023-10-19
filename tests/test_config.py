import yaml
import pytest
import datetime as dt
from schedlib import config as cfg

@pytest.fixture
def config_str():
    return """
    date_ref: !datetime 2014-01-01 00:00:00
    """

def test_loader(config_str):
    loader = cfg.get_loader()
    config = yaml.load(config_str, Loader=loader)
    assert config['date_ref'] == dt.datetime(2014, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)