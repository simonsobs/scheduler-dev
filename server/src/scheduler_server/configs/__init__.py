from .basic import config as basic_config

CONFIGS = {
    'basic': basic_config 
}
def get_default_config(name):
    return CONFIGS.get(name, {})