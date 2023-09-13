from . import utils
import yaml

def datetime_constructor(loader, node):
    """load a datetime"""
    return utils.str2datetime(loader.construct_scalar(node))

def get_loader():
    # add some useful tags to the loader 
    loader = yaml.SafeLoader
    loader.add_constructor('!datetime', datetime_constructor)
    # ignore unknown tags
    loader.add_multi_constructor('!', lambda loader, tag_suffix, node: None)
    return loader