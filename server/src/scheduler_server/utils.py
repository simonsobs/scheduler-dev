#!/usr/bin/env python3
import numpy as np
import random
import yaml

def split_into_parts(N, m):
    parts = []
    for i in range(m-1):
        parts.append(random.uniform(0, N/m))
        N -= parts[-1]
    parts.append(N)
    random.shuffle(parts)
    return parts

def rand_upto(x):
    return np.random.uniform(low=0, high=x)

def rand_between(x, y):
    return np.random.uniform(low=x, high=y)

def load_config(config_file=None):
    """yaml config file loader"""
    if not config_file: return {}
    with open(config_file, 'r') as file:
        config_data = yaml.safe_load(file)
    if config_data is None:
        config_data = {}
    return config_data

def nested_update(dictionary, update_dict, new_keys_allowed=False):
    """update a nested dictionary recursively but
    never add new keys"""
    for key, value in update_dict.items():
        if key in dictionary and isinstance(dictionary[key], dict) and isinstance(value, dict):
            nested_update(dictionary[key], value)
        else:
            # never add new keys
            if key in dictionary or new_keys_allowed:
                dictionary[key] = value
    return dictionary