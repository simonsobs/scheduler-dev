#!/usr/bin/env python3
import numpy as np
import random

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
