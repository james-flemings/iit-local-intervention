import torch
import random
import numpy as np
import itertools

def generate_data(ran: list[int], num_addends: int = 4, seed: int = 0):
    random.seed(seed)
    addends = list(itertools.product(ran, repeat=num_addends))
    sums = [np.sum(addend) for addend in addends]

    return addends, sums


def one_hot_encode(addend, lower=-10, upper=10):
    one_hot_addend = np.zeros(abs(lower) + abs(upper) + 1)
    index = addend - lower 
    one_hot_addend[index] = 1
    return one_hot_addend

def create_embedding(num_examples=500, embed_dim=50, lower=-0.5, upper=0.5):
    m, n = num_examples, embed_dim
    return np.array([random.uniform(lower, upper) for i in range(m*n)]).reshape(m, n)

def get_embed_rep(addends, sums, embed):
    pass
