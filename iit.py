import torch
import random
import numpy as np


def generate_data(num_examples=500, num_addends=4, embed_dim=50, seed=0, lower=-10, upper=10):
    random.seed(seed)
    addends = np.array([np.random.randint(low=lower, high=upper, size=num_addends) for _ in range(num_examples)])
    sums = np.array([np.sum(addends[i]) for i in range(num_examples)])

    return addends, sums

    '''
    addends_one_hot = []
    sums_one_hot = [] 

    addends_one_hot = np.array([np.array([one_hot_encode(addend, lower, upper) for addend in addends[i]]) for i in range(num_examples)])
    sums_one_hot = np.array([one_hot_encode(sum, num_addends * lower, num_addends * upper)  for sum in sums])

    return addends_one_hot, sums_one_hot
    '''

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
