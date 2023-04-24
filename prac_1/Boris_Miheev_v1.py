# There should be no main() in this file!!! 
# Nothing should start running when you import this file somewhere.
# You may add other supporting functions to this file.
#
# Important rules:
# 1) Function pa_bc must return tensor which has dimensions (#a x #b x #c),
#    where #v is a number of different values of the variable v.
#    For input variables #v = how many input values of this variable you give to the function.
#    For output variables #v = number of all possible values of this variable.
#    Ex. for pb_a: #b = bmax-bmin+1,   #a is arbitrary.
# 2) Random variables in function names must be written in alphabetic order
#    e.g. pda_cb is an improper function name (pad_bc must be used instead)
# 3) Single dimension must be explicitly stated:
#    if you give only one value of a variable a to the function pb_a, i.e. #a=1, 
#    then the function pb_a must return tensor of shape (#b, 1), not (#b,).
#
# The format of all the functions for distributions is the following:
# Inputs:
# params - dictionary with keys 'amin', 'amax', 'bmin', 'bmax', 'p1', 'p2', 'p3'
# model - model number, number from 1 to 4
# all other parameters - values of the conditions (variables a, b, c, d).
#                        Numpy vectors of size (k,), where k is an arbitrary number.
#                        For variant 3: c and d must be numpy arrays of size (k,N),
#                        where N is a number of lectures.
# Outputs:
# prob, val
# prob - probabilities for different values of the output variable with different input conditions
#        prob[i,...] = p(v=val[i]|...)
# val - support of a distribution, numpy vector of size (#v,) for variable v
#
# Example 1:
#    Function pc_ab - distribution p(c|a,b)
#    Input: a of size (k_a,) and b of size (k_b,)
#    Result: prob of size (cmax-cmin+1,k_a,k_b), val of size (cmax-cmin+1,) 
#
# Example 2 (for variant 3):
#    Function pb_ad - distribution p(b|a,d_1,...,d_N)
#    Input: a of size (k_a,) and d of size (k_d,N)
#    Result: prob of size (bmax-bmin+1,k_a,k_d), val of size (bmax-bmin+1,)
#
# The format the generation function from variant 3 is the following:
# Inputs:
# N - how many points to generate
# all other inputs have the same format as earlier
# Outputs:
# d - generated values of d, numpy array of size (N,#a,#b)

# In variant 1 the following functions are required:

import numpy as np
from scipy.stats import binom, poisson
from scipy.special import logsumexp

def pa(params, model): 
    prob = np.full((params['amax'] - params['amin'] + 1,), 1 / (params['amax'] - params['amin'] + 1))
    val = np.arange(params['amin'], params['amax'] + 1)
    return prob, val
    
def pb(params, model):
    prob = np.full((params['bmax'] - params['bmin'] + 1,), 1 / (params['bmax'] - params['bmin'] + 1))
    val = np.arange(params['bmin'], params['bmax'] + 1)
    return prob, val

def pc_ab(a, b, params, model):  
    val = np.arange(params['amax'] + params['bmax'] + 1)
    prob = np.zeros((val.shape[0], a.shape[0], b.shape[0]))
    #Use broadcasting and fictional axes to get tensors with shape of (#c, #a) and (#c, #b) respectively
    #Idea was inspired by:
    #https://stackoverflow.com/questions/49282238/calculate-binomial-distribution-probability-matrix-with-python
    if model == 1:
        bins_a = binom.pmf(val[:, None], a, params['p1'])
        bins_b = binom.pmf(val[:, None], b, params['p2'])
    elif model == 2:
        bins_a = poisson.pmf(val[:, None], a * params['p1'])
        bins_b = poisson.pmf(val[:, None], b * params['p2'])
    for c in val:
        prob[c] = np.dot(bins_a[:c + 1].T, bins_b[:c + 1][::-1])
    
    return prob, val


def pc(params, model):
    range_a = np.arange(params['amin'], params['amax'] + 1)
    range_b = np.arange(params['bmin'], params['bmax'] + 1)
    pc_ab_vals = pc_ab(range_a, range_b, params, model)[0]
    prob = np.sum(pc_ab_vals, axis=(1, 2)) * pa(params, model)[0][0] * pb(params, model)[0][0]
    val = np.arange(params['amax'] + params['bmax'] + 1)
    return prob, val
    
def pc_a(a, params, model):
    range_b = np.arange(params['bmin'], params['bmax'] + 1)
    pc_ab_vals = pc_ab(a, range_b, params, model)[0]
    prob = np.sum(pc_ab_vals, axis=2) * pb(params, model)[0][0]
    val = np.arange(params['amax'] + params['bmax'] + 1)
    return prob, val

def pc_b(b, params, model):
    range_a = np.arange(params['amin'], params['amax'] + 1)
    pc_ab_vals = pc_ab(range_a, b, params, model)[0]
    prob = np.sum(pc_ab_vals, axis=1) * pa(params, model)[0][0]
    val = np.arange(params['amax'] + params['bmax'] + 1)
    return prob, val

def pd_c(c, params, model):
    val = np.arange(2 * (params['amax'] + params['bmax']) + 1)
    prob = np.zeros((val.shape[0], c.shape[0]))
    #Use broadcasting and fictional axes to get tensor with shape (#d, #c) with all (d-c) values combinations
    #Idea was inspired by:
    #https://stackoverflow.com/questions/49282238/calculate-binomial-distribution-probability-matrix-with-python 
    prob = binom.pmf(val[:, None] - c, c, params['p3'])
    return prob, val

def pd(params, model):
    range_c = np.arange(params['amax'] + params['bmax'] + 1)
    pd_c_vals = pd_c(range_c, params, model)[0]
    prob = np.sum(pd_c_vals * pc(params, model)[0], axis=1)
    val = np.arange(2 * (params['amax'] + params['bmax']) + 1)
    return prob, val

def pc_d(d, params, model):
    val = np.arange(params['amax'] + params['bmax'] + 1)
    prob = pd_c(val, params, model)[0][d] * pc(params, model)[0]
    pd_vals = np.sum(prob, axis=1).reshape((prob.shape[0], 1))
    arg = np.argwhere(pd_vals == 0)
    #Add machine eps to null values to avoid NaNs occurence
    pd_vals[arg] += np.finfo(float).eps
    prob /= pd_vals
    #prob = np.nan_to_num(prob, 0)
    return prob.T, val

    
def pc_abd(a, b, d, params, model):
    val = np.arange(params['amax'] + params['bmax'] + 1)
    pd_c_vals = pd_c(val, params, model)[0][d]
    pc_ab_vals = pc_ab(a, b, params, model)[0]
    prob = np.zeros((d.shape[0], val.shape[0], a.shape[0], b.shape[0]))
    #Use broadcasting and fictional axes to get tensor with shape of (#d, #c, #a, #b)
    #Idea was inspired by:
    #https://stackoverflow.com/questions/49282238/calculate-binomial-distribution-probability-matrix-with-python 
    #Also it is necessary to place axes in right order for correct multiplication: d, c, a, b
    prob = pd_c_vals[:, :, None, None] * pc_ab_vals[None, :, :, :]
    denominator = np.sum(prob, axis=1)[:, None, :, :]
    prob = prob / denominator
    #Place axes in required order: c, a, b, d
    prob = np.transpose(prob, axes=[1, 2, 3, 0])
    return prob, val
