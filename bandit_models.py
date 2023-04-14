#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:01:34 2020

@author: celia
"""
import pandas as pd
import numpy as np
import os
from tqdm.auto import trange

# Fit an empirical lookup table model to the data (without symmetry)
from collections import defaultdict
from itertools import count, product
from sklearn.linear_model import LogisticRegression


# general behavioral features characterization

def count_actions(sessions, memory, model_predict=False, predict=None):
    
    '''label action-outcome sequences for memory length and count frequency of occurrences alongside choices'''
    counts = defaultdict(int)
    switches = defaultdict(int)
    actions = defaultdict(int)
    
    if model_predict:
        
        idx=0
        for i in trange(len(sessions),disable=True):
            choices, rewards = sessions[i]
            choice_predict = predict[i]
            for t in range(memory, len(choices)):
                key = (tuple(choices[t-memory:t]), tuple(rewards[t-memory:t]))
                actions[key] += choice_predict[t]
                counts[key] += 1
                switches[key] += choice_predict[t]!=key[0][-1]
                #switches[key] += predict[1][idx]
                idx+=1
    
    elif type(sessions)==pd.DataFrame:
        for i in sessions.Session.unique():
            session = sessions[sessions.Session==i].copy()
            choices = session.Decision.values
            rewards = session.Reward.values
    
            for t in range(memory, len(choices)):
                key = (tuple(choices[t-memory:t]), tuple(rewards[t-memory:t]))
                actions[key] += choices[t]
                counts[key] += 1
                switches[key] += choices[t]!=key[0][-1]
    
    else:

        for i in trange(len(sessions),disable=True):
            choices, rewards = sessions[i]
            for t in range(memory, len(choices)):
                key = (tuple(choices[t-memory:t]), tuple(rewards[t-memory:t]))
                actions[key] += choices[t]
                counts[key] += 1
                switches[key] += choices[t]!=key[0][-1]
                
    return actions, counts, switches


def make_empirical_policy(sessions, memory, model_predict=False, predict=None, smooth=1e-4):
    
    """now compute probabilities for each sequence using action counts over total occurrences of that sequence"""
    
    actions, counts, switches = count_actions(sessions, memory, model_predict, predict)
    
    # normalize actions to make a probability
    policy = defaultdict(lambda: 0.5)
    switch_policy = defaultdict(float)
    for key in actions:
        policy[key] = (actions[key] + smooth) / (counts[key] + 2 * smooth)
        switch_policy[key] = switches[key] / (counts[key]+smooth)
    
    return policy, switch_policy, counts

def flip(key):
    
    """mapping for inverted choice labeling, i.e. RrL--> LlR"""
    
    return tuple(map(lambda x: 1 - x, key))


def make_symm_empirical_policy(sessions, memory, ll=False, model_predict=False, predict=None, smooth=1e-4):
    
    '''compress right-left history notation to a-b notation. Initialize all possible history sequences and 
    initialize policy at 0.5. This results in some history sequences with n=0, can be dropped later).'''
    
    actions, counts, switches = count_actions(sessions, memory, model_predict, predict)

    symm_actions = defaultdict(int)
    symm_counts = defaultdict(int)
    symm_switches = defaultdict(int)

    for choice_key in product(*([(0, 1)] * memory)):
        for reward_key in product(*([(0, 1)] * memory)):
            key = (choice_key, reward_key)

            # Add actions/counts for this key
            symm_actions[key] += actions[key]
            symm_counts[key] += counts[key]
            symm_switches[key] += switches[key]

            # Add its symmetric counterpart
            flipped_key = (flip(choice_key), reward_key)
            symm_actions[key] += counts[flipped_key] - actions[flipped_key]
            symm_counts[key] += counts[flipped_key]
            symm_switches[key] += switches[flipped_key]

    # normalize actions to make a probability
    policy = defaultdict(lambda: 0.5)
    policy_switch = defaultdict(lambda: 0.5)
    
    for key in symm_actions:
        policy[key] = (symm_actions[key] + smooth) / (symm_counts[key] + 2 * smooth)
        policy_switch[key] = (symm_switches[key] + smooth) / (symm_counts[key] + 2 * smooth)
        
    # Create a symmetric policy with only the non-redundant keys
    policy_compressed = defaultdict(float)
    policy_switch_compressed = defaultdict(float)
    counts_compressed = defaultdict(float)

    # eliminate redundancy from symmetrical policy labeling
    for key in policy:
        choice_key, reward_key = key
        if choice_key[0] == 1:
            policy_compressed[key] = policy[key] # prob choosing 'A'
            policy_switch_compressed[key] = policy_switch[key]
            counts_compressed[key] = symm_counts[key]
    
    if ll:
        return policy, policy_switch, counts
    else:
        return policy_compressed, policy_switch_compressed, counts_compressed


def tick(key):
    mapping = {(0,0): 'b', (0,1): 'B', (1,0): 'a', (1,1): 'A'} 
    return ''.join([mapping[(c,r)] for c,r in zip(*key)])


def lr_tick(key):
    mapping = {(1,0): 'l', (1,1): 'L', (0,0): 'r', (0,1): 'R'}
    return ''.join([mapping[(c,r)] for c,r in zip(*key)])


from scipy.stats import bernoulli
def log_likelihood_empirical_policy(policy, test_sessions, memory):
    ll = 0
    n = 0
    for i in trange(len(test_sessions),disable=True):
        choices, rewards = test_sessions[i]
        for t in range(memory, len(choices)):
            key = (tuple(choices[t-memory:t]), tuple(rewards[t-memory:t]))
            # ll += bernoulli.logpmf(choices[t], policy[key])
            ll += choices[t] * np.log(policy[key]) + (1 - choices[t]) * np.log(1 - policy[key])
            n += 1
    return ll / n

# logistic regression functions
pm1 = lambda x: 2 * x - 1
feature_functions = [
    lambda cs, rs: pm1(cs),                # choices
    lambda cs, rs: rs,                     # rewards
    lambda cs, rs: pm1(cs) * rs,           # +1 if choice = 1 and reward, 0 if no reward, -1 if choice=0 and reward
    lambda cs, rs: np.ones(len(cs))        # overall bias term
    
]

# Helper to encode sessions in features and outcomes
def encode_session(choices, rewards, memories, featfun):
    assert len(memories) == len(featfun)  
    
    # Construct the features
    features = []
    for fn, memory in zip(featfun, memories): 
        for lag in range(1, memory+1):
            # encode the data and pad with zeros
            x = fn(choices[:-lag], rewards[:-lag])
            x = np.concatenate((np.zeros(lag), x))
            features.append(x)
    features = np.column_stack(features)
    return features, choices
    
def fit_logreg_policy(sessions, memories, featfun=feature_functions):
    encoded_sessions = [encode_session(*session, memories, featfun=featfun) for session in sessions]
    X = np.row_stack([session[0] for session in encoded_sessions])
    y = np.concatenate([session[1] for session in encoded_sessions])
    
    # Construct the logistic regression model and fit to training sessions
    lr = LogisticRegression(C=1.0, fit_intercept=False)
    lr.fit(X, y)
    return lr

# Evaluate log odds
def compute_logreg_probs(sessions, lr_args, featfun=feature_functions):
    lr, memories = lr_args
    
    all_probs = []
    for choices, rewards in sessions:
        X, y = encode_session(choices, rewards, memories, featfun=featfun)
        probs = lr.predict_proba(X)#[:, 1]
        all_probs.append(probs)
    return all_probs

# Evaluate log likelihood
def log_likelihood_logreg_policy(sessions, lr, memories, featfun=feature_functions):
    n = 0
    ll = 0
    for choices, rewards in sessions:
        X, y = encode_session(choices, rewards, memories, featfun=featfun)
        p = lr.predict_proba(X)[:, 1]
        ll += bernoulli.logpmf(y, p).sum() 
        n += y.size
    return (ll / n)#, bernoulli.logpmf(y, p)


def expected_reward(policies, targets, ptrue, choice_policy='stochastic'):
    import itertools
    
    cum_rewards = 0
    session_rewards = []
    for i in trange(len(policies), disable=True):
        t = targets[i] # targets for session
        posteriors = policies[i] # hmm beliefs for session
        
        # choose direction Thompson sampling on policy
        if choice_policy=='stochastic':
            choices = [np.random.rand() < posteriors[n,1] for n in range(len(posteriors))]
        elif choice_policy=='greedy':
            choices = [0.5 <= posteriors[n,1] for n in range(len(posteriors))]
        prs = [ptrue if np.equal(choices[n],t[n]) else (1-ptrue) for n in range(len(t))]
        
        # observe outcome
        rewards = [np.random.rand() < prs[n] for n in range(len(prs))]
        cum_rewards += np.sum(rewards)
        session_rewards.append(np.mean(rewards))
        
    return (cum_rewards / len(list(itertools.chain(*targets)))), session_rewards
        
def single_state_model_sessions(df, parameters):
    from scipy.special import expit as sigmoid

    alpha, beta, tau = parameters  # unpack parameters
    
    gamma = np.exp(-1 / tau)
    
    # compute probability of next choice
    psi_sessions=[]

    for choices, rewards in df:
        psi=np.zeros((len(choices), 2))
        psi[0,:]=[0.5,0.5]
    
        # recode choices
        cbar = 2 * choices - 1
        # initialize "belief state"
        phi = beta * rewards[0] * cbar[0]
        for t in range(1, len(choices)):
            # evaluate probability of this choice
            psi[t,:] = 1-sigmoid(phi + (alpha * cbar[t-1])),sigmoid(phi + (alpha * cbar[t-1]))
            
            # update belief state for next time step
            phi = gamma * phi + (beta*(rewards[t] * cbar[t])) 

        psi_sessions.append(psi)

    return psi_sessions