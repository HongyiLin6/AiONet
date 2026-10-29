"""Benefit/harm regression and exported MLP inference."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


def scenario_weights(ids):
    names, counts = np.unique(ids, return_counts=True)
    frequency = dict(zip(names, counts))
    return np.asarray([len(ids)/(len(names)*frequency[s]) for s in ids], np.float32)


def score(model, x):
    x = np.array(x, dtype=np.float32, copy=True)[:, :model['dimensions']]
    if model.get('without_age_horizon'): x[:, [0, 1]] = 0
    if model.get('without_source'): x[:, 20:] = 0
    if not len(x): return np.empty(0)
    if model['kind'] == 'mlp':
        from .reliability_core import DescriptorNormalizer
        norm = DescriptorNormalizer()
        for key, value in model['normalizer'].items(): setattr(norm, key, np.array(value))
        x = norm.transform(x)
        for index, layer in enumerate(model['layers']):
            x = x @ layer['weight'].T + layer['bias']
            x = np.maximum(x, 0) if index < 2 else np.tanh(x)
        return -x.ravel()
    x = np.where(np.isfinite(x), x, model['median'])
    benefit = np.clip(model['heads'][0].predict(x), 0, 1)
    harm = np.clip(model['heads'][1].predict(x), 0, 1)
    return benefit-model['penalty']*harm


def fit(x, delta, scenes, leaves=15, penalty=1., without_age_horizon=False, without_source=False, seed=17):
    x=np.array(x,dtype=np.float32,copy=True);delta=np.asarray(delta,np.float32)
    if x.ndim != 2 or x.shape[1] not in (20,32) or len(x)!=len(delta):
        raise ValueError('Descriptors must have 20 or 32 columns and one loss label per row')
    if not np.isfinite(delta).all() or np.max(np.abs(delta))>1+1e-6:
        raise ValueError('Relative loss must lie in [-1,1]')
    if without_age_horizon:x[:,[0,1]]=0
    if without_source:x[:,20:]=0
    median=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in x.T])
    x=np.where(np.isfinite(x),x,median)
    heads=[]
    for y in (np.maximum(-delta,0),np.maximum(delta,0)):
        reg=HistGradientBoostingRegressor(loss='squared_error',max_iter=200,learning_rate=.05,
            max_leaf_nodes=leaves,min_samples_leaf=32,l2_regularization=1.,early_stopping=False,random_state=seed)
        reg.fit(x,y,sample_weight=scenario_weights(scenes));heads.append(reg)
    return dict(kind='two_head',heads=heads,median=median,penalty=penalty,dimensions=x.shape[1],
        without_age_horizon=without_age_horizon,without_source=without_source,seed=seed,leaves=leaves)
