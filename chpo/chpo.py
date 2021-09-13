import numpy as np
import math

import hyperopt.pyll.stochastic
from hyperopt import fmin, rand, tpe, hp, Trials

import ConfigSpace as CS
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.core.worker import Worker
from hpbandster.optimizers import HyperBand as HyperBand
from hpbandster.optimizers import BOHB as BOHB

class TPE:
  def __init__(self, space, objective, problem):
    self.space = space
    self.objective = objective
    self.problem = problem
    self.base_budget = 0
    
  def call(self, param):
    return self.objective(param, self.base_budget, self.problem, 'validation')
    
  def run(self, n_parent, n_base):
    self.base_budget = n_base
    trials = Trials()
    best = fmin(self.call, space=self.space, max_evals=n_parent, 
                algo=tpe.suggest, trials=trials, verbose=False, show_progressbar=False)
    loss = self.objective(best, n_parent, self.problem, 'test')

    return loss
    
class CTPE:
  def __init__(self, space, objective, problem, n_buckets):
    self.space = space
    self.objective = objective
    self.problem = problem
    self.n_buckets = n_buckets
    
  def call(self, param):
    return self.objective(param, self.base_budget, self.problem, 'validation')
    
  def run(self, n_parent, n_base):
    trials = Trials()
    bucket_size = n_parent / self.n_buckets
    evals = 0
    
    for bucket in range(1, self.n_buckets + 1):
      self.base_budget = bucket * n_base / self.n_buckets
      
      save = 1 - (self.n_buckets - bucket) / self.n_buckets
      parent_budget = bucket_size + save * bucket_size * self.n_buckets / bucket
      
      evals += int(parent_budget)
      best = fmin(self.call, space=self.space, max_evals=evals,
                  algo=tpe.suggest, trials=trials, verbose=False, show_progressbar=False)
    loss = self.objective(best, n_parent, self.problem, 'test')
    
    return loss
    
def hb(space, objective, problem, n_parent, n_base, eta):
    class HBWorker(Worker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def compute(self, config, budget, **kwargs):
            res = objective(config, budget, problem, 'validation')
            return({'loss': float(res), 'info': res})   

    NS = hpns.NameServer(run_id='hb', host='127.0.0.1', port=9000)
    NS.start()
    
    w = HBWorker(nameserver='127.0.0.1', run_id='hb', nameserver_port=9000)
    w.run(background=True)
    
    hb = HyperBand(configspace=space, run_id='hb', nameserver='127.0.0.1', nameserver_port=9000, min_budget=1, max_budget=n_base, eta=eta)
    res = hb.run(n_iterations=n_parent)
    
    hb.shutdown(shutdown_workers=True)
    NS.shutdown()
    
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()
    best = id2config[incumbent]['config']
    loss = objective(best, n_base, problem, 'test')
    
    return loss
    
def bohb(space, objective, problem, n_parent, n_base, eta):
    class BOHBWorker(Worker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def compute(self, config, budget, **kwargs):
            res = objective(config, budget, problem, 'validation')
            return({'loss': float(res), 'info': res})   

    NS = hpns.NameServer(run_id='hb', host='127.0.0.1', port=9000)
    NS.start()
    
    w = BOHBWorker(nameserver='127.0.0.1', run_id='hb', nameserver_port=9000)
    w.run(background=True)
    
    bohb = BOHB(configspace=space, run_id='hb', nameserver='127.0.0.1', nameserver_port=9000, min_budget=1, max_budget=n_base, eta=eta)
    res = bohb.run(n_iterations=n_parent)
    
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()
    
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()
    best = id2config[incumbent]['config']
    loss = objective(best, n_base, problem, 'test')
    
    return loss
