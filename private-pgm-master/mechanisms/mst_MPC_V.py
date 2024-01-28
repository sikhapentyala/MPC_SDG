import json

import numpy as np
from mbi import FactoredInference, Dataset, Domain
from scipy import sparse
from disjoint_set import DisjointSet
import networkx as nx
import itertools
from cdp2adp import cdp_rho
from scipy.special import logsumexp
import argparse

# Sikha start
from DataHolders import VDataHolder
from MPC import MPCComputations_VP
# Sikha end


"""
This is a generalization of the winning mechanism from the 
2018 NIST Differential Privacy Synthetic Data Competition.

Unlike the original implementation, this one can work for any discrete dataset,
and does not rely on public provisional data for measurement selection.  
"""

def MST(domain, epsilon, delta):
    rho = cdp_rho(epsilon, delta)
    sigma = np.sqrt(3/(2*rho))
    cliques = [(col,) for col in domain]

    domain_size = [domain.project(cl).size() for cl in cliques]
    workload_domain_size = [domain.project(cl).size() for cl in cliques]
    max_domain_size = max(workload_domain_size)
    alice.load_data(cliques)
    alice_workload_ids = alice.workload_ids
    bob.load_data(cliques)
    bob_workload_ids = bob.workload_ids
    alice.compute_answers(cliques,domain,max_domain_size,padded=True)
    bob.compute_answers(cliques,domain,max_domain_size,padded=True)

    mpc_1 = MPCComputations_VP(workload_domain_size, None, alice.workload_answers, bob.workload_answers,None,None,None)
    log1 = measure(mpc_1,cliques,domain_size,alice_workload_ids,bob_workload_ids, sigma)


    #data, log1, undo_compress_fn = compress_domain(data, log1)
    # MPC for select
    workloads = list(itertools.combinations(domain.attrs, 2))
    domain_size = [domain.project(cl).size() for cl in workloads]
    max_domain_size = max(domain_size)
    alice.workload_ids = alice.get_workload_ids(workloads)
    bob.workload_ids = bob.get_workload_ids(workloads)
    alice.compute_answers(workloads,domain,max_domain_size)
    bob.compute_answers(workloads,domain,max_domain_size)

    non_private_workload_ids = []
    non_private_workload_ids.extend(alice.workload_ids)
    non_private_workload_ids.extend(bob.workload_ids)
    workload_ids_to_be_computed_private = set([i for i in range(len(workloads))]) - set(non_private_workload_ids)
    # what are column ids asscoiated with each workload_id
    _column_ids = []
    domain_bins = []
    cols = alice.data.columns
    cols = list(cols.append(bob.data.columns))

    for cl in workload:
        indices = [cols.index(elem) for elem in cl]
        domain_bin = [domain.project(col).size() for col in cl]
        _column_ids.append(indices)
        domain_bins.append(domain_bin)

    mpc_2 = MPCComputations_VP(domain_size, None, alice.workload_answers, bob.workload_answers,workload_ids_to_be_computed_private,_column_ids,domain_bins)
    cliques = select(mpc_2, workloads, alice.data, bob.data,  rho/3.0, log1)

    domain_size = [domain.project(cl).size() for cl in cliques]
    workload_domain_size = [domain.project(cl).size() for cl in cliques]
    max_domain_size = max(workload_domain_size)
    alice.workload_ids = alice.get_workload_ids(cliques)
    bob.workload_ids = bob.get_workload_ids(cliques)
    alice.compute_answers(cliques,domain,max_domain_size,padded=True)
    bob.compute_answers(cliques,domain,max_domain_size,padded=True)

    non_private_workload_ids = []
    non_private_workload_ids.extend(alice.workload_ids)
    non_private_workload_ids.extend(bob.workload_ids)
    workload_ids_to_be_computed_private = set([i for i in range(len(workloads))]) - set(non_private_workload_ids)
    # what are column ids asscoiated with each workload_id
    _column_ids = []
    domain_bins = []
    cols = alice.data.columns
    cols = list(cols.append(bob.data.columns))

    for cl in cliques:
        indices = [cols.index(elem) for elem in cl]
        domain_bin = [domain.project(col).size() for col in cl]
        _column_ids.append(indices)
        domain_bins.append(domain_bin)

    mpc_3 = MPCComputations_VP(workload_domain_size, None, alice.workload_answers, bob.workload_answers,workload_ids_to_be_computed_private,_column_ids,domain_bins)
    log2 = measure(mpc_3,cliques,domain_size,alice.workload_ids,bob.workload_ids, sigma)

    engine = FactoredInference(domain, iters=5000)
    est = engine.estimate(log1+log2)
    synth = est.synthetic_data()
    return synth
    #return undo_compress_fn(synth)

def measure(mpc,cliques,domain_size,alice_workload_ids,bob_workload_ids,sigma, weights=None):
    if weights is None:
        weights = np.ones(len(cliques))
    weights = np.array(weights) / np.linalg.norm(weights)
    measurements = []
    clique_indices = {value: i for i, value in enumerate(cliques)}
    for proj, wgt in zip(cliques, weights):

        marginal_index = clique_indices[proj]
        size = domain_size[marginal_index]
        if marginal_index in alice_workload_ids:
            y = alice.get_noisy_measurement(proj,marginal_index, sigma,domain)
        elif marginal_index in bob_workload_ids:
            y = bob.get_noisy_measurement(proj,marginal_index, sigma,domain)
        else:
            y = mpc.get_noisy_measurement(marginal_index,sigma,alice.data,bob.data)
        Q = sparse.eye(size)
        measurements.append( (Q, y, sigma/wgt, proj) )
    return measurements

def compress_domain(data, measurements):
    supports = {}
    new_measurements = []
    for Q, y, sigma, proj in measurements:
        col = proj[0]
        sup = y >= 3*sigma
        supports[col] = sup
        if supports[col].sum() == y.size:
            new_measurements.append( (Q, y, sigma, proj) )
        else: # need to re-express measurement over the new domain
            y2 = np.append(y[sup], y[~sup].sum())
            I2 = np.ones(y2.size)
            I2[-1] = 1.0 / np.sqrt(y.size - y2.size + 1.0)
            y2[-1] /= np.sqrt(y.size - y2.size + 1.0)
            I2 = sparse.diags(I2)
            new_measurements.append( (I2, y2, sigma, proj) )
    undo_compress_fn = lambda data: reverse_data(data, supports)
    return transform_data(data, supports), new_measurements, undo_compress_fn

def exponential_mechanism(q, eps, sensitivity, prng=np.random, monotonic=False):
    coef = 1.0 if monotonic else 0.5
    scores = coef*eps/sensitivity*q
    probas = np.exp(scores - logsumexp(scores))
    return prng.choice(q.size, p=probas)

def select(mpc, workloads, alice_data, bob_data, rho, measurement_log, cliques=[]):
    engine = FactoredInference(domain, iters=1000)
    est = engine.estimate(measurement_log)

    est_ans = []
    for cl in workloads:
        data_vector = est.project(cl).datavector()
        #padded_data_vector = np.pad(data_vector, (0, max_domain_size - len(data_vector)), 'constant')
        est_ans.append(data_vector)
    mpc.est_ans = est_ans



    T = nx.Graph()
    T.add_nodes_from(domain.attrs)
    ds = DisjointSet()

    for e in cliques:
        T.add_edge(*e)
        ds.union(*e)

    r = len(list(nx.connected_components(T)))
    epsilon = np.sqrt(8*rho/(r-1))
    for i in range(r-1):
        candidates_loop = [e for e in workloads if not ds.connected(*e)]
        candidates_indices = [i for i, value in enumerate(workload) if value in candidates_loop]
        '''
        Sikha start MPC
        '''
        idx = mpc.select_marginal_worst_approximated(candidates_indices,epsilon,alice_data, bob_data, bounded = False,mst=True)
        '''
        Sikha end MPC
        '''
        e = candidates_loop[idx]
        T.add_edge(*e)
        ds.union(*e)

    return list(T.edges)

def transform_data(data, supports):
    df = data.df.copy()
    newdom = {}
    for col in data.domain:
        support = supports[col]
        size = support.sum()
        newdom[col] = int(size)
        if size < support.size:
            newdom[col] += 1
        mapping = {}
        idx = 0
        for i in range(support.size):
            mapping[i] = size
            if support[i]:
                mapping[i] = idx
                idx += 1
        assert idx == size
        df[col] = df[col].map(mapping)
    newdom = Domain.fromdict(newdom)
    return Dataset(df, newdom)

def reverse_data(data, supports):
    df = data.df.copy()
    newdom = {}
    for col in data.domain:
        support = supports[col]
        mx = support.sum()
        newdom[col] = int(support.size)
        idx, extra = np.where(support)[0], np.where(~support)[0]
        mask = df[col] == mx
        if extra.size == 0:
            pass
        else:
            df.loc[mask, col] = np.random.choice(extra, mask.sum())
        df.loc[~mask, col] = idx[df.loc[~mask, col]]
    newdom = Domain.fromdict(newdom)
    return Dataset(df, newdom)

def default_params():
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params['dataset'] = '../data/adult.csv'
    params['domain'] = '../data/adult-domain.json'
    params['epsilon'] = 1.0
    params['delta'] = 1e-9
    params['degree'] = 2
    params['num_marginals'] = None
    params['max_cells'] = 10000

    return params






if __name__ == '__main__':

    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('--dataset', help='dataset to use')
    parser.add_argument('--domain', help='domain to use')
    parser.add_argument('--epsilon', type=float, help='privacy parameter')
    parser.add_argument('--delta', type=float, help='privacy parameter')

    parser.add_argument('--degree', type=int, help='degree of marginals in workload')
    parser.add_argument('--num_marginals', type=int, help='number of marginals in workload')
    parser.add_argument('--max_cells', type=int, help='maximum number of cells for marginals in workload')

    parser.add_argument('--save', type=str, help='path to save synthetic data')

    parser.set_defaults(**default_params())
    args = parser.parse_args()


    # Sikha start
    # data = Dataset.load(args.dataset, args.domain)
    config = json.load(open(args.domain))
    domain = Domain(config.keys(), config.values())
    # Sikha end

    alice = VDataHolder("Alice", args.dataset,"vertical",n=0.5)
    bob = VDataHolder("Bob", args.dataset,"vertical",n=0.5)



    workload = list(itertools.combinations(domain, args.degree))
    workload = [cl for cl in workload if domain.size(cl) <= args.max_cells]
    if args.num_marginals is not None:
        workload = [workload[i] for i in prng.choice(len(workload), args.num_marginals, replace=False)]

    synth = MST(domain, args.epsilon, args.delta)
  
    if args.save is not None:
        synth.df.to_csv(args.save, index=False)
 
    errors = []
    data = Dataset.load(args.dataset, args.domain)
    for proj in workload:
        X = data.project(proj).datavector()
        Y = synth.project(proj).datavector()
        e = 0.5*np.linalg.norm(X/X.sum() - Y/Y.sum(), 1)
        errors.append(e)
    print('Average Error: ', np.mean(errors))


    #ML_utility_eval('income>50K',data,synth.df)




