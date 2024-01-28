import json
import os

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
from DataHolders import HDataHolder
from MPC import MPCComputations_HP
# Sikha end

import time


PATH_MPC = "/home/mpcuser/MP-SPDZ/"
PROTOCOL = "ring"
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

    '''MPC start'''
    #For local computations'''
    domain_size = [domain.project(cl).size() for cl in cliques]
    max_domain_size = max(domain_size)
    alice.load_data()
    bob.load_data()
    total = (alice.total_samples + bob.total_samples)
    alice.compute_answers(cliques,domain,max_domain_size,keep_pad=False)
    bob.compute_answers(cliques,domain,max_domain_size,keep_pad=False)


    #mpc = MPCComputations_HP(workload_domain_size, est_ans, alice.workload_answers, bob.workload_answers)
    compile_cmd = "cd "+ PATH_MPC +" && ./compile.py -R 64 mst_H " + str(max_domain_size) + " " + str(len(cliques))
    os.system(compile_cmd)


    with open(PATH_MPC+'Player-Data/Input-P1-0', 'w') as outfile:
        outfile.write('\n'.join([' '.join([str(num) for num in row]) for row in bob.workload_answers]))

    with open(PATH_MPC+'Player-Data/Input-P1-0', 'w') as outfile:
        outfile.write('\n'.join([' '.join([str(num) for num in row]) for row in alice.workload_answers]))

    #mpc_1 = MPCComputations_HP(domain_size, None, alice.workload_answers, bob.workload_answers)
    # MPC for measure
    log1 = measure(max_domain_size, cliques, domain_size, sigma)
    '''MPC end'''
    #data, log1, undo_compress_fn = compress_domain(data, log1)

    # MPC for select
    workloads = list(itertools.combinations(domain.attrs, 2))
    domain_size = [domain.project(cl).size() for cl in workloads]
    max_domain_size = max(domain_size)
    total = (alice.total_samples + bob.total_samples)
    alice.compute_answers(workloads,domain,max_domain_size,keep_pad=False)
    bob.compute_answers(workloads,domain,max_domain_size,keep_pad=False)




    mpc_2 = MPCComputations_HP(domain_size, None, alice.workload_answers, bob.workload_answers)
    cliques = select(mpc_2, workloads,domain, rho/3.0, log1)


    # MPC for measure
    domain_size = [domain.project(cl).size() for cl in cliques]
    max_domain_size = max(domain_size)
    total = (alice.total_samples + bob.total_samples)
    alice.compute_answers(cliques,domain,max_domain_size,keep_pad=False)
    bob.compute_answers(cliques,domain,max_domain_size,keep_pad=False)

    compile_cmd = "cd "+ PATH_MPC +" && ./compile.py -R 64 mst_H " + str(max_domain_size) + " " + str(len(cliques))
    os.system(compile_cmd)


    with open(PATH_MPC+'Player-Data/Input-P1-0', 'w') as outfile:
        outfile.write('\n'.join([' '.join([str(num) for num in row]) for row in bob.workload_answers]))

    with open(PATH_MPC+'Player-Data/Input-P1-0', 'w') as outfile:
        outfile.write('\n'.join([' '.join([str(num) for num in row]) for row in alice.workload_answers]))


    #log2 = measure(data, cliques, sigma)
    log2 = measure(max_domain_size, cliques,domain_size, sigma)
    engine = FactoredInference(domain, iters=5000)
    est = engine.estimate(log1+log2)
    synth = est.synthetic_data()
    return synth
    #return undo_compress_fn(synth)

def measure(max_domain_size, cliques,domain_size, sigma, weights=None):
    if weights is None:
        weights = np.ones(len(cliques))
    weights = np.array(weights) / np.linalg.norm(weights)
    measurements = []
    clique_indices = {value: i for i, value in enumerate(cliques)}
    for proj, wgt in zip(cliques, weights):
        '''
        Sikha: this one to be done in MPC
        proj is the name of the feature, x is the marginal
        
        '''
        marginal_index = clique_indices[proj]
        size = domain_size[marginal_index]
        with open(PATH_MPC+'Programs/Public-Input/mst_H-' + str(max_domain_size) + '-' + str(len(cliques)), 'w') as outfile:
            outfile.write(str(round(sigma/wgt * pow(2,16))))
            outfile.write("\n")
            outfile.write(str(marginal_index))
            outfile.write("\n")
            outfile.write(' '.join(str(num) for num in clique_indices.values()))
            outfile.write("\n")
            outfile.write(' '.join(str(num) for num in domain_size))

        run_cmd = "cd "+ PATH_MPC +" && Scripts/" + str(PROTOCOL) + ".sh mst_H-" + str(max_domain_size) + "-" + str(len(cliques)) + " -v > "+ PATH_MPC +"mpc_out.txt"
        os.system(run_cmd)

        with open(PATH_MPC +"mpc_out.txt") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Y"):
                    y = line.split(":")[1].strip()[1:-1].split(', ')

        y = np.array([float(val) for val in y])[:size]
        #x = data.project(proj).datavector()
        #y = x + np.random.normal(loc=0, scale=sigma/wgt, size=x.size)
        '''
        end
        '''
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

def select(mpc, workloads, domain, rho, measurement_log, cliques=[]):
    engine = FactoredInference(domain, iters=1000)
    est = engine.estimate(measurement_log)

    #weights = {}
    #candidates = list(itertools.combinations(data.domain.attrs, 2))

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

    '''
    Sikha: Compute weights in MPC
    '''
#    for a, b in candidates:
#        xhat = est.project([a, b]).datavector()
#        x = data.project([a, b]).datavector()
#        weights[a,b] = np.linalg.norm(x - xhat, 1) # dictionary {(a,b):w}


    '''
    Sikha: Compute weights in MPC
    '''
    for i in range(r-1):
        candidates_loop = [e for e in workloads if not ds.connected(*e)]
        candidates_indices = [i for i, value in enumerate(workload) if value in candidates_loop]
        '''
        Sikha start MPC
        '''
        idx = mpc.select_marginal_worst_approximated(candidates_indices,epsilon,mst=True)
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

    workload = list(itertools.combinations(domain, args.degree))
    workload = [cl for cl in workload if domain.size(cl) <= args.max_cells]
    if args.num_marginals is not None:
        workload = [workload[i] for i in prng.choice(len(workload), args.num_marginals, replace=False)]

    alice = HDataHolder("Alice", args.dataset,"horizontal",n=0.5)
    bob = HDataHolder("Bob", args.dataset,"horizontal",n=0.5)



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
