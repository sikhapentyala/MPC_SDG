import os
import time

import numpy as np
import itertools
from mbi import Dataset, GraphicalModel, FactoredInference
from scipy.special import softmax
from scipy import sparse
from cdp2adp import cdp_rho
import argparse

import json
from mbi import Domain
# Sikha start
from DataHolders import VDataHolder
from MPC import MPCComputations_VP
# Sikha end


"""
This file contains an implementation of MWEM+PGM that is designed specifically for marginal query workloads.
Unlike mwem.py, which selects a single query in each round, this implementation selects an entire marginal 
in each step.  It leverages parallel composition to answer many more queries using the same privacy budget.

This enhancement of MWEM was described in the original paper in section 3.3 (https://arxiv.org/pdf/1012.4763.pdf).

There are two additional improvements not described in the original Private-PGM paper:
- In each round we only consider candidate cliques to select if they result in sufficiently small model sizes
- At the end of the mechanism, we generate synthetic data (rather than query answers)
"""

PATH_MPC = "/home/mpcuser/MP-SPDZ/"
PROTOCOL = "ring"


def mwem_pgm(domain, epsilon, delta=0.0, workload=None, rounds=None, maxsize_mb = 25, pgm_iters=1000, noise='gaussian', bounded=False, alpha=0.9):
    """
    Implementation of MWEM + PGM

    :param data: an mbi.Dataset object
    :param epsilon: privacy budget
    :param delta: privacy parameter (ignored)
    :param workload: A list of cliques (attribute tuples) to include in the workload (default: all pairs of attributes)
    :param rounds: The number of rounds of MWEM to run (default: number of attributes)
    :param maxsize_mb: [New] a limit on the size of the model (in megabytes), used to filter out candidate cliques from selection.
        Used to avoid MWEM+PGM failure modes (intractable model sizes).
        Set to np.inf if you would like to run MWEM as originally described without this modification
        (Note it may exceed resource limits if run for too many rounds)

    Implementation Notes:
    - During each round of MWEM, one clique will be selected for measurement, but only if measuring the clique does
        not increase size of the graphical model too much
    """
    if workload is None:
        workload = list(itertools.combinations(domain, 2))
    if rounds is None:
        rounds = len(domain)


    if noise == 'laplace':
        eps_per_round = epsilon / rounds
        sigma = 1.0 / (alpha*eps_per_round)
        exp_eps = (1-alpha)*eps_per_round
        marginal_sensitivity = 2 if bounded else 1.0
    else:
        rho = cdp_rho(epsilon, delta)
        rho_per_round = rho / rounds
        sigma = np.sqrt(0.5 / (alpha*rho_per_round))
        exp_eps = np.sqrt(8*(1-alpha)*rho_per_round)
        marginal_sensitivity = np.sqrt(2) if bounded else 1.0



    def size(cliques):
        return GraphicalModel(domain, cliques).size * 8 / 2**20

    '''
    # SIKHA
    # Computing answers to data. Needs MPC
    # np.histogramdd(self.df.values, bins, weights=self.weights)[0] bins = [range(n+1) for n in self.domain.shape] flattened
    '''
    num_of_workloads = len(workload)
    workload_domain_size = [domain.project(cl).size() for cl in workload]
    max_domain_size = max(workload_domain_size)
    alice.load_data(workload)
    alice_workload_ids = alice.workload_ids
    bob.load_data(workload)
    bob_workload_ids = bob.workload_ids
    alice.compute_answers(workload,domain,max_domain_size)
    bob.compute_answers(workload,domain,max_domain_size)
    total = alice.total_samples if bounded else None # should this be made private or not?

    non_private_workload_ids = []
    non_private_workload_ids.extend(alice_workload_ids)
    non_private_workload_ids.extend(bob_workload_ids)
    workload_ids_to_be_computed_private = set([i for i in range(num_of_workloads)]) - set(non_private_workload_ids)
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
    #_column_ids.extend(alice.column_ids_for_workload_ids)
    #_column_ids.extend(bob.column_ids_for_workload_ids)
    N_cols = alice.total_columns


    with open(PATH_MPC+'Player-Data/Input-P1-0', 'w') as outfile:
        outfile.write('\n'.join([' '.join([str(num) for num in row]) for row in bob.data.values]))
        outfile.write('\n')
        outfile.write('\n'.join([' '.join([str(num) for num in row]) for row in bob.workload_answers]))
    start_time_cmp = time.time()
    compile_cmd = "cd "+ PATH_MPC +" && ./compile.py -R 64 mwem_V " + str(max_domain_size) + " " + str(num_of_workloads) + " " + str(alice.data.shape[0]) + " " + str(alice.data.shape[1]) + " " + str(bob.data.shape[1])
    print(compile_cmd)
    os.system(compile_cmd)
    print("Compile time:", time.time() - start_time_cmp)

    print(workload_ids_to_be_computed_private)
    print(_column_ids)
    print(domain_bins)

    
    engine = FactoredInference(domain, log=False, iters=pgm_iters, warm_start=True)
    measurements = []
    est = engine.estimate(measurements, total)
    cliques = []
    #rounds = 1
    for i in range(1, rounds+1):
        # [New] Only consider candidates that keep the model sufficiently small
        candidates = [cl for cl in workload if size(cliques+[cl]) <= maxsize_mb*i/rounds]
        candidates_indices = [i for i, value in enumerate(workload) if value in candidates]

        est_ans = []
        for cl in workload:
            data_vector = est.project(cl).datavector()
            padded_data_vector = np.pad(data_vector, (0, max_domain_size - len(data_vector)), 'constant')
            est_ans.append(padded_data_vector)

        with open(PATH_MPC+'Player-Data/Input-P0-0', 'w') as outfile:
            outfile.write('\n'.join([' '.join([str(num) for num in row]) for row in alice.data.values]))
            outfile.write('\n')
            outfile.write('\n'.join([' '.join([str(num) for num in row]) for row in alice.workload_answers]))
            outfile.write('\n')
            outfile.write('\n'.join([' '.join([str(num) for num in row]) for row in est_ans]))
        
        scale = marginal_sensitivity*sigma

        with open(PATH_MPC+"Programs/Public-Input/mwem_V-" + str(max_domain_size) + "-" + str(num_of_workloads) + "-" + str(alice.data.shape[0]) + "-" + str(alice.data.shape[1]) + "-" + str(bob.data.shape[1]), 'w') as outfile:
            outfile.write(str(round(exp_eps * pow(2,16))))
            outfile.write("\n")
            outfile.write(str(round(scale * pow(2,16))))
            outfile.write("\n")
            outfile.write(' '.join(str(num) for num in candidates_indices))
            outfile.write("\n")
            outfile.write(' '.join(str(num) for num in workload_domain_size))
         
        run_cmd = "cd "+ PATH_MPC +" && Scripts/" + str(PROTOCOL) + ".sh mwem_V-" + str(max_domain_size) + "-" + str(num_of_workloads) + "-" + str(alice.data.shape[0]) + "-" + str(alice.data.shape[1]) + "-" + str(bob.data.shape[1]) + " -v > "+ PATH_MPC +"mpc_out.txt"
        os.system(run_cmd)

        # if there was option to store then
        ''''''
        #ax_index = 0
        #with open(PATH_MPC +"mpc_out.txt") as f:
        #    lines = f.readlines()
        #    for line in lines:
        #        if line.startswith("Ax"):
        #            ax_index = int(line.split(":")[1])
        #if ax_index in alice_workload_ids:
        #    y = alice.get_noisy_measurement(ax, ax_index,scale,domain)
        #elif ax_index in bob_workload_ids:
        #    y = bob.get_noisy_measurement(ax, ax_index, scale,domain)
        #else:
        #     run_cmd = "cd "+ PATH_MPC +" && Scripts/" + str(PROTOCOL) + ".sh mwem_msr_V-" + str(max_domain_size) + "-" + str(num_of_workloads) + " -v > "+ PATH_MPC +"mpc_out.txt"
        #     os.system(run_cmd)   
        #     with open(PATH_MPC +"mpc_out.txt") as f:
        #        lines = f.readlines()
        #        for line in lines:
        #            if line.startswith("Y"):
        #                y = line.split(":")[1].strip()[1:-1].split(', ')          
        ''''''

        ax_index = 0
        with open(PATH_MPC +"mpc_out.txt") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Ax"):
                    ax_index = int(line.split(":")[1])
                if line.startswith("Y"):
                    y = line.split(":")[1].strip()[1:-1].split(', ')

        ax = candidates[ax_index]
        print('Round', i, 'Selected', ax , 'Model Size (MB)', est.size*8/2**20)
        n = domain.size(ax)
        y = np.array([float(val) for val in y])[:n]
        # Sikha end
        Q = sparse.eye(n)
        measurements.append((Q, y, 1.0, ax))
        est = engine.estimate(measurements, total)
        cliques.append(ax)
        
    print('Generating Data...')
    return est.synthetic_data()
    

def default_params():
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params['dataset'] = '../data/COMPAS_train.csv'
    params['domain'] = '../data/compass-domain.json'
    params['epsilon'] = 1.0
    params['delta'] = 1e-9
    params['rounds'] = None
    params['noise'] = 'gaussian'
    params['max_model_size'] = 25
    params['pgm_iters'] = 1000
    params['degree'] = 2
    params['num_marginals'] = None
    params['max_cells'] = 10000

    return params

if __name__ == "__main__":

    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('--dataset', help='dataset to use')
    parser.add_argument('--domain', help='domain to use')
    parser.add_argument('--epsilon', type=float, help='privacy parameter')
    parser.add_argument('--delta', type=float, help='privacy parameter')
    parser.add_argument('--rounds', type=int, help='number of rounds of MWEM to run')
    parser.add_argument('--noise', choices=['laplace','gaussian'], help='noise distribution to use')
    parser.add_argument('--max_model_size', type=float, help='maximum size (in megabytes) of model')

    parser.add_argument('--degree', type=int, help='degree of marginals in workload')
    parser.add_argument('--num_marginals', type=int, help='number of marginals in workload')
    parser.add_argument('--max_cells', type=int, help='maximum number of cells for marginals in workload')

    parser.add_argument('--pgm_iters', type=int, help='number of iterations')
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

    alice = VDataHolder("Alice", '../data/COMPAS_train_alice_v.csv',"vertical",n=0.5)
    bob = VDataHolder("Bob", '../data/COMPAS_train_bob_v.csv',"vertical",n=0.5)
    #args.save = '../syn_data/breast_mwem_V-'+str(args.epsilon)+'_0.csv'
    start_time = time.time()
    start_time1 = time.perf_counter()    
    synth = mwem_pgm(domain, args.epsilon, args.delta,
                     workload=workload,
                     rounds=args.rounds,
                     maxsize_mb=args.max_model_size,
                     pgm_iters=args.pgm_iters)
    end_time = time.time()
    end_time1 = time.perf_counter()
    print("Generated in sec:", end_time-start_time, end_time1-start_time1)

    if args.save is not None:
        synth.df.to_csv(args.save, index=False)

    # For evaluation so okay to have data access - but recent paper suggested this leaks information in practice.
    # The paper that performed attack based on released metrics
    errors = []
    data = Dataset.load(args.dataset, args.domain)
    for proj in workload:
        X = data.project(proj).datavector()
        Y = synth.project(proj).datavector()
        e = 0.5*np.linalg.norm(X/X.sum() - Y/Y.sum(), 1)
        errors.append(e)
    print('Average Error: ', np.mean(errors))

