import json

import numpy as np
import itertools
from mbi import Dataset, GraphicalModel, FactoredInference, Domain
from mechanism import Mechanism
from collections import defaultdict
#from hdmm.matrix import Identity
from matrix import Identity
from scipy.optimize import bisect
import pandas as pd
from mbi import Factor
import argparse

# Sikha start
from DataHolders import HDataHolder
from MPC import MPCComputations_HP
# Sikha end

def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1,len(s)+1))

def downward_closure(Ws):
    ans = set()
    for proj in Ws:
        ans.update(powerset(proj))
    return list(sorted(ans, key=len))

def hypothetical_model_size(domain, cliques):
    model = GraphicalModel(domain, cliques)
    return model.size * 8 / 2**20


def compile_workload(workload):
    def score(cl):
        return sum(len(set(cl)&set(ax)) for ax in workload)
    return { cl : score(cl) for cl in downward_closure(workload) }

def filter_candidates(candidates, model, size_limit):
    ans = { }
    free_cliques = downward_closure(model.cliques)
    for cl in candidates:
        cond1 = hypothetical_model_size(model.domain, model.cliques + [cl]) <= size_limit
        cond2 = cl in free_cliques
        if cond1 or cond2:
            ans[cl] = candidates[cl]
    return ans

class AIM(Mechanism):
    def __init__(self,epsilon,delta,prng=None,rounds=None,max_model_size=80,structural_zeros={}):
        super(AIM, self).__init__(epsilon, delta, prng)
        self.rounds = rounds
        self.max_model_size = max_model_size
        self.structural_zeros = structural_zeros

    def worst_approximated(self, candidates, answers, model, eps, sigma):
        errors = {}
        sensitivity = {}
        for cl in candidates:
            wgt = candidates[cl]
            x = answers[cl]
            bias = np.sqrt(2/np.pi)*sigma*model.domain.size(cl)
            xest = model.project(cl).datavector()
            errors[cl] = wgt * (np.linalg.norm(x - xest, 1) - bias)
            sensitivity[cl] = abs(wgt)

        max_sensitivity = max(sensitivity.values()) # if all weights are 0, could be a problem
        return self.exponential_mechanism(errors, eps, max_sensitivity)

    def run(self, domain, W):
        rounds = self.rounds or 16*len(domain)
        workload = [cl for cl, _ in W]
        candidates = compile_workload(workload)

        # Requires MPC, Compute answers and keep
        workload_domain_size = [domain.project(cl).size() for cl in candidates]
        max_domain_size = max(workload_domain_size)
        alice.load_data()
        bob.load_data()
        alice.compute_answers(candidates,domain,max_domain_size,keep_pad=True)
        bob.compute_answers(candidates,domain,max_domain_size,keep_pad=True)

        #answers = { cl : data.project(cl).datavector() for cl in candidates }

        oneway = [cl for cl in candidates if len(cl) == 1]
        #oneway_indices = {value: i for i, value in enumerate(candidates) if value}
        oneway_indices = {value:i for i, value in enumerate(candidates) if value in oneway}

        sigma = np.sqrt(rounds / (2*0.9*self.rho))
        epsilon = np.sqrt(8*0.1*self.rho/rounds)

        measurements = []
        print('Initial Sigma', sigma)
        rho_used = len(oneway)*0.5/sigma**2


        # MPC to compute Matrix
        mpc = MPCComputations_HP(workload_domain_size, None, alice.workload_answers, bob.workload_answers)
        #Y = mpc.get_noisy_measurements(oneway_indices,sigma)

        #for y in Y:
        #    I = Identity(y.size)
        #    measurements.append((I, y, sigma, cl))

        for cl in oneway:
            # MPC part
            marginal_index = oneway_indices[cl]
            y = mpc.get_noisy_measurement(marginal_index,sigma)
            #x = data.project(cl).datavector()
            #y = x + self.gaussian_noise(sigma,x.size)
            I = Identity(y.size)
            measurements.append((I, y, sigma, cl))

        zeros = self.structural_zeros
        engine = FactoredInference(domain,iters=1000,warm_start=True,structural_zeros=zeros)
        model = engine.estimate(measurements)

        t = 0
        terminate = False
        while not terminate:
            t += 1
            if self.rho - rho_used < 2*(0.5/sigma**2 + 1.0/8 * epsilon**2):
                # Just use up whatever remaining budget there is for one last round
                remaining = self.rho - rho_used
                sigma = np.sqrt(1 / (2*0.9*remaining))
                epsilon = np.sqrt(8*0.1*remaining)
                terminate = True

            rho_used += 1.0/8 * epsilon**2 + 0.5/sigma**2
            size_limit = self.max_model_size*rho_used/self.rho

            small_candidates = filter_candidates(candidates, model, size_limit)
            small_candidates_indices = {value:i for i, value in enumerate(candidates) if value in small_candidates.keys()}

            # Requires MPC
            bias = np.zeros(len(candidates))
            wgt = np.ones(len(candidates))
            sensitivity =[]
            for i,cl in zip(small_candidates_indices.values(),small_candidates.keys()):
                wt = small_candidates[cl]
                wgt[i] = wt
                bias[i] = np.sqrt(2/np.pi)*sigma*model.domain.size(cl)
                sensitivity.append(abs(wt))
            max_sensitivity = max(sensitivity)


            est_ans = []
            for cl in candidates:
                data_vector = model.project(cl).datavector()
                #padded_data_vector = np.pad(data_vector, (0, max_domain_size - len(data_vector)), 'constant')
                est_ans.append(data_vector)

            mpc.est_ans = est_ans

            ax = mpc.select_marginal_worst_approximated_aim(small_candidates_indices.values(),epsilon,max_sensitivity,bias,wgt)
            cl = next((key for key, value in small_candidates_indices.items() if value == ax), None)

            #cl = self.worst_approximated(small_candidates, answers, model, epsilon, sigma)

            n = domain.size(cl)
            Q = Identity(n)

            #Requires MPC

            marginal_index = small_candidates_indices[cl]
            y = mpc.get_noisy_measurement(marginal_index,sigma)
            #x = data.project(cl).datavector()
            #y = x + self.gaussian_noise(sigma, n)
            measurements.append((Q, y, sigma, cl))

            z = model.project(cl).datavector()

            model = engine.estimate(measurements)
            w = model.project(cl).datavector()
            print('Selected',cl,'Size',n,'Budget Used',rho_used/self.rho)
            if np.linalg.norm(w-z, 1) <= sigma*np.sqrt(2/np.pi)*n:
                print('(!!!!!!!!!!!!!!!!!!!!!!) Reducing sigma', sigma/2)
                sigma /= 2
                epsilon *= 2

        print('Generating Data...')
        engine.iters = 2500
        model = engine.estimate(measurements)
        synth = model.synthetic_data()

        return synth

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
    params['noise'] = 'laplace'
    params['max_model_size'] = 80
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
    parser.add_argument('--max_model_size', type=float, help='maximum size (in megabytes) of model')
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

    workload = [(cl, 1.0) for cl in workload]
    mech = AIM(args.epsilon, args.delta, max_model_size=args.max_model_size)
    synth = mech.run(domain, workload)

    if args.save is not None:
        synth.df.to_csv(args.save, index=False)

    errors = []
    data = Dataset.load(args.dataset, args.domain)
    for proj, wgt in workload:
        X = data.project(proj).datavector()
        Y = synth.project(proj).datavector()
        e = 0.5*wgt*np.linalg.norm(X/X.sum() - Y/Y.sum(), 1)
        errors.append(e)
    print('Average Error: ', np.mean(errors))
