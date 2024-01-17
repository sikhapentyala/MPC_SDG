import numpy as np
import pandas as pd
from scipy.special import softmax


class MPCComputations_HP:
    def __init__(self,domain,est_ans,alice_ans,bob_ans):
        self.domain = domain
        self.est_ans = est_ans
        self.alice_ans = alice_ans
        self.bob_ans = bob_ans


    def compute_workload_answer(self):
        '''
        This is for vertical partitioning"
        '''
        pass

    def select_marginal_worst_approximated(self,candidate_workloads,eps, bounded = False):
        errors = np.array([])
        for marginal_index in candidate_workloads:
            #reduce number of additions by taking only domain size
            size = self.domain[marginal_index]
            x = self.alice_ans[marginal_index][:size] + self.bob_ans[marginal_index][:size]
            xest = self.est_ans[marginal_index][:size]
            errors = np.append(errors, np.abs(x - xest).sum()-size)
        sensitivity = 2.0 if bounded else 1.0
        prob = softmax(0.5*eps/sensitivity*(errors - errors.max()))
        key = np.random.choice(len(errors), p=prob)
        return key

    def get_noisy_measurement(self,marginal_index,scale):
        size = self.domain[marginal_index]
        x = self.alice_ans[marginal_index][:size] + self.bob_ans[marginal_index][:size]
        y = x + np.random.normal(loc=0, scale=scale, size=size)
        return y

    def get_noisy_results(self,candidates,exp_eps,measure_scale):
        ax =  self.select_marginal_worst_approximated(candidates,exp_eps)
        y =  self.get_noisy_measurement(ax,measure_scale)
        return ax,y



class MPCComputations_VP:
    def __init__(self,domain,est_ans,alice_ans,bob_ans,workload_ids_to_be_computed_private,_column_ids,domain_bins):
        self.domain = domain
        self.est_ans = est_ans
        self.alice_ans = alice_ans
        self.bob_ans = bob_ans
        self.workload_ids_to_be_computed_private = workload_ids_to_be_computed_private
        self._column_ids = _column_ids
        self.domain_bins = domain_bins


    def compute_workload_answer(self, alice_data, bob_data,flatten=True):
        '''
        This is for vertical partitioning"
        '''
        x = self.alice_ans + self.bob_ans
        #data = alice_data.append(bob_data)
        #data_transposed = list(map(list, zip(*data)))
        data =  pd.concat([alice_data, bob_data], axis=1)
        for index in self.workload_ids_to_be_computed_private:
            columns = self._column_ids[index]
            #a = alice_data.iloc[:,columns[0]].values#data_transposed[columns[0]]
            #b = bob_data.iloc[:,N_cols+columns[1]].values
            ab = data.iloc[:,columns].values
            bins = self.domain_bins[index]
            # TODO: This part will be different in MPC and is costly, it is data cube computations for which we have MPC protocol in slides.
            bins = [range(n+1) for n in bins]
            ans = np.histogramdd(ab, bins, weights=None)[0]
            data_vector = ans.flatten() if flatten else ans
            padded_data_vector = np.pad(data_vector, (0, max(self.domain) - len(data_vector)), 'constant')
            x[index]= padded_data_vector
        return x

    def compute_answer(self, marginal_index, alice_data, bob_data,scale, size, flatten=True):
        columns = self._column_ids[marginal_index]
        data =  pd.concat([alice_data, bob_data], axis=1)
        ab = data.iloc[:,columns].values

        bins = self.domain_bins[marginal_index]
        # TODO: This part will be different in MPC and is costly
        bins = [range(n+1) for n in bins]
        ans = np.histogramdd(ab, bins, weights=None)[0]
        data_vector = ans.flatten() if flatten else ans
        data_vector = data_vector + np.random.normal(loc=0, scale=scale, size=size)
        padded_data_vector = np.pad(data_vector, (0, max(self.domain) - len(data_vector)), 'constant')
        return padded_data_vector

    def select_marginal_worst_approximated(self,candidate_workloads,eps,alice_data, bob_data, bounded = False):
        errors = np.array([])
        x = self.compute_workload_answer(alice_data, bob_data)
        for marginal_index in candidate_workloads:
            #reduce number of additions by taking only domain size
            size = self.domain[marginal_index]
            xest = self.est_ans[marginal_index][:size]
            errors = np.append(errors, np.abs(x[marginal_index][:size] - xest).sum()-size)
        sensitivity = 2.0 if bounded else 1.0
        prob = softmax(0.5*eps/sensitivity*(errors - errors.max()))
        key = np.random.choice(len(errors), p=prob)
        return key

    def get_noisy_measurement(self,marginal_index,scale,alice_data, bob_data):
        size = self.domain[marginal_index]
        x = self.compute_answer(marginal_index,alice_data, bob_data, scale, size)
        #TODO: efficient noise generation for large vector, we have comparison on slides.
        #TODO: Should we add noise to non-zero parts too
        #noise = np.random.normal(loc=0, scale=scale, size=size)
        #y = x + np.random.normal(loc=0, scale=scale, size=size)
        return x

