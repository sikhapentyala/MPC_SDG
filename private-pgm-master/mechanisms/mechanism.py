import numpy as np
#from autodp import privacy_calibrator
from functools import partial
from cdp2adp import cdp_rho
from scipy.special import softmax

from math import exp, sqrt
from scipy.special import erf
from scipy.optimize import brentq

from autodp import rdp_acct, rdp_bank

def pareto_efficient(costs):
    eff = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if eff[i]:
            eff[eff] = np.any(costs[eff]<=c, axis=1)  # Keep any point with a lower cost
    return np.nonzero(eff)[0]

def generalized_em_scores(q, ds, t):
    q = -q
    idx = pareto_efficient(np.vstack([q, ds]).T)
    r = q + t*ds
    r = r[:,None] - r[idx][None,:]
    z = ds[:,None] + ds[idx][None,:]
    s = (r/z).max(axis=1)
    return -s

class Mechanism:
    def __init__(self, epsilon, delta, bounded, prng=np.random):
        """
        Base class for a mechanism.  
        :param epsilon: privacy parameter
        :param delta: privacy parameter
        :param bounded: privacy definition (bounded vs unbounded DP) 
        :param prng: pseudo random number generator
        """
        self.epsilon = epsilon
        self.delta = delta
        self.rho = 0 if delta == 0 else cdp_rho(epsilon, delta)
        self.bounded = bounded
        self.prng = prng

    def run(self, dataset, workload):
        pass

    def generalized_exponential_mechanism(self, qualities, sensitivities, epsilon, t=None, base_measure=None):
        if t is None:
            t = 2*np.log(len(qualities) / 0.5) / epsilon
        if isinstance(qualities, dict):
            keys = list(qualities.keys())
            qualities = np.array([qualities[key] for key in keys])
            sensitivities = np.array([sensitivities[key] for key in keys])
            if base_measure is not None:
                base_measure = np.log([base_measure[key] for key in keys])
        else:
            keys = np.arange(qualities.size)
        scores = generalized_em_scores(qualities, sensitivities, t)
        key = self.exponential_mechanism(scores, epsilon, 1.0, base_measure=base_measure)
        return keys[key]

    def permute_and_flip(self, qualities, epsilon, sensitivity=1.0):
        """ Sample a candidate from the permute-and-flip mechanism """
        q = qualities - qualities.max()
        p = np.exp(0.5*epsilon/sensitivity*q)
        for i in np.random.permutation(p.size):
            if np.random.rand() <= p[i]:
                return i

    def exponential_mechanism(self, qualities, epsilon, sensitivity=1.0, base_measure=None):
        if isinstance(qualities, dict):
            #import pandas as pd
            #print(pd.Series(list(qualities.values()), list(qualities.keys())).sort_values().tail())
            keys = list(qualities.keys())
            qualities = np.array([qualities[key] for key in keys])
            if base_measure is not None:
                base_measure = np.log([base_measure[key] for key in keys])
        else:
            qualities = np.array(qualities)
            keys = np.arange(qualities.size)

        """ Sample a candidate from the permute-and-flip mechanism """
        q = qualities - qualities.max()
        if base_measure is None:
            p = softmax(0.5*epsilon/sensitivity*q)
        else:
            p = softmax(0.5*epsilon/sensitivity*q + base_measure)

        return keys[self.prng.choice(p.size, p=p)]

    def gaussian_noise_scale(self, l2_sensitivity, epsilon, delta):
        """ Return the Gaussian noise necessary to attain (epsilon, delta)-DP """
        if self.bounded: l2_sensitivity *= 2.0
        #return l2_sensitivity * privacy_calibrator.ana_gaussian_mech(epsilon, delta)['sigma']
        return l2_sensitivity * ana_gaussian_mech(epsilon, delta)['sigma']

    def laplace_noise_scale(self, l1_sensitivity, epsilon):
        """ Return the Laplace noise necessary to attain epsilon-DP """
        if self.bounded: l1_sensitivity *= 2.0
        return l1_sensitivity / epsilon

    def gaussian_noise(self, sigma, size):
        """ Generate iid Gaussian noise  of a given scale and size """
        return self.prng.normal(0, sigma, size)

    def laplace_noise(self, b, size):
        """ Generate iid Laplace noise  of a given scale and size """
        return self.prng.laplace(0, b, size)

    def best_noise_distribution(self, l1_sensitivity, l2_sensitivity, epsilon, delta):
        """ Adaptively determine if Laplace or Gaussian noise will be better, and
            return a function that samples from the appropriate distribution """
        b = self.laplace_noise_scale(l1_sensitivity, epsilon)
        sigma = self.gaussian_noise_scale(l2_sensitivity, epsilon, delta)
        dist = self.gaussian_noise if np.sqrt(2)*b > sigma else self.laplace_noise
        if np.sqrt(2)*b < sigma:
            return partial(self.laplace_noise, b)
        return partial(self.gaussian_noise, sigma)


# Subsampling lemma and its inverse

def subsample_epsdelta(eps,delta,prob):
    """
    :param eps: privacy loss eps of the base mechanism
    :param delta: privacy loss delta of the base mechanism
    :param prob: subsampling probability
    :return: Amplified eps and delta

    This result applies to both subsampling with replacement and Poisson subsampling.

    The result for Poisson subsmapling is due to Theorem 1 of :
    Li, Ninghui, Qardaji, Wahbeh, and Su, Dong. On sampling, anonymization, and differential privacy or,
    k-anonymization meets differential privacy

    The result for Subsampling with replacement is due to:
    Jon Ullman's lecture notes: http://www.ccs.neu.edu/home/jullman/PrivacyS17/HW1sol.pdf
    See the proof of (b)

    """
    if prob == 0:
        return 0,0
    return np.log(1+prob*(np.exp(eps)-1)), prob*delta


def subsample_epsdelta_inverse(eps,delta,prob):
    # Give a target subsampled epsilon and delta, and subsampling probability. Calibrate the base eps, delta
    assert(prob > 0 and prob <=1)
    return np.log((np.exp(eps)-1)/prob + 1), np.minimum(delta/prob,1.0)


def subsample_epsdelta_get_prob(eps_target,delta_target, eps_base,delta_base):
    """
    Calibrate the probability of subsampling
    :param eps_target: Target eps in subsampled mechanisms
    :param delta_target: Target delta in subsampled mechanisms
    :param eps_base: base eps, in subsampled mechanisms
    :param delta_base: base delta in subsampled mechanisms.
    :return: subsampling probability  prob
    """
    return np.mininum(1.0,np.minimum(delta_target/delta_base,  (np.exp(eps_target)-1)/(np.exp(eps_base)-1)))



# we start with a general calibration function.

def RDP_mech(rdp_func, eps, delta, param_name, params, bounds=[0,np.inf],k=1,prob=1.0):
    # Take an analytical RDP, find the smallest noise level to achieve (eps, delta)-DP.
    """
    :param rdp_func: the RDP function that takes in params and alpha like those in 'rdp_bank'.
    :param eps:  the required eps
    :param delta:  the required delta
    :param param_name
    :param params: a template dictionary to modify from.
    :param bounds: a pair of numbers indicating the valid ranges of the parameters
    :return: params_out:  the calibrated params.
    """
    assert (eps > 0 and delta > 0)


    def func(x):
        # We assume that the rdp_func and param_name is chosen such that this function is either monotonically
        # increasing or decreasing.
        params[param_name] = x
        rdp = lambda alpha: rdp_func(params, alpha)
        tmp_acct = rdp_acct.anaRDPacct()

        if prob < 1.0 and prob >0:
            tmp_acct.compose_subsampled_mechanism(rdp, prob,coeff=k)
        else:
            tmp_acct.compose_mechanism(rdp,coeff=k)

        eps_tmp = tmp_acct.get_eps(delta)
        return eps_tmp - eps

    # Project to feasible region
    a=np.minimum(bounds[1], np.maximum(1.0, bounds[0]))
    b=np.maximum(bounds[0], np.minimum(2.0, bounds[1]))
    maxiter = 100
    count = 1
    # find a valid range
    while np.sign(func(a)) == np.sign(func(b)):
        a = np.maximum(a/2,bounds[0])
        b = np.minimum(b*2,bounds[1])
        count = count + 1
        if count >=maxiter:
            # infeasible
            raise ValueError('Infeasible privacy parameters for given RDP function and parameter bounds.')

    root = brentq(func, a, b)

    # assign calibarated results
    params_out = params
    params_out[param_name] = root

    return params_out

def subsampled_RDP_mech_get_prob(rdp_func, eps, delta, params,k=1):
    """
    This function calibrates the probability of subsampling to achieve a prescribed privacy goal.
    :param rdp_func: rdp of the base mechanism (as in those in rdp_bank)
    :param eps:  the required eps
    :param delta:  the required delta
    :param params: the parameter object for the first argument, rdp_func
    :param k: (optional) number of rounds to compose.
    :return:
    """
    assert (eps > 0 and delta > 0)
    def func(x):
        rdp = lambda alpha: rdp_func(params, alpha)
        tmp_acct = rdp_acct.anaRDPacct()
        tmp_acct.compose_subsampled_mechanism(rdp, x, coeff=k)
        eps_tmp = tmp_acct.get_eps(delta)
        return eps_tmp - eps

    root = brentq(func, 0, 1)
    return root


def gaussian_mech(eps,delta,k=1,prob=1.0):
    """
    Calibrate the scale parameter b of the Gaussian mechanism using RDP

    :param eps: prescribed eps
    :param delta: prescribed delta
    :param k: (optional) number of times to run this mechanism.
    :return: the parameter structure for this randomized algorithm
    """
    assert(eps>0)

    params = {}
    param_name = 'sigma'
    params = RDP_mech(rdp_bank.RDP_gaussian, eps, delta, param_name, params,k=k,prob=prob)

    return params



def laplace_mech(eps, delta, k=1, prob=1.0):
    """
    Calibrate the scale parameter b of the Laplace mechanism

    :param eps: prescribed eps
    :param delta: prescribed delta
    :param k: (optional) number of times to run this mechanism.
    :return: the parameter structure for this randomized algorithm
    """
    assert(eps>0 and delta >= 0)

    binf = 1.0/eps*k
    if delta == 0:
        params = {'b':binf}
    else:
        params = {}
        param_name = 'b'
        params = RDP_mech(rdp_bank.RDP_laplace, eps, delta, param_name, params, k=k,prob=prob)
        if params['b'] > binf:
            # further take maximum w.r.t. alpha = infty
            params['b'] = binf

    return params


def randresponse_mech(eps, delta, k=1,prob=1.0):
    """
    Calibrate the bernoulli parameter p of the randomized response mechanisms

    :param eps: prescribed eps
    :param delta: prescribed delta
    :param k: (optional) number of times to run this mechanism.
    :return: the parameter structure for this randomized algorithm
    """
    assert (eps > 0 and delta >= 0)

    pinf= np.exp(1.0*eps/k)/(1+np.exp(1.0*eps/k))
    if delta ==1:
        return {'p':1}
    if delta == 0:
        params = {'p':pinf}
    else:
        params = {}
        param_name = 'p'
        params = RDP_mech(rdp_bank.RDP_randresponse, eps, delta, param_name, params,
                          bounds=[np.exp(1.0*eps/k/2)/(1+np.exp(1.0*eps/2/k)),1-1e-8],k=k,prob=prob)
        if params['p'] < pinf:
            # further take maximum w.r.t. alpha = infty
            params['p'] = pinf

    return params


def classical_gaussian_mech(eps,delta):
    """
    The classical gaussian mechanism. For benchmarking purposes only.
    DO NOT USE in practice as it is dominated by `ana_gaussian_mech' and `gaussian_mech`.

    :param eps: prescribed 0< eps <1
    :param delta: prescribed 0 < delta <1
    :return: required noise level.
    """
    assert(eps > 0 and eps <=1), \
        "The classical Gaussian mechanism only supports 0 < eps <1, try `gaussian_mech` and `ana_gaussian_mech`"
    if delta <= 0:
        return np.inf
    if delta >= 1:
        return 0
    params = {'sigma': 1.0 * np.sqrt(2 * np.log(1.25 / delta)) / eps}
    return params


def ana_gaussian_mech(epsilon, delta, tol=1.e-12):
    """ Calibrate a Gaussian perturbation for differential privacy using the analytic Gaussian mechanism of [Balle and Wang, ICML'18]

    Modified from https://github.com/BorjaBalle/analytic-gaussian-mechanism/blob/master/agm-example.py

    Arguments:
    epsilon : target epsilon (epsilon > 0)
    delta : target delta (0 < delta < 1)
    tol : error tolerance for binary search (tol > 0)
    Output:
    params : a dictionary that contains field `sigma' --- the standard deviation of Gaussian noise needed to achieve
        (epsilon,delta)-DP under global sensitivity 1
    """

    def Phi(t):
        return 0.5*(1.0 + erf(float(t)/sqrt(2.0)))

    def caseA(epsilon,s):
        return Phi(sqrt(epsilon*s)) - exp(epsilon)*Phi(-sqrt(epsilon*(s+2.0)))

    def caseB(epsilon,s):
        return Phi(-sqrt(epsilon*s)) - exp(epsilon)*Phi(-sqrt(epsilon*(s+2.0)))

    def doubling_trick(predicate_stop, s_inf, s_sup):
        while(not predicate_stop(s_sup)):
            s_inf = s_sup
            s_sup = 2.0*s_inf
        return s_inf, s_sup

    def binary_search(predicate_stop, predicate_left, s_inf, s_sup):
        s_mid = s_inf + (s_sup-s_inf)/2.0
        while(not predicate_stop(s_mid)):
            if (predicate_left(s_mid)):
                s_sup = s_mid
            else:
                s_inf = s_mid
            s_mid = s_inf + (s_sup-s_inf)/2.0
        return s_mid

    delta_thr = caseA(epsilon, 0.0)

    if (delta == delta_thr):
        alpha = 1.0
        sigma = alpha / sqrt(2.0 * epsilon)

    else:
        if (delta > delta_thr):
            predicate_stop_DT = lambda s : caseA(epsilon, s) >= delta
            function_s_to_delta = lambda s : caseA(epsilon, s)
            predicate_left_BS = lambda s : function_s_to_delta(s) > delta
            function_s_to_alpha = lambda s : sqrt(1.0 + s/2.0) - sqrt(s/2.0)

        else:
            predicate_stop_DT = lambda s : caseB(epsilon, s) <= delta
            function_s_to_delta = lambda s : caseB(epsilon, s)
            predicate_left_BS = lambda s : function_s_to_delta(s) < delta
            function_s_to_alpha = lambda s : sqrt(1.0 + s/2.0) + sqrt(s/2.0)

        predicate_stop_BS = lambda s : abs(function_s_to_delta(s) - delta) <= tol

        s_inf, s_sup = doubling_trick(predicate_stop_DT, 0.0, 1.0)
        s_final = binary_search(predicate_stop_BS, predicate_left_BS, s_inf, s_sup)
        alpha = function_s_to_alpha(s_final)
        sigma = alpha/sqrt(2.0*epsilon)


    params = {'sigma': sigma}
    return params


