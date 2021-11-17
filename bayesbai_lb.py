# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(1) # for fixing the results
import matplotlib
try:
    from google.colab import files
except: # https://stackoverflow.com/questions/35737116/runtimeerror-invalid-display-variable
    matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy import stats
from scipy.stats import truncnorm
import copy
import pickle
import sys
import os
import argparse

parser = argparse.ArgumentParser(description='process some integers.') 
parser.add_argument('-m', '--mode', action='store', 
    default="", \
    type=str, \
    choices=None)
parse_result = parser.parse_args()
run_mode = parse_result.mode

# Note that full run is very heavy (> 1 day(s) with a 10-core desktop)
# You may change full with Runnum=10K or default with Runnum=4K to obtain a reasonable result
if run_mode == "full": #large
    # full simulation: Heavy
    Runnum = 50000
    print("full mode (heavy)")
elif run_mode == "debug": # small
    # for testing
    Runnum = 10
    print("debug mode (lightweight)")
else:
    # default mode 
    Runnum = 1000
    print("default mode")
print(f"Runnum = {Runnum}")
sys.stdout.flush()

# parallel computation.
# avoiding randomness with https://joblib.readthedocs.io/en/latest/auto_examples/parallel_random_state.html
def run_sim(sim, random_state):
    rng = np.random.RandomState(random_state)
    #print(f"rng = {rng}")
    sim.run(rng)
    return sim
from joblib import Parallel, delayed

class TwostagePolicy: # TSE
    def __init__(self, q, T, K):
        self.q, self.T, self.K = q, T, K
        self.bestArmCandidates = []
        self.Kprime = -1
    def kl(self, p, q):
      v = 0
      if p > 0:
        v += p * np.log(p/q)
      if p < 1:
        v += (1-p) * np.log((1-p)/(1-q))
      return v
    def dkl(self, p, q):
      return (q-p)/(q*(1.0-q))
    def lb_kl(self, mu, N): # Chernoff LB
      # q < mu such that exp(- N KL(mu, q) ) = 1/T^2
      eps = 0.00001
      if mu <= eps: return mu
      l = eps
      u = mu
      threshold = 2. * np.log(T) / N
      for i in range(20):
        m = (l+u)/2.
        if self.kl(mu, m) > threshold:
          l = m
        else:
          u = m
      return (l+u)/2
    def ub_kl(self, mu, N): # Chernoff LB
      # q > mu such that exp(- N KL(mu, q) ) = 1/T^2
      eps = 0.00001
      if mu >= 1.0-eps: return mu
      l = mu
      u = 1 - eps
      threshold = 2. * np.log(T) / N
      for i in range(20):
        m = (l+u)/2.
        if self.kl(mu, m) > threshold:
          u = m
        else:
          l = m
      return (l+u)/2
    def ConfBound(self):
        q, T, K = self.q, self.T, self.K
        return np.sqrt(np.log(T)/(q*T/K))
    def nextArm(self, t, S_list, N_list):
        q, T, K = self.q, self.T, self.K
        if t < q * T: #uniform exploration
            if t+1 >= q*T:
                #conf = self.ConfBound()
                emp_mu_list, lb_list, ub_list = np.zeros(K), np.zeros(K), np.zeros(K)
                for i in range(K):
                    emp_mu_list[i] = S_list[i] / max(1, N_list[i])
                    lb_list[i] = self.lb_kl(emp_mu_list[i], N_list[i])
                    ub_list[i] = self.ub_kl(emp_mu_list[i], N_list[i])
                #print(f"emp_mu_list = {emp_mu_list}")
                #print(f"lb_list = {lb_list}")
                #print(f"ub_list = {ub_list}")
                #print(f"conf = {conf}")
                lb_max = np.max(lb_list)
                self.bestArmCandidates = []
                for i in range(K):
                    if ub_list[i] > lb_max:
                        self.bestArmCandidates.append(i)
                self.Kprime = len(self.bestArmCandidates)
            return t % K #mod
        else:
            return self.bestArmCandidates[ t % self.Kprime ]


def argmax(alist):
    i_max = 0
    K = len(alist)
    for i in range(K):
        if alist[i_max] < alist[i]:
            i_max = i
    return i_max

def get_Delta_list(mu_list):
    K = len(mu_list)
    i_max = argmax(mu_list)
    Delta_list = np.zeros(K)
    for i in range(K):
        Delta_list[i] = mu_list[i_max] - mu_list[i]
    return Delta_list

def unifTheoreticalBound(K): 
    C = K*K/((K+1)*(K+2))
    return C

class Simulation:
    def __init__(self, K, T, q, weighted_prior = True):
        self.K, self.T, self.q, self.weighted_prior = K, T, q, weighted_prior
    def drawUniformPrior(self, K):
        return self.rng.rand(K), 1
    def drawTopHeavyPrior(self, K, T):
        sqT = np.sqrt(T)
        while True:
            mus = self.rng.rand(K)
            s = sorted(mus)
            Delta = s[-1] - s[-2]
            prob = max(min(1/(2*sqT*Delta), 1), 0.01) # np.exp(-(Delta*Delta*T/2.)) #acceptance ratio
            if self.rng.random() < prob:
                return (mus, 1./prob)
    def run(self, rng):
        K, T, q, weighted_prior = self.K, self.T, self.q, self.weighted_prior
        self.rng = rng
        policy = TwostagePolicy(q, T, K)
        if weighted_prior:
            mu_list, weight = self.drawTopHeavyPrior(K, T)
        else:
            mu_list, weight = self.drawUniformPrior(K)
        S_list, N_list = np.zeros(K), np.zeros(K)
        for t in range(T):
            It = policy.nextArm(t, S_list, N_list) # next arm
            S_list[It] += int(self.rng.rand() < mu_list[It])
            N_list[It] += 1
        emp_mu_list = np.zeros(K)
        for i in range(K):
            emp_mu_list[i] = S_list[i] / max(1, N_list[i])
        est_i_max = argmax(emp_mu_list)
        true_i_max = argmax(mu_list)
        Delta_list = get_Delta_list(mu_list)
        s_mu_list = sorted(mu_list)
        Deltamin = s_mu_list[-1] - s_mu_list[-2]
        Rt = Delta_list[est_i_max] #simple regret
        self.Rt = Rt
        if len(policy.bestArmCandidates) >= 3:
            self.Rt_three = Rt
        else:
            self.Rt_three = 0
        self.weight = weight
        self.Deltamin = Deltamin
    
def main(K, T, q, weighted_prior = True):
    print(f"K, T, q = {K}, {T}, {q}")
    sys.stdout.flush()
    Rt_list = np.zeros(Runnum)
    Rt_three_list = np.zeros(Runnum) #Regret for bestcand >= 3
    Weight_list = np.zeros(Runnum)
    Deltamin_list = np.zeros(Runnum)
    rss = np.random.randint(np.iinfo(np.int32).max, size=Runnum)
    n_cpus = os.cpu_count()
    sims = [Simulation(K, T, q, weighted_prior) for r in range(Runnum)] 
    sims = Parallel(n_jobs=max(int(n_cpus * 0.8),1))( [delayed(run_sim)(sims[r], rss[r]) for r in range(Runnum)] ) #parallel computation
    for run in range(Runnum):
        Rt_list[run] = sims[run].Rt
        Rt_three_list[run] = sims[run].Rt_three
        Weight_list[run] = sims[run].weight
        Deltamin_list[run] = sims[run].Deltamin
    #print(f"Rt_mean = {np.mean(Rt_list)*T}")
    #print(f"weights = {Weight_list}")
    #print(f"Deltamins = {Deltamin_list}")
    #print(f"Rt_mean = {np.average(Rt_list, weights=Weight_list)}")
    Rt_mean = np.average(Rt_list, weights=Weight_list)
    Rt_three_mean = np.average(Rt_three_list, weights=Weight_list)
    # https://www.math.arizona.edu/~tgk/mc/book_chap6.pdf (6.6) calculation of var for rejection sampling
    if not weighted_prior:
        Rt_std = np.std(Rt_list) 
    else: #weighted
        weight_avg = np.mean(Weight_list)
        Rt_list_weighted = np.zeros(Runnum)
        for run in range(Runnum):
            Rt_list_weighted[run] = Rt_list[run]*Weight_list[run]/weight_avg
        Rt_std = np.std(Rt_list_weighted)
    return Rt_mean, Rt_three_mean, Rt_std

do_varK = True
do_varT = True
weighted_prior = True #rejection sampling
print(f"do_varK = {do_varK}, do_varT = {do_varT}, weighted_prior = {weighted_prior}")

if do_varK:
    CRegret_emp_K_list = []
    CRegret_three_emp_K_list = []
    CRegret_emp_K_std_list = []
    CRegret_theory_K_list = []
    K_list = np.array([2, 3, 5, 10])
    for K in K_list:
        if run_mode == "full":
            T, q = 200000*K, 0.5
        else:
            T, q = 10000*K, 0.5
        q_coef = K/(2*(1-q) + q *K)
        Tprime = T / q_coef
        Regmean, Regthreemean, Regstd = main(K, T, q, weighted_prior=weighted_prior)
        CRegret_emp = Regmean * Tprime
        CRegret_theory = unifTheoreticalBound(K)
        CRegret_emp_K_list.append(CRegret_emp)
        CRegret_three_emp_K_list.append(Regthreemean * Tprime)
        CRegret_emp_K_std_list.append(Regstd * Tprime)
        CRegret_theory_K_list.append(CRegret_theory)
    CRegret_emp_K_list = np.array(CRegret_emp_K_list)
    CRegret_three_emp_K_list = np.array(CRegret_three_emp_K_list)
    CRegret_emp_K_std_list = np.array(CRegret_emp_K_std_list)
    CRegret_theory_K_list = np.array(CRegret_theory_K_list)
    # save pickles
    var_K_results = (CRegret_emp_K_list, CRegret_three_emp_K_list, CRegret_emp_K_std_list, CRegret_theory_K_list)
    with open('pic/varK.pickle', 'wb') as f:
      pickle.dump(var_K_results, f)

if do_varT:
    CRegret_emp_T_list = []
    CRegret_three_emp_T_list = []
    CRegret_emp_T_std_list = []
    CRegret_theory_T_list = []
    if run_mode == "full":
        T_list = np.array([3000, 10000, 30000, 100000, 300000, 1000000, 3000000, 10000000])
    elif run_mode == "debug":
        T_list = np.array([300, 1000])
    else:
        T_list = np.array([3000, 10000, 30000, 100000, 300000, 1000000])
    for T in T_list:
        K, q = 5, 0.5
        q_coef = K/(2*(1-q) + q *K)
        Tprime = T / q_coef
        Regmean, Regthreemean, Regstd = main(K, T, q, weighted_prior=weighted_prior)
        CRegret_emp = Regmean * Tprime
        CRegret_theory = unifTheoreticalBound(K)
        CRegret_emp_T_list.append(CRegret_emp)
        CRegret_three_emp_T_list.append(Regthreemean * Tprime)
        CRegret_emp_T_std_list.append(Regstd * Tprime)
        CRegret_theory_T_list.append(CRegret_theory)
    CRegret_emp_T_list = np.array(CRegret_emp_T_list)
    CRegret_three_emp_T_list = np.array(CRegret_three_emp_T_list)
    CRegret_emp_T_std_list = np.array(CRegret_emp_T_std_list)
    CRegret_theory_T_list = np.array(CRegret_theory_T_list)
    # save pickles
    var_T_results = (CRegret_emp_T_list, CRegret_three_emp_T_list, CRegret_emp_T_std_list, CRegret_theory_T_list)
    with open('pic/varT.pickle', 'wb') as f:
      pickle.dump(var_T_results, f)

# Plotting
Figsize = (6,4)
plt.rcParams["font.size"] = 14
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["figure.subplot.bottom"] = 0.14
save_img = True
confidence_bound = True

COLOR_EMP = "tab:blue"
COLOR_THEO = "black"
LINESTYLE_EMP = "solid"
LINESTYLE_THEO = "dashdot"
def my_show():
    try:
        from google.colab import files
        plt.show()
    except:
        pass
def colab_save(filename):
    try:
        from google.colab import files
        files.download(filename)  
    except:
        pass
def plot():
    if do_varK:
        fig = plt.figure(figsize=Figsize)
        plt.plot(K_list, CRegret_emp_K_list, label = "TSE", marker='o', color = COLOR_EMP, linestyle = LINESTYLE_EMP)
        plt.plot(K_list, CRegret_three_emp_K_list, label = "TSE_three", marker='o', color = "green", linestyle = "dotted")
        plt.plot(K_list, CRegret_theory_K_list, label = "Theory", marker='o', color = COLOR_THEO, linestyle = LINESTYLE_THEO)
        if confidence_bound:
            plt.errorbar(K_list, CRegret_emp_K_list, yerr=2*CRegret_emp_K_std_list/np.sqrt(Runnum), fmt='o', capsize = 3, color = COLOR_EMP) #2 sigma
        plt.legend()
        plt.ylabel("C")
        plt.xlabel("K")
        my_show()
        if save_img:
            fig.savefig("varK.pdf", dpi=fig.dpi, bbox_inches='tight')
            colab_save("varK.pdf")
        plt.clf()
    if do_varT:
        fig = plt.figure(figsize=Figsize)
        plt.plot(T_list, CRegret_emp_T_list, label = "TSE", marker='o', color = COLOR_EMP, linestyle = LINESTYLE_EMP)
        plt.plot(T_list, CRegret_three_emp_T_list, label = "TSE_three", marker='o', color = "green", linestyle = "dotted")
        plt.plot(T_list, CRegret_theory_T_list, label = "Theory", marker='o', color = COLOR_THEO, linestyle = LINESTYLE_THEO)
        if confidence_bound:
            plt.errorbar(T_list, CRegret_emp_T_list, yerr=2*CRegret_emp_T_std_list/np.sqrt(Runnum), fmt='o', capsize = 3, color = COLOR_EMP) #2 sigma
        plt.legend()
        plt.ylabel("C")
        plt.xscale("log")
        plt.xlabel("T")
        my_show()
        if save_img:
            fig.savefig("varT.pdf", dpi=fig.dpi, bbox_inches='tight')
            colab_save("varT.pdf")
        plt.clf()

plot()
