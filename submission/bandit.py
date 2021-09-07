import numpy as np
import argparse

class Bandit:
    # This class represents a bandit instance.

    class Arm:
        # This class represents a bandit arm.
        def __init__(self, pdf, support = [0, 1]):
            # prob: the probability distribution of different outcomes
            # support: the support vector of the bandit
            # support and prob should have the same length
            self.pdf = pdf                    # the probability distribution of different outcomes
            self.support = support            # the support vector of the arm    
            self.cdf = []                     # the cumulative distribution function
            last = 0
            for p in pdf:
                last += p
                self.cdf.append(last)
            
            self.expectation = 0               # the expected value of the arm
            for p, s in zip(self.pdf, self.support):
                self.expectation += p * s

        def pull(self):
            sample =  np.random.uniform()
            for i in range(len(self.cdf)):
                if sample < self.cdf[i]:
                    return self.support[i]

    def __init__(self):
        self.arms = []
        self.n = 0                              # number of arms
        self.t = 0                              # number of pulls
        self.pulls = np.empty(0, dtype=int)     # number of pulls for each arm
        self.emp = np.empty(0, dtype=float)     # empirical mean for each arm
    
    # Add a new arm to the bandit.
    def addArm(self, pdf, support = [0, 1]):
        # pdf: the probability distribution of different outcomes
        # support: the support vector of the arm
        self.arms.append(self.Arm(pdf, support))
        self.n += 1
        self.pulls = np.append(self.pulls, 0)
        self.emp = np.append(self.emp, 0.5)
    
    # Pull i-th arm
    def pull(self, i):
        r = self.arms[i].pull()
        self.t += 1
        self.pulls[i] += 1
        self.emp[i] += (r - self.emp[i]) / self.pulls[i]
        return r
    
    # Get the maximum expected reward of the bandit
    def maxExpectation(self):
        return max([a.expectation for a in self.arms])

################## Read instance file and return bandit object ##################
def constructBandit(instance, mode = 0):
    sp = [0, 1]
    f = open(instance, 'r')
    if not mode == 0:
        sp = list(map(float, f.readline().split()))
    
    bandit = Bandit()

    for line in f:
        pdf = list(map(float, line.split()))
        if len(pdf) == 0:
            break
        if mode == 0:
            pdf = [1-pdf[0], pdf[0]]
        bandit.addArm(pdf, sp)
    
    f.close()
    return bandit

################## epsilon-greedy algorithm ##################
def epsGreedy(ins, e, hor):
    bandit = constructBandit(ins)
    n = bandit.n
    REW = 0
    
    for _ in range(hor):
        e_ = np.random.uniform(0, 1)
        if e_ < e:
            i = np.random.choice(n)
        else:
            i = np.argmax(bandit.emp)
        r = bandit.pull(i)
        REW += r
    
    REG = hor * bandit.maxExpectation() - REW
    return REG

################## UCB algorithm ##################
def ucb(ins, hor, c = 2, mode = 0):
    bandit = constructBandit(ins, mode)
    n = bandit.n
    REW = 0

    for i in range(n):
        r = bandit.pull(i)
        REW += r
        if bandit.t == hor:
            break
    
    while bandit.t < hor:
        i = np.argmax(bandit.emp + np.sqrt(c * np.log(bandit.t) / bandit.pulls))
        r = bandit.pull(i)
        REW += r
    
    REG = hor * bandit.maxExpectation() - REW
    return REG

################## KL-UCB algorithm ##################
def calc_klucb(p, u, t, c = 3):
    if u == 0:
        return 1
        
    def KL(p, q):
        if p == 0:
            return -np.log(1-q)
        if p == 1:
            return -np.log(q)
        return p * (np.log(p) - np.log(q)) + (1 - p) * (np.log(1 - p) - np.log(1 - q))

    lo = p
    hi = 1
    mid = (lo + hi) / 2
    while hi - lo > 1e-6:
        if u * KL(p, mid) < np.log(t) + c * np.log(np.log(t)):
            lo = mid
        else:
            hi = mid
        mid = (lo + hi) / 2
    return mid
    
def klucb(ins, hor, mode = 0):
    bandit = constructBandit(ins, mode)
    n = bandit.n
    REW = 0

    for _ in range(3):
        i = np.random.choice(n)
        r = bandit.pull(i)
        REW += r
        if bandit.t == hor:
            break
    
    while bandit.t < hor:
        i = np.argmax([calc_klucb(bandit.emp[i], bandit.pulls[i], bandit.t) for i in range(n)])
        r = bandit.pull(i)
        REW += r
    
    REG = hor * bandit.maxExpectation() - REW
    return REG

################## Thompson sampling algorithm ##################
def thompson(ins, hor, th = 0.5, mode = 0):
    bandit = constructBandit(ins, mode)
    n = bandit.n
    REW = 0

    succ = [0] * n
    
    while bandit.t < hor:
        i = np.argmax([np.random.beta(succ[i] + 1, bandit.pulls[i] - succ[i] + 1) for i in range(n)])
        r = bandit.pull(i)
        r = int(r > th)
        succ[i] += r
        REW += r
    
    REG = hor * bandit.maxExpectation() - REW
    return REG, REW

def main_function(instance, algorithm, randomSeed, epsilon, scale, threshold, horizon):
    np.random.seed(randomSeed)

    REG = 0
    HIGHS = 0

    if algorithm == 'epsilon-greedy-t1':
        REG = epsGreedy(instance, epsilon, horizon)
    elif algorithm == 'ucb-t1':
        REG = ucb(instance, horizon)
    elif algorithm == 'kl-ucb-t1':
        REG = klucb(instance, horizon)
    elif algorithm == 'thompson-sampling-t1':
        REG, _ = thompson(instance, horizon)
    elif algorithm == 'ucb-t2':
        REG = ucb(instance, horizon, scale)
    elif algorithm == 'alg-t3':
        REG = ucb(instance, horizon, 2, mode = 1)
    elif algorithm == 'alg-t4':
        _, HIGHS = thompson(instance, horizon, threshold, mode = 1)
    
    return REG, HIGHS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=str, required=True)
    parser.add_argument('--algorithm', type=str, required=True)
    parser.add_argument('--randomSeed', type=int, required=True)
    parser.add_argument('--epsilon', type=float, required=True)
    parser.add_argument('--scale', type=float, required=True)
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--horizon', type=int, required=True)

    args = parser.parse_args()
    
    REG, HIGHS = main_function(args.instance, args.algorithm, args.randomSeed, args.epsilon, args.scale, args.threshold, args.horizon)
    print(", ".join([args.instance, args.algorithm, str(args.randomSeed), str(args.epsilon), str(args.scale), str(args.threshold), str(args.horizon), '%.3f'%REG, str(HIGHS)]))
