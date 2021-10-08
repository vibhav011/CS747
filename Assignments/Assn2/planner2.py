import numpy as np
import argparse
import pulp as pl

class MDP:
    def __init__(self, states, actions):
        self.S = states
        self.A = actions
        self.T = np.zeros((self.S, self.A, self.S))
        self.R = np.zeros((self.S, self.A, self.S))
        self.gamma = 1.0
        self.isNonTerminal = np.ones(self.S, dtype=bool)
        self.numTerm = 0

def constructMDP(path):
    f = open(path, 'r')
    S = int(f.readline().split()[1])
    A = int(f.readline().split()[1])
    mdp = MDP(S, A)

    end = f.readline().split()
    for i in range(1, len(end)):
        t = int(end[i])
        if t < 0:
            continue
        mdp.T[t, :, t] = 1.0
        mdp.isNonTerminal[t] = False
        mdp.numTerm += 1
    
    line = f.readline().split()
    while line[0] == 'transition':
        mdp.R[int(line[1]), int(line[2]), int(line[3])] = float(line[4])
        mdp.T[int(line[1]), int(line[2]), int(line[3])] = float(line[5])
        line = f.readline().split()
    
    mdp.gamma = float(f.readline().split()[1])

    f.close()
    return mdp

def value_eval(mdp, P):
    P = P.astype(int)
    isNonTerminal = mdp.isNonTerminal
    T1 = np.array([mdp.T[i, P[i], :] for i in range(mdp.S)])
    T1 = T1[isNonTerminal, :]
    R1 = np.array([mdp.R[i, P[i], :] for i in range(mdp.S)])
    R1 = R1[isNonTerminal, :]
    
    U = np.sum(np.multiply(T1, R1), axis = 1)
    T1 = T1[:, isNonTerminal]
    A = np.eye(np.sum(isNonTerminal)) - mdp.gamma * T1

    V = np.zeros(mdp.S)
    V[isNonTerminal] = np.linalg.solve(A, U)

    return V

def optPolicy(mdp, V):
    P = np.zeros(mdp.S, dtype=int)
    for s in range(mdp.S):
        P[s] = np.argmax(np.sum(np.multiply(mdp.T[s, :, :], mdp.R[s, :, :]), axis=1) + mdp.gamma * np.dot(mdp.T[s, :, :], V))
    return P

def valueIter(mdp):
    V = np.zeros(mdp.S)
    while True:
        V_old = V
        V = value_eval(mdp, optPolicy(mdp, V_old))
        if np.allclose(V, V_old):
            break
    return V, optPolicy(mdp, V)

def HowardPI(mdp):
    P = np.zeros(mdp.S, dtype=int)
    V = np.zeros(mdp.S)
    optFound = False
    TR = np.sum(np.multiply(mdp.T, mdp.R), axis=2)

    while not optFound:
        optFound = True
        V = value_eval(mdp, P)
        for s in range(mdp.S):
            for a in range(mdp.A):
                if a == P[s]:
                    continue
                new_V = TR[s, a] + mdp.gamma * np.dot(mdp.T[s, a, :], V)
                if new_V > V[s]:
                    P[s] = a
                    optFound = False
                    break
    return V, P

def linearProgramming(mdp):
    Vs = [pl.LpVariable('V' + str(i)) for i in range(mdp.S)]
    prob = pl.LpProblem("MDPPlanning", pl.LpMinimize)
    for s in range(mdp.S):
        for a in range(mdp.A):
            prob += Vs[s] >= pl.lpSum([mdp.T[s, a, s_] * (mdp.R[s, a, s_] + mdp.gamma*Vs[s_]) for s_ in range(mdp.S)])
        if not mdp.isNonTerminal[s]:
            prob += Vs[s] == 0
    prob += pl.lpSum(Vs)

    prob.solve(pl.PULP_CBC_CMD(msg=0))
    return np.array([v.value() for v in Vs]), optPolicy(mdp, np.array([v.value() for v in Vs]))

def main_function(mdp_path, alg):
    mdp = constructMDP(mdp_path)

    V = P = np.zeros(mdp.S)

    if alg == 'vi':
        V, P = valueIter(mdp)
    elif alg == 'hpi':
        V, P = HowardPI(mdp)
    elif alg == 'lp':
        V, P = linearProgramming(mdp)
    
    return V, P

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mdp', type=str, required=True)
    parser.add_argument('--algorithm', type=str, default='vi')

    args = parser.parse_args()
    
    V, P = main_function(args.mdp, args.algorithm)
    print('\n'.join(['%.6f %s'%(V[i], P[i]) for i in range(len(V))]))
