import numpy as np
import argparse
import pulp as pl
class MDP:
    def __init__(self, states, actions):
        self.S = states
        self.A = actions
        self.STR = [[[] for a in range(actions)] for s in range(states)]
        self.TR = np.zeros((states, actions), dtype=float)
        self.isNonTerminal = np.ones((states, actions), dtype=bool)
        self.gamma = 1.0
    
    def computeTerminal(self):
        for s in range(self.S):
            for a in range(self.A):
                if len(self.STR[s][a]) == 0:
                    self.isNonTerminal[s, a] = False
                    continue
                if self.STR[s][a][0][0] == s and self.STR[s][a][0][1] == 1.0 and self.STR[s][a][0][2] == 0.0:
                    self.isNonTerminal[s, a] = False

def constructMDP(path):
    f = open(path, 'r')
    S = int(f.readline().split()[1])
    A = int(f.readline().split()[1])
    mdp = MDP(S, A)

    end = f.readline().split()
    # for i in range(1, len(end)):
    #     t = int(end[i])
    #     if t < 0:
    #         continue
    #     for a in range(A):
    #         mdp.STR[t][a].append((t, 1.0, 0.0))

    line = f.readline().split()
    while line[0] == 'transition':
        a, b, c, d, e = int(line[1]), int(line[2]), int(line[3]), float(line[4]), float(line[5])
        mdp.STR[a][b].append((c, e, d))
        mdp.TR[a][b] += d * e
        line = f.readline().split()
    
    for s in range(S):
        for a in range(A):
            mdp.STR[s][a].sort()
    
    mdp.computeTerminal()
    
    mdp.gamma = float(f.readline().split()[1])

    f.close()
    return mdp

def value_eval(mdp, P):
    P = P.astype(int)
    isNonTerminal = mdp.isNonTerminal[np.arange(mdp.S), P]
    k = np.sum(isNonTerminal)
    
    T1 = np.zeros((mdp.S,mdp.S))
    for i in range(mdp.S):
        if not isNonTerminal[i]:
            continue
        for t in mdp.STR[i][P[i]]:
            s2 = t[0]
            if not isNonTerminal[s2]:
                continue
            T1[i, s2] = t[1]
    T1 = T1[:, isNonTerminal]
    T1 = T1[isNonTerminal, :]

    U = mdp.TR[np.arange(mdp.S), P]
    U = U[isNonTerminal]

    A = np.eye(k) - mdp.gamma * T1

    V = np.zeros(mdp.S)
    V[isNonTerminal] = np.linalg.solve(A, U)

    return V

def calcQ(mdp, s, a, V):
    Q = 0
    for t in mdp.STR[s][a]:
        Q += t[1] * V[t[0]]
    Q = mdp.TR[s,a] + mdp.gamma * Q
    return Q

def optPolicy(mdp, V):
    P = np.zeros(mdp.S, dtype=int)
    for s in range(mdp.S):
        Qs = [calcQ(mdp, s, a, V) if len(mdp.STR[s][a]) > 0 else -np.inf for a in range(mdp.A)]
        P[s] = np.argmax(Qs)
    return P

def valueIter(mdp):
    V = np.zeros(mdp.S)
    V_old = np.zeros(mdp.S)
    while True:
        for s in range(mdp.S):
            Qs = [calcQ(mdp, s, a, V_old) for a in range(mdp.A) if len(mdp.STR[s][a]) > 0]
            V[s] = np.max(Qs) if len(Qs) > 0 else 0.0
        if np.allclose(V, V_old, rtol=0, atol=1e-9):
            break
        V_old = V.copy()
    return V, optPolicy(mdp, V)

def HowardPI(mdp):
    P = np.zeros(mdp.S, dtype=int)
    for s in range(mdp.S):
        for a in range(mdp.A):
            if len(mdp.STR[s][a]) == 0:
                continue
            P[s] = a
            break
    V = np.zeros(mdp.S)
    optFound = False

    while not optFound:
        optFound = True
        V = value_eval(mdp, P)

        for s in range(mdp.S):
            if not mdp.isNonTerminal[s, P[s]]:
                continue
            for a in range(mdp.A):
                if a == P[s]:
                    continue

                if calcQ(mdp, s, a, V) > V[s]:
                    P[s] = a
                    optFound = False
                    break
    return V, P

def linearProgramming(mdp):
    isNonTerminal = np.any(mdp.isNonTerminal, axis=1)

    Vs = [pl.LpVariable('V' + str(i)) for i in range(mdp.S)]
    prob = pl.LpProblem("MDPPlanning", pl.LpMinimize)
    for s in range(mdp.S):
        for a in range(mdp.A):
            prob += Vs[s] >= pl.lpSum([t[1] * (t[2] + mdp.gamma*Vs[t[0]]) for t in mdp.STR[s][a]])
        if not isNonTerminal[s]:
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
    parser.add_argument('--algorithm', type=str, default='hpi')

    args = parser.parse_args()
    
    V, P = main_function(args.mdp, args.algorithm)
    print('\n'.join(['%.6f %s'%(V[i], P[i]) for i in range(len(V))]))
