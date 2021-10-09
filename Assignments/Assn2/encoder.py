import argparse

def readStates(statespath):
    f = open(statespath, 'r')
    s = f.read().splitlines()
    f.close()

    return s

def readPolicy(policypath):
    f = open(policypath, 'r')
    opponent = int(f.readline())
    current_player = 3 - opponent
    opponent_policy = {}

    for line in f:
        l = line.split()
        opponent_policy[l[0]] = list(map(float, l[1:]))
    f.close()

    return current_player, opponent_policy

def gameOver(s):
    for p in ['1', '2']:
        for i in range(3):
            if s[3*i] == p and s[3*i+1] == p and s[3*i+2] == p:
                return 2
            if s[i] == p and s[i+3] == p and s[i+6] == p:
                return 2
        if s[0] == p and s[4] == p and s[8] == p:
            return 2
        if s[2] == p and s[4] == p and s[6] == p:
            return 2
    
    if not '0' in s:
        return 1

    return 0

def printTransitions(states, current_player, opponent_policy):
    term = len(states)
    for idx, s in enumerate(states):
        for a in range(9):
            if s[a] != '0':
                print("transition", idx, a, idx, -10.0, 1.0)
                continue
            _s = s[:a] + str(current_player) + s[a+1:]

            if gameOver(_s) > 0:
                print("transition", idx, a, term, 0.0, 1.0)
                continue
            
            for a2 in range(9):
                if opponent_policy[_s][a2] > 0:
                    new_s = _s[:a2] + str(3-current_player) + _s[a2+1:]
                    next_s = term
                    rew = 1.0
                    g = gameOver(new_s)
                    if g == 0:
                        next_s = states.index(new_s)
                        rew = 0.0
                    elif g == 1:
                        rew = 0.0
                    print("transition", idx, a, next_s, rew, opponent_policy[_s][a2])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--states', type=str, required=True)

    args = parser.parse_args()
    
    states = readStates(args.states)
    current_player, opponent_policy = readPolicy(args.policy)

    print("numStates", len(states)+1)
    print("numActions", 9)
    # len(states) is a win, loss or draw
    print("end",len(states))
    printTransitions(states, current_player, opponent_policy)
    print("mdptype episodic")
    print("discount ", 1)