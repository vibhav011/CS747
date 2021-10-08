import argparse

def readStates(statespath):
    f = open(statespath, 'r')
    s = f.read().splitlines()
    f.close()

    return s

def readValuePolicy(policypath):
    f = open(policypath, 'r')
    s = f.read().splitlines()
    policy = [int(u.split()[1]) for u in s]
    f.close()

    return policy[:-2]

def printOptPolicy(states, policy):
    current = [0.0] * 9

    for s, a in zip(states, policy):
        current[a] = 1.0
        print(s, " ".join(map(str, current)))
        current[a] = 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--value-policy', type=str, required=True)
    parser.add_argument('--states', type=str, required=True)
    parser.add_argument('--player-id', type=int, required=True)

    args = parser.parse_args()

    states = readStates(args.states)
    policy = readValuePolicy(args.value_policy)
    current_player = args.player_id

    print(current_player)
    printOptPolicy(states, policy)
