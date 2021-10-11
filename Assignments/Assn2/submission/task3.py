import subprocess,os
import numpy as np
from pathlib import Path
import os

STATES = ["./data/attt/states/states_file_p1.txt", "./data/attt/states/states_file_p2.txt"]
DIR = "./policies"
P1_INITIAL_POLICY = "./data/attt/policies/p1_policy2.txt"
P2_INITIAL_POLICY = None

def convertToPolicy(cmd_output):
    lines = cmd_output.splitlines()
    policy = {}

    for line in lines[1:]:
        if line == '':
            continue
        l = line.split()
        policy[l[0]] = np.argmax(list(map(float, l[1:])))

    return policy

def findOptimalPolicy(opponent_policypath, current_policypath, current_player):
    cmd_encoder = "python3","encoder.py","--policy",opponent_policypath,"--states",STATES[current_player-1]
    f = open('encoded_mdp.txt','w')
    subprocess.call(cmd_encoder,stdout=f)
    f.close()

    cmd_planner = "python3","planner.py","--mdp","encoded_mdp.txt"
    f = open('opt_policy.txt','w')
    subprocess.call(cmd_planner,stdout=f)
    f.close()

    cmd_decoder = "python3","decoder.py","--value-policy","opt_policy.txt","--states",STATES[current_player-1] ,"--player-id",str(current_player)
    cmd_output = subprocess.check_output(cmd_decoder,universal_newlines=True)
    
    f = open(current_policypath,'w')
    f.write(cmd_output)
    f.close()

    os.remove('encoded_mdp.txt')
    os.remove('opt_policy.txt')

    return convertToPolicy(cmd_output)

if __name__ == '__main__':
    Path(DIR).mkdir(exist_ok=True)

    policy_paths = [P1_INITIAL_POLICY, P2_INITIAL_POLICY]
    start = 0 if policy_paths[0] is not None else 1

    policy_paths[1-start] = f"{DIR}/opt_{2-start}_0.txt"

    print(f"Starting with a policy for P{1+start} present at {policy_paths[start]}\n")

    policies = [None, None]
    policies_new = [None, None]
    policies[start] = convertToPolicy(open(policy_paths[start]).read())
    policies[1-start] = findOptimalPolicy(policy_paths[start], policy_paths[1-start], 2-start)

    for i in range(1, 11):
        policy_paths[start] = f"{DIR}/opt_{1+start}_{i}.txt"
        policies_new[start] = findOptimalPolicy(policy_paths[1-start], policy_paths[start], 1+start)

        policy_paths[1-start] = f"{DIR}/opt_{2-start}_{i}.txt"
        policies_new[1-start] = findOptimalPolicy(policy_paths[start], policy_paths[1-start], 2-start)

        dif1 = sum([1 if policies[0][k] != policies_new[0][k] else 0 for k in policies[0]])
        dif2 = sum([1 if policies[1][k] != policies_new[1][k] else 0 for k in policies[1]])

        print("Iteration: ",i)
        print(f"P1 policy differs at {dif1} states")
        print(f"P2 policy differs at {dif2} states")
        print("")

        policies[0], policies[1] = policies_new[0], policies_new[1]
    