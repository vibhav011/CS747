import subprocess,os
import numpy as np

STATES = ["./data/attt/states/states_file_p1.txt", "./data/attt/states/states_file_p2.txt"]

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

    p1_policy_path = "./data/attt/policies/p1_policy1.txt"
    p2_policy_path = "opt_2_0.txt"

    print(f"Starting with a policy for P1 present at {p1_policy_path}\n")

    p1_policy = convertToPolicy(open(p1_policy_path).read())
    p2_policy = findOptimalPolicy(p1_policy_path, p2_policy_path, 2)

    i = 1
    while True:
        if p1_policy_path[:3] == "opt":
            os.remove(p1_policy_path)
        p1_policy_path = "opt_1_" + str(i) + ".txt"
        p1_policy_new = findOptimalPolicy(p2_policy_path, p1_policy_path, 1)

        if p2_policy_path[:3] == "opt":
            os.remove(p2_policy_path)
        p2_policy_path = "opt_2_" + str(i) + ".txt"
        p2_policy_new = findOptimalPolicy(p1_policy_path, p2_policy_path, 2)

        dif1 = sum([1 if p1_policy[k] != p1_policy_new[k] else 0 for k in p1_policy])
        dif2 = sum([1 if p2_policy[k] != p2_policy_new[k] else 0 for k in p2_policy])

        print("Iteration: ",i)
        print(f"P1 policy differs at {dif1} states")
        print(f"P2 policy differs at {dif2} states")
        print("")

        p1_policy, p2_policy = p1_policy_new, p2_policy_new
        i = i + 1

        if dif1 == 0 and dif2 == 0:
            break
    
    print(f"Final policy files:\nP1: {p1_policy_path}\nP2: {p2_policy_path}")