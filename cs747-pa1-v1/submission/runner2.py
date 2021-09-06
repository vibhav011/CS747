from bandit import main_function
import threading

def main_wrapper(instance, algorithm, seed, epsilon, scale, threshold, horizon, f):
    REG, HIGHS = main_function(instance, algorithm, seed, epsilon, scale, threshold, horizon)
    f.write(", ".join([instance, algorithm, str(seed), str(epsilon), str(scale), str(threshold), str(horizon), '%.3f'%REG, str(HIGHS)]) + '\n')

def callBandit(instances, algorithms, horizons, scale, epsilon, threshold):
    # lines = []
    f = open("outputData2.txt", "a")
    threads = []
    for instance in instances:
        for algorithm in algorithms:
            for horizon in horizons:
                for seed in range(50):
                    threads.append(threading.Thread(target=main_wrapper, args = (instance, algorithm, seed, epsilon, scale, threshold, horizon, f)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    f.close()
    # lines.sort()
    # with open("outputData2.txt", "a") as f:
    #     f.write("\n".join(lines) + "\n")


def task1():
    instances = ["../instances/instances-task1/i-1.txt", "../instances/instances-task1/i-2.txt", "../instances/instances-task1/i-3.txt"]
    algorithms = ["epsilon-greedy-t1", "ucb-t1", "kl-ucb-t1", "thompson-sampling-t1"]
    horizons = [100, 400, 1600, 6400, 25600, 102400]
    
    callBandit(instances, algorithms, horizons, 2, 0.02, 0)

task1()
