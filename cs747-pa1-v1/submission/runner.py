from bandit import main_function

def callBandit(instances, algorithms, epsilon, scales, thresholds, horizons):
    # lines = []
    f = open("outputData.txt", "a")
    for instance in instances:
        for algorithm in algorithms:
            for seed in range(50):
                for scale in scales:
                    for threshold in thresholds:
                        for horizon in horizons:
                            REG, HIGHS = main_function(instance, algorithm, seed, epsilon, scale, threshold, horizon)
                            f.write(", ".join([instance, algorithm, str(seed), str(epsilon), str(scale), str(threshold), str(horizon), '%.3f'%REG, str(HIGHS)]) + '\n')
    
    # lines.sort()
    f.close()

def task1():
    instances = ["../instances/instances-task1/i-1.txt", "../instances/instances-task1/i-2.txt", "../instances/instances-task1/i-3.txt"]
    algorithms = ["epsilon-greedy-t1", "ucb-t1", "kl-ucb-t1", "thompson-sampling-t1"]
    horizons = [100, 400, 1600, 6400, 25600, 102400]
    
    callBandit(instances, algorithms, 0.02, [2], [0], horizons)

def task2():
    instances = ["../instances/instances-task2/i-1.txt", "../instances/instances-task2/i-2.txt", "../instances/instances-task2/i-3.txt", "../instances/instances-task2/i-4.txt", "../instances/instances-task2/i-5.txt"]
    algorithms = ["ucb-t2"]
    scales = [i/100 for i in range(2, 31, 2)]
    horizons = [10000]
    
    callBandit(instances, algorithms, 0.02, scales, [0], horizons)

def task3():
    instances = ["../instances/instances-task3/i-1.txt", "../instances/instances-task3/i-2.txt"]
    algorithms = ["alg-t3"]
    horizons = [100, 400, 1600, 6400, 25600, 102400]
    
    callBandit(instances, algorithms, 0.02, [2], [0], horizons)

task1()
task2()
task3()