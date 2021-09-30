from bandit import main_function
from threading import Thread, Lock

def main_wrapper(instances, algorithms, seed, epsilon, scales, thresholds, horizons, f, mutex):
    for instance in instances:
        for algorithm in algorithms:
            for scale in scales:
                for threshold in thresholds:
                    for horizon in horizons:
                        REG, HIGHS = main_function(instance, algorithm, seed, epsilon, scale, threshold, horizon)
                        mutex.acquire()
                        f.write(", ".join([instance, algorithm, str(seed), str(epsilon), str(scale), str(threshold), str(horizon), '%.3f'%REG, str(HIGHS)]) + '\n')
                        mutex.release()

def callBandit(instances, algorithms, epsilon, scales, thresholds, horizons):
    f = open("outputData_thread.txt", "a")
    threads = []
    mutex = Lock()
    for seed in range(50):
        threads.append(Thread(target=main_wrapper, args=(instances, algorithms, seed, epsilon, scales, thresholds, horizons, f, mutex)))
                        
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()    
    f.close()

def task1():
    instances = ["../instances/instances-task1/i-1.txt", "../instances/instances-task1/i-2.txt", "../instances/instances-task1/i-3.txt"]
    algorithms = ["epsilon-greedy-t1", "ucb-t1", "kl-ucb-t1", "thompson-sampling-t1"]
    horizons = [100, 400, 1600, 6400, 25600, 102400]
    
    callBandit(instances, algorithms, 0.02, [2], [0], horizons)
    print("Task 1 done")

def task2():
    instances = ["../instances/instances-task2/i-1.txt", "../instances/instances-task2/i-2.txt", "../instances/instances-task2/i-3.txt", "../instances/instances-task2/i-4.txt", "../instances/instances-task2/i-5.txt"]
    algorithms = ["ucb-t2"]
    scales = [i/100 for i in range(2, 31, 2)]
    horizons = [10000]
    
    callBandit(instances, algorithms, 0.02, scales, [0], horizons)
    print("Task 2 done")

def task3():
    instances = ["../instances/instances-task3/i-1.txt", "../instances/instances-task3/i-2.txt"]
    algorithms = ["alg-t3"]
    horizons = [100, 400, 1600, 6400, 25600, 102400]
    
    callBandit(instances, algorithms, 0.02, [2], [0], horizons)
    print("Task 3 done")

def task4():
    instances = ["../instances/instances-task4/i-1.txt", "../instances/instances-task4/i-2.txt"]
    algorithms = ["alg-t4"]
    horizons = [100, 400, 1600, 6400, 25600, 102400]
    thresholds = [0.2, 0.6]
    
    callBandit(instances, algorithms, 0.02, [2], thresholds, horizons)
    print("Task 4 done")

task1()
task2()
task3()
task4()