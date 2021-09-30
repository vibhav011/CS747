import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
import numpy as np

class DataInstance:
    def __init__(self):
        self.algos = set()
        self.scales = set()
        self.horizons = set()

        x = lambda : defaultdict(lambda : defaultdict(float))
        self.totREG = defaultdict(x)
        self.totHIGHS = defaultdict(x)
        self.nums = defaultdict(x)

    def addLine(self, algorithm, scale, horizon, REG, HIGHS):
        self.nums[algorithm][scale][horizon] += 1
        self.totREG[algorithm][scale][horizon] += REG
        self.totHIGHS[algorithm][scale][horizon] += HIGHS
        self.scales.add(scale)
        self.horizons.add(horizon)
        self.algos.add(algorithm)

    def getAvg(self, algorithm, scale, horizon, reg = 1):
        if self.nums[algorithm][scale][horizon] == 0:
            return 0
        
        if reg == 1:
            return self.totREG[algorithm][scale][horizon] / self.nums[algorithm][scale][horizon]
        return self.totHIGHS[algorithm][scale][horizon] / self.nums[algorithm][scale][horizon]

def getMaxHigh(instance, th):
    f = open(instance, 'r')
    sp = list(map(float, f.readline().split()))

    max_high = 0
    for line in f:
        pdf = list(map(float, line.split()))
        if len(pdf) == 0:
            break
        
        hi = 0
        for i in range(len(pdf)):
            hi += int(sp[i] > th) * pdf[i]
        if hi > max_high:
            max_high = hi
    
    return max_high

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=str, required=True)
    parser.add_argument('--threshold', type=float, required=True)
    args = parser.parse_args()

    INSTANCE = args.instance
    THRESHOLD = args.threshold

    data = defaultdict(DataInstance)
    with open("outputData.txt", "r") as f:
        for line in f:
            l = [d.strip() for d in line.split(",")]
            
            if not INSTANCE in l[0] or not float(l[5]) == THRESHOLD:
                continue

            data[l[0]].addLine(l[1], float(l[4]), int(l[6]), float(l[7]), int(l[8]))

    algos = list(list(data.values())[0].algos)
    scales = list(list(data.values())[0].scales)
    scales.sort()
    horizons = list(list(data.values())[0].horizons)
    horizons.sort()
    
    if "task1" in INSTANCE:
        for algo in algos:
            yps = [data[INSTANCE].getAvg(algo, scales[0], hor) for hor in horizons]
            plt.plot(horizons, yps, label=algo)
        plt.legend()
        plt.xscale("log")
        plt.ylabel("Avg. Regret")
        plt.xlabel("Horizon")
        plt.title("Instance: Task 1, " + INSTANCE.split('/')[-1])
        plt.show()
    
    elif "task2" in INSTANCE:
        instances = ["../instances/instances-task2/i-1.txt", "../instances/instances-task2/i-2.txt", "../instances/instances-task2/i-3.txt", "../instances/instances-task2/i-4.txt", "../instances/instances-task2/i-5.txt"]
        for ins in instances:
            yps = [data[ins].getAvg(algos[0], s, horizons[0]) for s in scales]
            plt.plot(scales, yps, label=ins)
            i = np.argmin(yps)
            print("Best scale for " + ins + ": " + str(scales[i]))
            
        plt.legend()
        plt.ylabel("Avg. Regret")
        plt.xlabel("Scale")
        plt.title("Task 2")
        plt.show()
    
    elif "task3" in INSTANCE:
        yps = [data[INSTANCE].getAvg(algos[0], scales[0], hor) for hor in horizons]
        plt.plot(horizons, yps, label=algos[0])
        plt.legend()
        plt.xscale("log")
        plt.ylabel("Avg. Regret")
        plt.xlabel("Horizon")
        plt.title("Instance: Task 3, " + INSTANCE.split('/')[-1])
        plt.show()
    
    elif "task4" in INSTANCE:
        max_high = getMaxHigh(INSTANCE, THRESHOLD)
        yps = [max_high*hor - data[INSTANCE].getAvg(algos[0], scales[0], hor, 0) for hor in horizons]
        plt.plot(horizons, yps, label=algos[0])
        plt.legend()
        plt.xscale("log")
        plt.ylabel("Avg. HIGHS_REGRET")
        plt.xlabel("Horizon")
        plt.title("Instance: Task 4, " + INSTANCE.split('/')[-1] + "\nThreshold: " + str(THRESHOLD))
        plt.show()
