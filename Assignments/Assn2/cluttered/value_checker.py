import sys
import numpy as np

if len(sys.argv) != 3:
    print("python3 value_checker.py path_file1 path_file2")

with open(sys.argv[1]) as f:
    content = f.readlines()
    v1 = np.array([float(line.strip().split()[0]) for line in content if line.strip()!=''])
    p1 = np.array([float(line.strip().split()[1]) for line in content if line.strip()!=''])

with open(sys.argv[2]) as f:
    content = f.readlines()
    v2 = np.array([float(line.strip().split()[0]) for line in content if line.strip()!=''])
    p2 = np.array([float(line.strip().split()[1]) for line in content if line.strip()!=''])


if abs(v1 - v2).max() < 1e-5:
    print("Correct")
else:
    print("Incorrect")

if abs(p1 - p2).max() < 1e-5:
    print("Correct")
else:
    print("Incorrect")