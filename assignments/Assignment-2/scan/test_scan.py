import subprocess
import os
import os.path as osp

p = subprocess.Popen(["make"])
p.wait(100)
i = 2
while i < 2**15:
    print("test on input_size:", i)
    p3 = subprocess.Popen(["./cudaScan -m scan -n {} -i test1".format(i)], shell=True)
    p3.wait(100)
    print("test on input_size:", i+1)
    p3 = subprocess.Popen(["./cudaScan -m scan -n {} -i test1".format(i+1)], shell=True)
    p3.wait(100)
    print("test on input_size:", i+2)
    p3 = subprocess.Popen(["./cudaScan -m scan -n {} -i test1".format(i+2)], shell=True)
    p3.wait(100)
    print("test on input_size:", i+3)
    p3 = subprocess.Popen(["./cudaScan -m scan -n {} -i test1".format(i+3)], shell=True)
    p3.wait(100)
    print("test on input_size:", i+4)
    p3 = subprocess.Popen(["./cudaScan -m scan -n {} -i test1".format(i+4)], shell=True)
    p3.wait(100)
    i *= 2
