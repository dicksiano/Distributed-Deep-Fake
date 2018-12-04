import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import savgol_filter

filepath1 = os.path.join('pesos', 'weightsdi1.log')
filepath2 = os.path.join('pesos', 'weigths_nogs.log')

num_lines1 = sum(1 for line in open(filepath1, 'r'))
num_lines2 = sum(1 for line in open(filepath2, 'r'))

print('Num Lines', num_lines1, num_lines2)

file1 = open(filepath1, 'r')
file2 = open(filepath2, 'r')

t1 = []
f1lossA = []
f1lossB = []

t2 = []
f2lossA = []
f2lossB = []

for line in file1:
    time, lossA, lossB = line.split(';')
    hr, min, seg = time.split(':')
    seg = seg.split(',')[0]
    hr = hr[-2:]
    # print(time, hr, float(seg))
    # input()
    t1.append(float(hr) * 60 + 60 * float(min) + float(seg))
    f1lossA.append(float(lossA))
    f1lossB.append(float(lossB))
t1 = np.array(t1)
t1 -= t1[0]
for line in file2:
    time, lossA, lossB = line.split(';')
    hr, min, seg = time.split(':')
    seg = seg.split(',')[0]
    hr = hr[-2:]
    # print(time, float(hr), float(hr) * 3600 + 60 * float(min) + float(seg))
    # input()

    t2.append(float(hr) * 3600 + 60 * float(min) + float(seg))
    f2lossA.append(float(lossA))
    f2lossB.append(float(lossB))
t2 = np.array(t2)
t2 -= t2[0]
# print(t2[0:280])

fig = plt.figure()
ax = plt.axes()

x = np.linspace(0,num_lines1,num_lines1)
# print(x)
# yhat1 = savgol_filter(f1lossA, 51,3)
plt.plot(t1, f1lossA,'g',ms=1, label='4 GPU')
# plt.plot(t1, yhat1,'g')
# plt.plot(x, f1lossB)
# yhat2 = savgol_filter(f2lossA[0:20],19,3)
plt.plot(t2[0:15], f2lossA[0:15],'r', ms=1, label='Distribuido Local')
# plt.plot(t2[0:len(yhat2)], yhat2,'b')
plt.title('Comparando GPU x Distribuido Local')
plt.ylabel('loss autoencoderA')
plt.xlabel('tempo (s)')
plt.legend()
plt.show()
