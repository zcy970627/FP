#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib.pyplot as plt
from efficient_nondominated import NonDominatedSorting1

hv = [[],[],[],[],[],[]]
hv_mean = []
for i in range(6):  # five kinds of neighbor function
    for j in range(5): # run each neighbor function for 5 times
        df = open('/work/publications/norcas2020/experiments/results/'+str(i+1)+'/hv_list'+str(j+1)+'.pkl','rb')
        data = pickle.load(df)
        hv[i].append(data)
        df.close()
    zipped = list(zip(*(hv[i])))
    mean = np.mean(zipped,axis=1)

    f = open('/work/publications/norcas2020/experiments/results/hv'+str(i+1)+'_mean.pkl','wb')
    pickle.dump(mean,f)
    f.close()

    hv_mean.append(mean)

hv_nsga = []
for i in range(5):
    df = open('/work/publications/norcas2020/experiments/results/nsga2/hv'+str(i+1)+'_nsga2.pkl','rb')
    data = pickle.load(df)
    hv_nsga.append(data)
    zipped = list(zip(*(hv_nsga)))
    mean_nsga = np.mean(zipped,axis=1)
# print(hv_nsga)
# print(len(hv_nsga))
# print(len(mean_nsga))


# convergence figure
x = list(range(len(hv_mean[0])))
x_axis = [(k+1) * 100 for k in x]
plt.plot(x_axis, hv_mean[0], color='green', label='Flat Sweep')
plt.plot(x_axis, hv_mean[1], color='black', label='Flat Sweep (twice)')
plt.plot(x_axis, hv_mean[2], color='blue', label='OSTC Sweep')
plt.plot(x_axis, hv_mean[3], color='yellow', label='Sparse-Table-Regenerate Sweep')
plt.plot(x_axis, hv_mean[4], color='purple', label='Block-wise Sweep')
plt.plot(x_axis, hv_mean[5], color='pink', label='RGB Channel-wise Sweep')
# plt.plot(x_axis, mean_nsga, color='red', label='NSGA-II')
# plt.xlim(39000,40000)
# plt.ylim(22000,22500)
plt.legend()
plt.xlabel('Evaluations')
plt.ylabel('Hypervolume')
plt.savefig('/work/publications/norcas2020/experiments/results/convergence.eps')
plt.show()


