
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from numpy import genfromtxt

models = ["0", "1"]
ndim = ["50", "100"]
nlayers = ["1", "3"]


plt.rc('font', family='serif')
plt.rc('xtick', labelsize='medium')
plt.rc('ytick', labelsize='medium')

# fig = plt.figure(figsize=(4, 3))
# ax = fig.add_subplot(1, 1, 1)
#
# x = np.linspace(1., 8., 30)
# ax.plot(x, x ** 1.5, color='k', ls='solid')
# ax.plot(x, 20/x, color='0.50', ls='dashed')
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Temperature (K)')

csvpath = "/data/amar/npl_project/tmp/"

for i in range(len(models)):

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8,5))


    for (n_row, n_col), axes in np.ndenumerate(ax):
        axes.set_title('(hdim %s, nlayers %s)' % (ndim[n_row], nlayers[n_col]))

        name = csvpath+"model_"+models[i]+"_hdim_"+ndim[n_row]+"_nlayers_"+nlayers[n_col]+"_loss.csv"
        loss = genfromtxt(name, delimiter=',')

        vname = csvpath+"model_" + models[i] + "_hdim_" + ndim[n_row] + "_nlayers_" + nlayers[n_col]+"_val_loss.csv"
        vloss = genfromtxt(vname, delimiter=',')

        axes.plot(loss[:, 1], loss[:, 2], label='Training Loss')
        axes.legend()
        axes.plot(vloss[:, 1], vloss[:, 2], label='Testing Loss')
        axes.legend()
        axes.set_xlabel('Epochs')
        axes.set_ylabel('Loss')

    plt.tight_layout()
    plt.savefig("model_"+str(i)+"_lossfig.pdf")
    plt.savefig("model_" + str(i) + "_lossfig.png")

plt.show()


import matplotlib.pyplot as plt
import numpy as np

em = np.array([81.25, 72.65, 78.12, 75.00])
f1 = np.array([96.65, 95.56, 95.89, 94.61])
ed = np.array([0.21, 0.31, 0.36, 0.42])
gm = np.array([86.72, 85.15, 89.94, 89.06])

x = np.array([em, f1, ed, gm])
x[0, :] /=x[0,0]
x[1, :] /= x[1,0]
x[2, :] /= x[2,0]
x[3, :] /= x[3,0]

ind = np.arange(32)

ax = plt.subplot(111)
w = 1
ax.bar(ind[1::8], x[:, 0], width=w, color='b', align='center', label='with directions - Test-Repeated')
ax.legend()
ax.bar(ind[2::8], x[:, 1], width=w, color='g', align='center', label='with directions - Test-New')
ax.legend()
ax.bar(ind[3::8], x[:, 2], width=w, color='r', align='center', label='without directions - Test-Repeated')
ax.legend()
ax.bar(ind[4::8], x[:, 3], width=w, color='y', align='center', label='without directions - Test-New')
ax.legend()

ax.set_title('Metrics')
ax.set_xticks(ind[1::8] + 1)
ax.set_xticklabels(('EM', 'F1', 'ED', 'GM'))

ax.autoscale_view()


ax.autoscale(tight=True)

plt.tight_layout()
plt.savefig("metricsfig.pdf")
plt.savefig("metricsfig.png")

plt.show()