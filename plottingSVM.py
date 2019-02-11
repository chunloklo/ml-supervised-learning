import scipy as sp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm

############## Options to generate nice figures
fig_width_pt = 500.0  # Get this from LaTeX using \showthe\column-width
inches_per_pt = 1.0 / 72.27  # Convert pt to inch
golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = fig_width * golden_mean * 2/3  # height in inches
fig_size = [fig_width, fig_height]

############## Colors I like to use
my_yellow = [235. / 255, 164. / 255, 17. / 255]
my_blue = [58. / 255, 93. / 255, 163. / 255]
dark_gray = [68./255, 84. /255, 106./255]
my_red = [163. / 255, 93. / 255, 58. / 255]

my_color = dark_gray # pick color for theme

params_keynote = {
    'axes.labelsize': 16,
    'font.size': 16,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    # 'text.usetex': True,
    # 'text.latex.preamble': '\\usepackage{sfmath}',
    'font.family': 'sans-serif',
    'figure.figsize': fig_size
}
############## Parameters I use for IEEE papers
params_ieee = {
    'figure.autolayout' : True,
    'axes.labelsize': 8,
    'font.size': 8,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    # 'text.usetex': True,
    # 'text.latex.preamble': '\\usepackage{sfmath}',
    'font.family': 'sans-serif',
    'figure.figsize': fig_size
}

############## Choose parameters you like
matplotlib.rcParams.update(params_ieee)


import numpy as np
test_acc = np.load("npData/20190206-171558-MAGIC-LinearSVM-C-test_acc.npy")
train_acc = np.load("npData/20190206-171558-MAGIC-LinearSVM-C-train_acc.npy")
vals = np.load("npData/20190206-171558-MAGIC-LinearSVM-C-vals.npy")

test_acc2 = np.load("npData/20190206-171852-MAGIC-SVM-C-test_acc.npy")
train_acc2 = np.load("npData/20190206-171852-MAGIC-SVM-C-train_acc.npy")
vals2 = np.load("npData/20190206-171558-MAGIC-LinearSVM-C-vals.npy")

ax = plt.subplot(121)
ax.set_title("MAGIC Gamma Telescope")
plt.xlabel("C")
plt.ylabel("Accuracy")
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# plt.grid(True)
plt.plot(vals, test_acc, marker='.', color='orange', label="Linear Validation")
plt.plot(vals, train_acc, marker='.',color='r', label="Linear Train")
plt.plot(vals, test_acc2, marker='.', color='darkcyan', label="RBF Validation")
plt.plot(vals, train_acc2, marker='.', color='royalblue',label="RBF Train")

# acc1 = np.load("npData/20190206-132640-MAGIC-ADABoost-n_estimators-test_acc.npy")
# acc2 = np.load("npData/20190206-021010_MAGIC-adaboost_n_estimators-test_acc-depth2.npy")
# acc3 = np.load("npData/20190206-120107-MAGIC-adaboost-learning_rate-test_acc-depth3.npy")
# acc4 = np.load("npData/20190206-120139-MAGIC-adaboost-learning_rate-test_acc-depth4.npy")
# vals2 = np.load("npData/20190206-021010_MAGIC-adaboost_n_estimators-vals-depth2.npy")
# plt.plot(vals2, acc1, label='max_depth = 1', marker='.')
# plt.plot(vals2, acc2, label='max_depth = 2', marker='.')
# plt.plot(vals2, acc3, label='max_depth = 3', marker='.')
# plt.plot(vals2, acc4, label='max_depth = 4', marker='.')
plt.legend()


test_acc = np.load("npData/20190208-144302-COVER-LinearSVM-C-test_acc.npy")
train_acc = np.load("npData/20190208-144302-COVER-LinearSVM-C-train_acc.npy")
vals = np.load("npData/20190208-144302-COVER-LinearSVM-C-vals.npy")
test_acc2 = np.load("npData/20190208-145244-COVER-SVM-C-test_acc.npy")
train_acc2 = np.load("npData/20190208-145244-COVER-SVM-C-train_acc.npy")
vals2 = np.load("npData/20190208-144302-COVER-LinearSVM-C-vals.npy")


ax = plt.subplot(122)
ax.set_title("Cover Type")
plt.xlabel("C")
plt.ylabel("Accuracy")
# plt.plot(vals2, test_acc2, marker='.', color=my_yellow, label="Test Accuracy")
# plt.plot(vals2, train_acc2, marker='.', color=my_blue, label="Train Accuracy")
plt.plot(vals, test_acc, marker='.', color='orange', label="Linear Validation")
plt.plot(vals, train_acc, marker='.',color='r', label="Linear Train")
plt.plot(vals, test_acc2, marker='.', color='darkcyan', label="RBF Validation")
plt.plot(vals, train_acc2, marker='.', color='royalblue',label="RBF Train")
my_legend = plt.legend()
my_legend.get_frame().set_alpha(0.8)

print(np.max(test_acc))
print(vals[np.argmax(test_acc)])
plt.savefig("../tex/images/SVM-Tuning.pdf")
plt.show()
