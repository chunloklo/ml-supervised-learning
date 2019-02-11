import scipy as sp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm

############## Options to generate nice figures
fig_width_pt = 250.0  # Get this from LaTeX using \showthe\column-width
inches_per_pt = 1.0 / 72.27  # Convert pt to inch
golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = fig_width * golden_mean  # height in inches
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
# test_acc = np.load("npData/20190206-015417_MAGIC-ADABoost_n_estimators-test_acc.npy")
# test_acc1 = np.load("npData/20190206-021010_MAGIC-adaboost_n_estimators-test_acc-depth2.npy")
# test_acc2 = np.load("npData/20190206-120107-MAGIC-adaboost-learning_rate-test_acc-depth3.npy")
# test_acc3 = np.load("npData/20190206-120139-MAGIC-adaboost-learning_rate-test_acc-depth4.npy")
# vals = np.load("npData/20190206-120139-MAGIC-adaboost-learning_rate-vals-depth4.npy")
# # test_acc4 = np.load("npData/20190209-001547-COVER-adaboost-learning_rate-test_acc-p00001.npy")

fpr = np.load("npData/20190209-173429-MAGIC-NN-fpr.npy")
tpr = np.load("npData/20190209-173526-MAGIC-NN-tpr.npy")


tpr1 = np.load("npData/20190209-174555-MAGIC-ADA-tpr.npy")
fpr1 = np.load("npData/20190209-174555-MAGIC-ADA-fpr.npy")



# ax = plt.subplot(121)
# ax.set_title("MAGIC Gamma Telescope")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.plot(fpr, tpr, color=my_yellow, label="Neural Network")
plt.plot(fpr1, tpr1, color=my_blue, label="AdaBoost")
plt.plot([0, 1], [0, 1], color=dark_gray, linestyle='dashed')



plt.legend()

# # ax = plt.subplot(122)
# ax.set_title("Cover Type")
# plt.xlabel("Number of Weak Learners")
# plt.ylabel("Validation Accuracy")
# # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

# plt.rcParams.update({'legend.fontsize': 5.6})

# plt.plot(vals, test_acc, marker='.',  label="LR: 0.1, Max depth: 1")
# plt.plot(vals, test_acc1, marker='.',  label="LR: 0.01, Max depth: 1")
# plt.plot(vals, test_acc2, marker='.',  label="LR: 0.001, Max depth: 1")
# # plt.plot(vals, test_acc3, marker='.',  label="LR: 0.0001, Max depth: 1")
# plt.plot(vals2, test_acc5, marker='.',  color=my_blue, label="LR: 0.0001, Max depth = 1")
# plt.plot(vals, test_acc4, marker='.',  label="LR: 0.00001, Max depth: 1")

# plt.plot(vals2, test_acc6, marker='.', color=my_yellow, label="LR: 0.0001, Tuned Decision Tree")





my_legend = plt.legend()
my_legend.get_frame().set_alpha(0.8)

# print(np.max(test_acc))
# print(vals[np.argmax(test_acc)])
plt.savefig("../tex/images/MAGIC-ROC.pdf")
plt.show()
