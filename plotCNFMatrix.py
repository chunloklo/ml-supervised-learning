import numpy as np
import matplotlib.pyplot as plt
import itertools

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





def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # if normalize:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title('MAGIC Gamma Telescope Data Set')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    #plt.colorbar()

plt.tight_layout()
ax = plt.subplot(121)
ax.set_title("MAGIC Gamma Telescope")
# cnf_matrix= np.load("npData/20190208-011721-MAGIC-decisionTree-cnf_matrix.npy")
cnf_matrix = np.array([[1105,  125],
       [ 198,  474]])
plot_confusion_matrix(cnf_matrix, ['gamma', 'hadron'])

ax = plt.subplot(122)
ax.set_title("Cover Type")
matplotlib.rcParams.update({'font.size': 4})
# cnf_matrix= np.load("npData/20190209-145502-COVER-SVM-cnf_matrix.npy")
cnf_matrix = np.array([[167097,  28318,     31,      0,   8111,    863,   5800],
       [ 83335, 133041,   4396,     38,  47895,  12339,    637],
       [     0,    342,  21901,   1239,   1536,   9116,      0],
       [     0,      0,     66,   1020,      1,     40,      0],
       [    56,    511,    102,      0,   7062,    141,      1],
       [     7,    209,   1630,    264,    555,  13082,      0],
       [  4690,     41,     45,      0,     31,      0,  14083]])
plot_confusion_matrix(cnf_matrix, ['SF', 'LP', 'PP', 'CW', 'A', 'DF', 'K'])

plt.savefig("../tex/images/NN-cnf.pdf")
plt.show()