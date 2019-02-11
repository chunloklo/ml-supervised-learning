import numpy as np
import matplotlib.pyplot as plt

celldata = np.array([[0.69801825, 0.79550962, 0.71385474, 0.19585091, 0.15199869,
        0.42923229, 0.46764034],
        [0.68066101, 0.61186735, 0.75367625, 0.94889267, 0.88463112,
        0.79930295, 0.91433243],
[0.68923037, 0.69170716, 0.73322522, 0.32468668, 0.25942293,
        0.55852958, 0.61879472],
[209680, 281141,  33594,    587,   7333,  15207,  18350]])

print(celldata)
plt.table(cellText=celldata)
plt.show()