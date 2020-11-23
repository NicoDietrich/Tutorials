import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import csv

fieldnames = ['index', 'h', 'F_xph', 'F_xmh', 'central_dif', 'adj_grad']
with open('results.csv', 'r') as f:
    reader = csv.DictReader(f, fieldnames)
    data = [d for d in reader]

h = data[1]['h']
adjoint_grads = np.array([float(d['adj_grad']) for d in data[1:]])
central_dif = np.array([float(d['central_dif']) for d in data[1:]])

difs = np.abs(adjoint_grads - central_dif)

fig, ax = plt.subplots()
ax.plot(difs, ls='', marker='x', color='blue')
ax.set_title(f'|adj_grad - cent_dif_grad|, h={h}')
ax.set_xlabel('index degree of freedome')
ax.grid(True)

with PdfPages("finite_difference_verification.pdf") as pdffile:
    pdffile.savefig(fig)
