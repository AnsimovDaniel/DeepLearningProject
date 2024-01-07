from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch

import random
import numpy as np
import pandas as pd
from scipy.io.wavfile import read as wavread
import matplotlib.pyplot as plt
import soundfile as sf

MIC_NUM = 15
df = pd.read_csv('stats.csv')
files = df.iloc[:, 0].values
labels = df.iloc[:, -3:].values

files = [files[MIC_NUM*i : MIC_NUM*i + MIC_NUM] for i in range(int(len(files) // MIC_NUM))]
labels = np.array([labels[MIC_NUM*i] for i in range(int (len(labels) // MIC_NUM))]).astype(np.float32)

nexamples = len(labels)
instance, _ = sf.read(files[0][0])

nlen = instance.shape[0]
_nlen = int(nlen/8)

mat = np.zeros((nexamples, MIC_NUM, _nlen))

for i in range(nexamples):
    for j in range(MIC_NUM):
        instance, _ = sf.read(files[i][j])
        instance = instance[::8]
        instance = instance[:_nlen]
        if len(instance) < _nlen:
            mat[i,j, :len(instance)] = instance.astype(np.float32)
        else:
            mat[i, j, :] = instance

matReshaped = mat.reshape(mat.shape[0], -1)
print(mat.shape)
np.savetxt('mat.csv', matReshaped)
