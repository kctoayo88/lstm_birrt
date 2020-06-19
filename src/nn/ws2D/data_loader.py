import torch
import torch.utils.data as data
import os
import numpy as np
import random
from torch.autograd import Variable
import torch.nn as nn

# Environment Encoder
dataPath = './dataset/s2d'
modelPath = './src/nn/ws2D/AE/ae_model/'

class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    self.encoder = nn.Sequential(nn.Linear(2800, 512), nn.PReLU(), nn.Linear(512, 256), nn.PReLU(), nn.Linear(256, 128), nn.PReLU(), nn.Linear(128, 28))
      
  def forward(self, x):
    x = self.encoder(x)
    return x

#N=number of environments; NP=Number of Paths
def load_dataset(N = 100, NP = 4000):
  Q = Encoder()
  Q.load_state_dict(torch.load(modelPath + '/cae_encoder.pkl'))
  if torch.cuda.is_available():
    Q.cuda()

  obs_rep = np.zeros((N, 28), dtype = np.float32)
  for i in range(0, N):
    #load obstacle point cloud
    temp = np.fromfile(dataPath + '/obs_cloud/obc' + str(i) + '.dat')
    temp = temp.reshape(len(temp) // 2, 2)
    obstacles = np.zeros((1, 2800), dtype = np.float32)
    obstacles[0] = temp.flatten()
    inp = torch.from_numpy(obstacles)
    inp = Variable(inp).cuda()
    output = Q(inp)
    output = output.data.cpu()
    obs_rep[i] = output.numpy()

  ## calculating length of the longest trajectory
  max_length = 0
  path_lengths = np.zeros((N, NP), dtype = np.int8)
  for i in range(0, N):
    for j in range(0, NP):
      fname = dataPath + '/e' + str(i) + '/path' + str(j) + '.dat'
      if os.path.isfile(fname):
        path = np.fromfile(fname)
        path = path.reshape(len(path) // 2, 2)
        path_lengths[i][j] = len(path)
        if len(path) > max_length:
          max_length = len(path)

  ## padded paths
  paths = np.zeros((N, NP, max_length, 2), dtype = np.float32)

  for i in range(0, N):
    for j in range(0, NP):
      fname = dataPath + '/e' + str(i) + '/path' + str(j) + '.dat'
      if os.path.isfile(fname):
        path = np.fromfile(fname)
        path = path.reshape(len(path) // 2, 2)
        for k in range(0, len(path)):
          paths[i][j][k] = path[k]

  dataset = []
  targets = []
  for i in range(0, N):
    for j in range(0, NP):
      if path_lengths[i][j] > 0:
        for m in range(0, path_lengths[i][j]-1):
          data = np.zeros(32, dtype=np.float32)
          for k in range(0, 28):
            data[k] = obs_rep[i][k]
          data[28] = paths[i][j][m][0]
          data[29] = paths[i][j][m][1]
          data[30] = paths[i][j][path_lengths[i][j]-1][0]
          data[31] = paths[i][j][path_lengths[i][j]-1][1]

          targets.append(paths[i][j][m+1])
          dataset.append(data)

  data = list(zip(dataset, targets))
  random.shuffle(data)
  dataset, targets=list(zip(*data))

  return  np.asarray(dataset), np.asarray(targets)