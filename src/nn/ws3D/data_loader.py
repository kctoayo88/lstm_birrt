import torch
import torch.utils.data as data
import os
import numpy as np
import random
from torch.autograd import Variable
import torch.nn as nn

# Environment Encoder
dataPath = './dataset/c3d'
modelPath = './src/nn/ws3D/AE/ae_model'

class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    self.encoder = nn.Sequential(nn.Linear(6000, 768), nn.PReLU(), nn.Linear(768, 512), nn.PReLU(), nn.Linear(512, 256), nn.PReLU(), nn.Linear(256, 60))
      
  def forward(self, x):
    x = self.encoder(x)
    return x

#N=number of environments; NP=Number of Paths
def load_dataset(N = 100, NP = 4000):
  Q = Encoder()
  Q.load_state_dict(torch.load(modelPath + '/cae_encoder.pkl'))
  if torch.cuda.is_available():
    Q.cuda()

  obs_rep = np.zeros((N, 60), dtype = np.float32)
  for i in range(0, N):
    #load obstacle point cloud
    temp = np.fromfile(dataPath + '/obs_cloud/obc' + str(i) + '.dat')
    obstacles = np.zeros((1, 6000), dtype = np.float32)
    obstacles[0] = temp
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
        path = path.reshape(len(path) // 3, 3)
        path_lengths[i][j] = len(path)
        if len(path) > max_length:
          max_length = len(path)

  ## padded paths
  paths = np.zeros((N, NP, max_length, 3), dtype = np.float32)

  for i in range(0, N):
    for j in range(0, NP):
      fname = dataPath + '/e' + str(i) + '/path' + str(j) + '.dat'
      if os.path.isfile(fname):
        path = np.fromfile(fname)
        path = path.reshape(len(path) // 3, 3)
        for k in range(0, len(path)):
          paths[i][j][k] = path[k]

  dataset = []
  targets = []
  for i in range(0, N):
    for j in range(0, NP):
      if path_lengths[i][j] > 0:
        for m in range(0, path_lengths[i][j]-1):
          data = np.zeros(66, dtype=np.float32)
          for k in range(0, 60):
            data[k] = obs_rep[i][k]
          data[60] = paths[i][j][m][0]
          data[61] = paths[i][j][m][1]
          data[62] = paths[i][j][m][2]
          data[63] = paths[i][j][path_lengths[i][j]-1][0]
          data[64] = paths[i][j][path_lengths[i][j]-1][1]
          data[65] = paths[i][j][path_lengths[i][j]-1][2]

          targets.append(paths[i][j][m+1])
          dataset.append(data)

  data = list(zip(dataset, targets))
  random.shuffle(data)
  dataset, targets = list(zip(*data))

  return  np.asarray(dataset), np.asarray(targets)

if __name__ == '__main__':
  dataset, targets = load_dataset()
  print(dataset.shape)
  print(targets.shape)