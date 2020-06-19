import numpy as np

def load_dataset(N = 30000):
  obstacles = np.zeros((N, 6000), dtype = np.float32)
  for i in range(0, N):
    temp = np.fromfile('./dataset/c3d/obs_cloud/obc' + str(i) + '.dat')
    obstacles[i] = temp
  return 	obstacles

if __name__ == '__main__':
  a = load_dataset()
  print(a.shape)