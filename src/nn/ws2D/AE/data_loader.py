import numpy as np

def load_dataset(N = 30000):
  obstacles = np.zeros((N, 2800), dtype = np.float32)
  for i in range(0, N):
    temp = np.fromfile('./dataset/s2d/obs_cloud/obc' + str(i) + '.dat')
    temp = temp.reshape(len(temp)//2, 2)
    obstacles[i] = temp.flatten()
  return 	obstacles	

if __name__ == '__main__':
  a = load_dataset()
  print(a.shape)