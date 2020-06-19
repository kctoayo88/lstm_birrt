import numpy as np
import torch
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn as nn

# Environment Encoder
class Encoder_2D(nn.Module):
    def __init__(self):
        super(Encoder_2D, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(2800, 512), nn.PReLU(), nn.Linear(512, 256), nn.PReLU(), nn.Linear(256, 128), nn.PReLU(), nn.Linear(128, 28))

    def forward(self, x):
        x = self.encoder(x)
        return x

class Encoder_3D(nn.Module):
    def __init__(self):
        super(Encoder_3D, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(6000, 768), nn.PReLU(), nn.Linear(768, 512), nn.PReLU(), nn.Linear(512, 256), nn.PReLU(), nn.Linear(256, 60))

    def forward(self, x):
        x = self.encoder(x)
        return x

class LoadDataset_2D(object):
    def __init__(self):
        self.obs_data_path = './dataset/s2d/obs_cloud/'
        self.encoder_model_path = './src/nn/ws2D/AE/ae_model/cae_encoder.pkl'
        self.index = 0

    def load_obs_data(self, obs_index):
        self.index = obs_index

        # load raw obs data
        temp = np.fromfile(self.obs_data_path + 'obc' + str(obs_index) + '.dat')
        raw_obs = temp.reshape(len(temp)//2, 2)
        flatten_obs = raw_obs.flatten()

        vertices_obs = self.get_obs_vertices(flatten_obs)

        # load encoder model
        Q = Encoder_2D()
        Q.load_state_dict(torch.load(self.encoder_model_path))
        if torch.cuda.is_available():
            Q.cuda()

        # encode the orginal obstacles data to 28 dimensions
        inp = torch.from_numpy(flatten_obs).float()
        inp = Variable(inp).cuda()
        output = Q(inp)
        output = output.data.cpu()
        encoded_obs = np.zeros((1, 28), dtype = np.float32)
        encoded_obs = output.numpy()

        return raw_obs, vertices_obs, encoded_obs

    def get_obs_vertices(self, flatten_obs):
        # compute the vertices for each obstacle to create a obstacles list
        obs_data = flatten_obs.reshape(7, 200, 2)
        obs_vertices_list = []
        for i in range(len(obs_data)):
            trans_obs = np.transpose(obs_data[i])
            x_max = max(trans_obs[0])
            x_min = min(trans_obs[0])
            y_max = max(trans_obs[1])
            y_min = min(trans_obs[1])
            lower_left = [x_min, y_min]
            upper_right = [x_max, y_max]
            lower_left.extend(upper_right)
            obs_vertices_list.append(lower_left)
        obs_vertices_list = np.array(obs_vertices_list)

        return obs_vertices_list

    def load_init_goal(self, env_index):
        # load the init and goal position in dataset
        env_data_path = './dataset/s2d/e' + str(self.index) + '/path' + str(env_index) + '.dat'
        temp = np.fromfile(env_data_path)
        path = temp.reshape(len(temp)//2, 2)
        x_init = (path[0][0], path[0][1])
        x_goal = (path[-1][0], path[-1][1])
    
        return x_init, x_goal

class LoadDataset_3D(object):
    def __init__(self):
        self.obs_data_path = './dataset/c3d/obs_cloud/'
        self.encoder_model_path = './src/nn/ws3D/AE/ae_model/cae_encoder.pkl'
        self.index = 0

    def load_obs_data(self, obs_index):
        self.index = obs_index

        # load raw obs data
        temp = np.fromfile(self.obs_data_path + 'obc' + str(obs_index) + '.dat')
        raw_obs = temp.reshape(len(temp)//3, 3)
        flatten_obs = raw_obs.flatten()

        vertices_obs = self.get_obs_vertices(flatten_obs)

        # load encoder model
        Q = Encoder_3D()
        Q.load_state_dict(torch.load(self.encoder_model_path))
        if torch.cuda.is_available():
            Q.cuda()

        # encode the orginal obstacles data to 60 dimensions
        inp = torch.from_numpy(flatten_obs).float()
        inp = Variable(inp).cuda()
        output = Q(inp)
        output = output.data.cpu()
        encoded_obs = np.zeros((1, 60), dtype = np.float32)
        encoded_obs = output.numpy()

        return raw_obs, vertices_obs, encoded_obs

    def get_obs_vertices(self, flatten_obs):
        # compute the vertices for each obstacle to create a obstacles list
        obs_data = flatten_obs.reshape(-1, 200, 3)
        
        obs_vertices_list = []
        for i in range(len(obs_data)):
            trans_obs = np.transpose(obs_data[i])
            x_max = max(trans_obs[0])
            x_min = min(trans_obs[0])
            y_max = max(trans_obs[1])
            y_min = min(trans_obs[1])
            z_max = max(trans_obs[2])
            z_min = min(trans_obs[2])
            lower_left = [x_min, y_min, z_min]
            upper_right = [x_max, y_max, z_max]
            lower_left.extend(upper_right)
            obs_vertices_list.append(lower_left)
        obs_vertices_list = np.array(obs_vertices_list)

        return obs_vertices_list

    def load_init_goal(self, env_index):
        # load the init and goal position in dataset
        env_data_path = './dataset/c3d/e' + str(self.index) + '/path' + str(env_index) + '.dat'
        temp = np.fromfile(env_data_path)
        path = temp.reshape(len(temp) // 3, 3)
        x_init = (path[0][0], path[0][1], path[0][2])
        x_goal = (path[-1][0], path[-1][1], path[-1][2])
    
        return x_init, x_goal

if __name__ == '__main__':
    loader = LoadDataset_3D()
    raw_obs, vertices_obs, encoded_obs = loader.load_obs_data(1)