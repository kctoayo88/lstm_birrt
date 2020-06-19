import argparse
import os
import torch
import csv
import numpy as np
from torch import nn
from torch.autograd import Variable
from data_loader import load_dataset

class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    self.encoder = nn.Sequential(nn.Linear(6000, 768), nn.PReLU(), nn.Linear(768, 512), nn.PReLU(), nn.Linear(512, 256), nn.PReLU(), nn.Linear(256, 60))
      
  def forward(self, x):
    x = self.encoder(x)
    return x

class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()
    self.decoder = nn.Sequential(nn.Linear(60, 256), nn.PReLU(), nn.Linear(256, 512), nn.PReLU(), nn.Linear(512, 768), nn.PReLU(), nn.Linear(768, 6000))
  def forward(self, x):
    x = self.decoder(x)
    return x

mse_loss = nn.MSELoss()
lam = 1e-3
def loss_function(W, x, recons_x, h):
  mse = mse_loss(recons_x, x)
  contractive_loss = torch.sum(Variable(W)**2, dim=1).sum().mul_(lam)
  return mse + contractive_loss

def main(args):	
  if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

  obs = load_dataset()
  print(obs.shape)

  encoder = Encoder()
  decoder = Decoder()
  if torch.cuda.is_available():
    encoder.cuda()
    decoder.cuda()

  params = list(encoder.parameters()) + list(decoder.parameters())
  optimizer = torch.optim.Adam(params, lr=0.0001)
  for epoch in range(args.num_epochs):
    print ('epoch ' + str(epoch) + ':')
    avg_loss = 0
    for i in range(0, len(obs), args.batch_size):
      decoder.zero_grad()
      encoder.zero_grad()
      if i + args.batch_size < len(obs):
        inp = obs[i:i + args.batch_size]
      else:
        inp = obs[i:]
      inp = torch.from_numpy(inp)
      inp = Variable(inp).cuda()
      # ===================forward=====================
      h = encoder(inp)
      output = decoder(h)
      W = encoder.state_dict()['encoder.6.weight'] # regularize or contracting last layer of encoder. Print keys to displace the layers name. 
      loss = loss_function(W, inp, output, h)
      avg_loss = avg_loss + loss.item()
      # ===================backward====================
      loss.backward()
      optimizer.step()
    
    epoch_avg_loss = avg_loss / (len(obs) / args.batch_size)
    print ('--average loss:', epoch_avg_loss)
    print()

    # save the data into csv
    data = [epoch_avg_loss]
    with open(args.model_path + 'CAE_loss_t.csv', 'a+') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)

  avg_loss = 0
  for i in range(len(obs) - 5000, len(obs), args.batch_size):
    inp = obs[i:i + args.batch_size]
    inp = torch.from_numpy(inp)
    inp = Variable(inp).cuda()
    # ===================forward=====================
    output = encoder(inp)
    output = decoder(output)
    loss = mse_loss(output, inp)
    avg_loss = avg_loss + loss.item()
    # ===================backward====================
  print ('--Validation average loss:', avg_loss / (5000 / args.batch_size))

  torch.save(encoder.state_dict(), os.path.join(args.model_path, 'cae_encoder_t.pkl'))
  torch.save(decoder.state_dict(), os.path.join(args.model_path, 'cae_decoder_t.pkl'))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_path', type=str, default='./src/nn/ws3D/AE/ae_model/', help='path for saving trained models')

  # Model parameters
  parser.add_argument('--num_epochs', type=int, default=5000)
  parser.add_argument('--batch_size', type=int, default=256)
  args = parser.parse_args()
  print(args)
  main(args)
