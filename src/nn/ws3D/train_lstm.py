import argparse
import torch
import torch.nn as nn
import numpy as np
import time
import os
import csv
from data_loader import load_dataset 
from model import LSTM
from torch.autograd import Variable

def to_var(x, volatile = False):
  if torch.cuda.is_available():
    x = x.cuda()
  return Variable(x, volatile = volatile)

def get_input(i, data, targets, bs):
  if i + bs < len(data):
    bi = data[i: i + bs]
    bt = targets[i: i + bs]	
  else:
    bi = data[i:]
    bt = targets[i:]
  return torch.from_numpy(bi), torch.from_numpy(bt)

def main(args):
  # Create model directory
  if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)
    
  # Build the data loader
  dataset, targets = load_dataset()
  print('\nThe data are loaded')
  
  # Build the models
  lstm = LSTM(args.input_size, args.output_size)
  print('The model is build')
  print(lstm)
    
  if torch.cuda.is_available():
    lstm.cuda()

  # Loss and Optimizer
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(lstm.parameters(), lr = args.learning_rate) 

  # Train the Models
  toatal_time = 0
  sm = 50 # start saving models after 100 epochs

  for epoch in range(args.num_epochs):
    print ('\nepoch ' + str(epoch) + ':')
    avg_loss = 0
    start = time.time()

    for i in range (0, len(dataset), args.batch_size):
      lstm.zero_grad()
      bi, bt = get_input(i, dataset, targets, args.batch_size)
      bi = bi.view(-1, 1, 66)
      bi = to_var(bi)
      bt = to_var(bt)
      bo = lstm(bi)
      loss = criterion(bo, bt)
      avg_loss = avg_loss + loss.item()
      loss.backward()
      optimizer.step()

    epoch_avg_loss = avg_loss / (len(dataset) / args.batch_size)
    print ('--average loss:', epoch_avg_loss)

    end = time.time()
    epoch_time = end - start
    toatal_time = toatal_time + epoch_time
    print('time of per epoch:', epoch_time)
    
    # save the data into csv
    data = [epoch_avg_loss]
    with open(args.model_path + 'lstm_loss.csv', 'a+') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)

    if epoch == sm:
      model_path = 'lstm_' + str(sm) + '.pkl'
      torch.save(lstm.state_dict(), os.path.join(args.model_path, model_path))
      sm = sm + args.save_step 

  model_path = 'lstm_final.pkl'
  torch.save(lstm.state_dict(), os.path.join(args.model_path, model_path))

if __name__ == '__main__':
  # Parameters
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_path', type=str, default='./src/nn/ws3D/nn_model/lstm_t/', help='path for saving trained models')
  parser.add_argument('--save_step', type=int, default=50, help='step size for saving trained models')
  parser.add_argument('--input_size', type=int, default=66, help='dimension of the input vector')
  parser.add_argument('--output_size', type=int, default=3, help='dimension of the input vector')
  parser.add_argument('--batch_size', type=int, default=2048)
  parser.add_argument('--num_epochs', type=int, default=500)
  parser.add_argument('--learning_rate', type=float, default=1e-4)
  args = parser.parse_args()
  print(args)
  main(args)