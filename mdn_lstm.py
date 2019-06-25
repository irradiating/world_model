# coding: utf8

import numpy as np
import os, sys
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

device = torch.device("cpu")
# device = torch.device("cuda")

class LSTM(nn.Module):
    def __init__(self, n_gaussians=8, vector_size=4608, hidden_size=512, num_layers=2):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.n_gaussians = n_gaussians
        self.vector_size = vector_size
        self.reset_states()
        self.mse_loss = nn.MSELoss()

        ###########
        ## Encoding
        # self.fc1 = nn.Linear(in_features=vector_size+1, out_features=input_size)
        # batch_first=True - input shape ( batch, sequence, input_size )
        self.lstm = nn.LSTM(input_size=vector_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc_output = nn.Linear(in_features=hidden_size, out_features=vector_size)


        #########
        ## Output
        self.z_pi = nn.Linear(in_features=vector_size, out_features=n_gaussians * vector_size)
        self.z_sigma = nn.Linear(in_features=vector_size, out_features=n_gaussians * vector_size)
        self.z_mu = nn.Linear(in_features=vector_size, out_features=n_gaussians * vector_size)


    def reset_states(self):
        # num_layers, batch_size, hidden_size
        self.c = torch.zeros(self.num_layers, 1, self.hidden_size).to(device) # [2, 1, 512]
        self.h = torch.zeros(self.num_layers, 1, self.hidden_size).to(device) # [2, 1, 512]
        self.c.detach_()
        self.h.detach_()



    def forward(self, x):
        self.lstm.flatten_parameters()
        sequence_len = x.size(1)

        # print(x.size()) # [1, 3999, 1024]

        z, (self.h, self.c) = self.lstm(x, (self.h, self.c))
        # print(z.size()) # [1, 49, 512]
        self.h.detach_() # [2, 1, 512]
        self.c.detach_() # [2, 1, 512]
        z = F.relu( self.fc_output(z) ) # [1, 3999, 4608]


        # pi = self.z_pi(z)  # [1, 3999, 23040]
        pi = self.z_pi(z).view( -1, sequence_len, self.n_gaussians, self.vector_size ) # (batch_size, sequence_length, n_gaussians, latent_dimension)
        # [1, 3999, 8, 4608]
        pi = torch.softmax( pi, dim=2 ) # [1, 3999, 8, 4608]

        sigma = torch.exp( self.z_sigma(z) ).view(-1, sequence_len, self.n_gaussians, self.vector_size) # (batch_size, sequence_length, n_gaussians, latent_dimension)
        # [1, 3999, 8, 4608]

        mu = self.z_mu(z).view( -1, sequence_len, self.n_gaussians, self.vector_size ) # (batch_size, sequence_length, n_gaussians, latent_dimension)
        # [1, 3999, 8, 4608]

        return pi, sigma, mu

    def predict(self, z):
        self.lstm.flatten_parameters()

        z_t, (self.h, self.c) = self.lstm(z, (self.h, self.c))
        self.h.detach_()
        self.c.detach_()
        z_t = self.fc_output(z_t) # [1, 1, 4608]

        return z_t


    def encode(self, x):
        self.lstm.flatten_parameters()

        z, (self.h, self.c) = self.lstm(x, (self.h, self.c))
        self.h.detach_()
        self.c.detach_()
        # z = self.fc_output(x)  # [1, 1, 4608]

        # return z
        return self.h, self.c

    def mean_mu(self, x):
        self.lstm.flatten_parameters()
        sequence = x.size(1)

        x = F.relu( self.fc1(x) ) # [90, 2304]

        # print( self.h0.size() ) # [2, 90, 512]
        # print( self.c0.size() ) # [2, 90, 512]

        z, (self.h, self.c) = self.lstm(x, (self.h, self.c)) # [1, 1, 512] [2, 1, 512] [2, 1, 512]
        self.h.detach_()
        self.c.detach_()

        mu = self.z_mu(z).view( -1, sequence, self.n_gaussians, self.vector_size ) # (batch_size, sequence_length, n_gaussians, latent_dimension)
        # [1, 1000, 5, 4608]
        mean_mu = torch.mean(mu, dim=2) # [1, 1, 4608]

        return mean_mu


    def mdn_loss_function(self, out_pi, out_sigma, out_mu, y):
        # vector_size = out_sigma.size(3)
        sequence_len = y.size(1)

        y = y.view(-1, sequence_len, 1, self.vector_size) # [1, 3999, 1, 4608]

        result = Normal(loc=out_mu, scale=out_sigma)
        result = torch.exp( result.log_prob(y) ) # [1, 3999, 8, 4608]
        result = torch.sum( result * out_pi, dim=2 ) # [1, 3999, 4608]
        result = -torch.log( 0.0001 + result ) # [1, 3999, 4608]
        return torch.mean( result )

    def mse_loss_function(self, actual, prediction):

        loss = self.mse_loss(prediction, actual)

        return loss




def train_on_saved_batch():

    lstm_mdn_filename = "weights/rnn_mdn_SonicAndKnuckles.pkl"
    batch_dir = "lstm_batches"
    batch = None
    batch_size = 12000
    sequence_len = batch_size - 1 # 3999
    counter = 1

    lstm_mdn = LSTM(sequence_len=sequence_len)
    lstm_mdn_optimizer = optim.Adam(lstm_mdn.parameters(), lr=0.00025)
    if os.path.exists(lstm_mdn_filename):
        print("loading lstm mdn weights")
        lstm_mdn.load_state_dict( torch.load(lstm_mdn_filename) ) # map_location="cpu"


    for filename_short in os.listdir(batch_dir):
        filename_full = os.path.join(batch_dir, filename_short) # lstm_batches/lstm_batch_20190605180849.pkl


        with open(filename_full, 'rb') as f:
            z = pickle.load(f) # [1000, 4608]

        if batch is None:
            batch = z
        elif batch.size(0) < batch_size:
            batch = torch.cat( (batch, z), dim=0 )

        if batch.size(0) == batch_size:

            for epoch in range(20):

                # +1 to represent future step
                # time vector:     [t,t,t,t,t]
                # latent vector: [l,l,l,l,l,l]


                # trunkate 1 element from right, to produce 3999 size time vectors
                # time vector:     [t,t,t,t,t]
                # latent vector: [l,l,l,l,l]
                batch_crop_right = batch[:sequence_len,:] # [3999, 4608]
                batch_crop_right = batch_crop_right.unsqueeze(0) # [1, 3999, 4608]

                pi, sigma, mu = lstm_mdn(batch_crop_right) # [1, 3999, 5, 4608]


                # trunkate 1 element from left
                # time vector:     [t,t,t,t,t]
                # latent vector:   [l,l,l,l,l]
                # 1000 elements buffer: predictions and actual line up
                batch_crop_left = batch[1:,:] # [3999, 4608]
                # batch_crop_left = batch_crop_left.unsqueeze(0) # [1, 3999, 4608]


                lstm_mdn_loss = lstm_mdn.mdn_loss_function(pi, sigma, mu, batch_crop_left)

                print( "training lstm mdn", lstm_mdn_loss )

                lstm_mdn_optimizer.zero_grad()

                lstm_mdn_loss.backward()

                lstm_mdn_optimizer.step()



            # reset batch
            batch = None
            # reset hidden state and cell state
            lstm_mdn.reset_states()


            counter += 1

        if counter % 8 == 0:
            print("save lstm mdn weights")
            torch.save(lstm_mdn.state_dict(), lstm_mdn_filename)
            counter = 1

    print()



if __name__ == "__main__":

    # train_on_saved_batch()

    print()