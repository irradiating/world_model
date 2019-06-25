# coding: utf8

import numpy as np
import os, sys
import cv2
import retro
import cma

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

from conv_b_vae import ConvVAE
from mdn_lstm import LSTM
from es import PEPG, OpenES, CMAES
from helper import dumyshape, dumyshape_gray_edges, reward_calculation, dumyshape_gray


class Controller(nn.Module):
    def __init__(self, input_size=8896, action_size=12): # 1112
        super(Controller, self).__init__()
        self.action_size = action_size

        self.fc1 = nn.Linear(in_features=input_size, out_features=action_size)

        self.ones = torch.ones(self.action_size)
        self.zeros = torch.zeros(self.action_size)


    def forward(self, x):
        # print(env.action_space) # MultiBinary(12)
        # print(env.action_space.sample()) # [1 1 1 0 1 0 1 0 0 1 1 1]

        x = torch.sigmoid( self.fc1(x) )

        x = torch.where(x >= 0.5, self.ones, self.zeros) # [1 1 1 0 1 0 1 0 0 1 1 1]

        return x


    def set_weights(self, set):
        # print(set.shape) # (79884,)

        weights = set[:-12] # (79872,)
        weights = np.reshape(weights, (12,6656)) # (12, 6656)
        weights = torch.FloatTensor(weights) # [12, 6656]

        bias = set[-12:] # (12,)
        bias = torch.FloatTensor(bias) # [12]


        # l1w = torch.FloatTensor( l1w ) # FloatTensor of size 16x2
        # l1b = torch.FloatTensor( l1b ) # FloatTensor of size 16
        # l2w = torch.FloatTensor( l2w ) # FloatTensor of size 1x16
        # l2b = torch.FloatTensor( l2b ) # FloatTensor of size 1


        self.fc1.weight.data.copy_(weights.view_as(self.fc1.weight.data))
        self.fc1.bias.data.copy_(bias.view_as(self.fc1.bias.data))
        # self.fc2.weight.data.copy_(l2w.view_as(self.fc2.weight.data))
        # self.fc2.bias.data.copy_(l2b.view_as(self.fc2.bias.data))

        # print( self.fc1.weight.data )

class Controller_RNN(nn.Module):
    def __init__(self, input_size=1024, action_size=12, hidden_size=512, batch_size=1):
        super(Controller_RNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = 1
        self.batch_size = batch_size
        self.reset_states()

        # batch_first=True - means input must be shaped
        # ( batch, sequence, input_size )
        self.rnn = nn.RNN(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, nonlinearity="relu")

        self.dence = nn.Linear(in_features=hidden_size, out_features=action_size)


    def reset_states(self):
        # num_layers, batch_size, hidden_size
        self.h = torch.zeros(self.num_layers, 1, self.hidden_size)
        self.h.detach_()


    def forward(self, x):

        x, self.h = self.rnn(x, self.h)
        self.h.detach_()

        x = self.dence(x)
        x = x[:,-1,:] # only last sequence [1, 12]
        x = torch.sigmoid(x)

        dist = Bernoulli(probs=x)

        return dist.sample()

    def set_weights(self, set):
        # print(set.shape) # (793612,)

        rnn_state_dict = {
        "weight_ih_l0" : torch.FloatTensor( np.reshape( set[:524288], (512,1024) ) ),  # [512, 1024]
        "weight_hh_l0" : torch.FloatTensor( np.reshape( set[524288:786432], (512,512) ) ),  # [512, 512]
        "bias_ih_l0" : torch.FloatTensor( set[786432:786944] ), # 512
        "bias_hh_l0" : torch.FloatTensor( set[786944:787456] )} # 512

        self.rnn.load_state_dict( rnn_state_dict )


        dence_weight = torch.FloatTensor( np.reshape( set[787456:793600], (12,512) ) ) # [12, 512]
        dence_bias = torch.FloatTensor( set[793600:] ) # 12

        self.dence.weight.data.copy_(dence_weight.view_as(self.dence.weight.data))
        self.dence.bias.data.copy_(dence_bias.view_as(self.dence.bias.data))
        # self.fc2.weight.data.copy_(l2w.view_as(self.fc2.weight.data))
        # self.fc2.bias.data.copy_(l2b.view_as(self.fc2.bias.data))

        # state_dict = controller.state_dict() # 6
        # rnn_weight_ih_l0 = state_dict["rnn.weight_ih_l0"] # [512, 1024]  524 288
        # rnn_weight_hh_l0 = state_dict["rnn.weight_hh_l0"] # [512, 512]   262 144
        # rnn_bias_ih_l0 = state_dict["rnn.bias_ih_l0"] # [512]        512
        # rnn_bias_hh_l0 = state_dict["rnn.bias_hh_l0"] # [512]        512
        # dence_weight = state_dict["dence.weight"] # [12, 512]    6144
        # dence_bias = state_dict["dence.bias"] # [12]         12
        # 793 612

        # flattened_controller_weights = flattened_controller_weights.data.numpy()
        # rnn_weight_ih_l0 = flattened_controller_weights[:524288]
        # rnn_weight_ih_l0 = torch.FloatTensor(rnn_weight_ih_l0).view(512, 1024)
        # rnn_weight_hh_l0 = flattened_controller_weights[524288:786432]
        # rnn_weight_hh_l0 = torch.FloatTensor(rnn_weight_hh_l0).view(512,512)

    def set_dence_weights(self, set):
        # print(set.shape)

        dence_weight = torch.FloatTensor( np.reshape( set[:-12], (12,512) ) ) # [12, 512]
        dence_bias = torch.FloatTensor( set[-12:] ) # 12

        self.dence.weight.data.copy_(dence_weight.view_as(self.dence.weight.data))
        self.dence.bias.data.copy_(dence_bias.view_as(self.dence.bias.data))
        # self.fc2.weight.data.copy_(l2w.view_as(self.fc2.weight.data))
        # self.fc2.bias.data.copy_(l2b.view_as(self.fc2.bias.data))

        # state_dict = controller.state_dict() # 6
        # rnn_weight_ih_l0 = state_dict["rnn.weight_ih_l0"] # [512, 1024]  524 288
        # rnn_weight_hh_l0 = state_dict["rnn.weight_hh_l0"] # [512, 512]   262 144
        # rnn_bias_ih_l0 = state_dict["rnn.bias_ih_l0"] # [512]        512
        # rnn_bias_hh_l0 = state_dict["rnn.bias_hh_l0"] # [512]        512
        # dence_weight = state_dict["dence.weight"] # [12, 512]    6144
        # dence_bias = state_dict["dence.bias"] # [12]         12
        # 793 612

        # flattened_controller_weights = flattened_controller_weights.data.numpy()
        # rnn_weight_ih_l0 = flattened_controller_weights[:524288]
        # rnn_weight_ih_l0 = torch.FloatTensor(rnn_weight_ih_l0).view(512, 1024)
        # rnn_weight_hh_l0 = flattened_controller_weights[524288:786432]
        # rnn_weight_hh_l0 = torch.FloatTensor(rnn_weight_hh_l0).view(512,512)




def test_controller():
    # env_name = "SonicTheHedgehog-Genesis" # None
    env_name = "SonicTheHedgehog2-Genesis"
    # env_name = "SonicAndKnuckles-Genesis"
    # env_name = "SonicTheHedgehog3-Genesis"
    # env_name = "SonicAndKnuckles3-Genesis"

    env = retro.make(env_name)
    # print(env.observation_space) # Box(224, 320, 3)
    # print(env.action_space) # MultiBinary(12)
    # print(env.action_space.sample()) # [1 1 1 0 1 0 1 0 0 1 1 1]


    # conv_vae_filename = "weights/conv_vae_SonicAndKnuckles.pkl" # 3, 4608
    # lstm_mdn_filename = "weights/lstm_mdn_SonicAndKnuckles.pkl" # 4608
    # controller_filename = "weights/controller_6656_12.pkl"

    conv_vae_filename = "weights/conv_vae_gray.pkl" # 1, 1024
    lstm_mdn_filename = "weights/lstm_mdn_gray.pkl" # 1024
    controller_filename = "weights/controller_rnn_1024_12.pkl"
    evaluator_filename = "weights/evaluator_openes_weights_20_0.499982.npz"

    population_size = 256
    generations = 5000

    # only forward pass
    conv_vae = ConvVAE((1,128,128), 1024)
    if os.path.exists(conv_vae_filename):
        print("loading conv vae weights")
        conv_vae.load_state_dict(torch.load(conv_vae_filename))

    # only forward pass
    lstm_mdn = LSTM(vector_size=1024)
    if os.path.exists(lstm_mdn_filename):
        print("loading lstm mdn weights")
        lstm_mdn.load_state_dict( torch.load(lstm_mdn_filename) )

    controller = Controller_RNN(input_size=1024, batch_size=2) # 6656
    if os.path.exists(controller_filename):
        print("loading controller weights")
        controller.load_state_dict( torch.load(controller_filename) )

    # evaluator restore
    if os.path.exists(evaluator_filename):
        print("loading evaluator data")
        data = np.load(evaluator_filename)
        weights = data["weights"]
        print("inserting weights into controller")
        controller.set_weights(weights)


    img = env.reset()

    while True:

        img = dumyshape_gray(img)
        img = torch.FloatTensor(img)
        img = img.unsqueeze(0)

        z = conv_vae(img, encode=True) # [1, 1024]
        z = z.unsqueeze(0).detach() # [1, 1, 1024]

        z_t = lstm_mdn.predict(z) # [1, 1, 1024]
        z_t = z_t.detach()

        input = torch.cat( (z, z_t), dim=1 ) # [1, 2, 1024]

        actions = controller(input)
        actions = actions.squeeze(0).data.numpy() # [1. 1. 0. 1. 1. 1. 0. 1. 0. 1. 0. 0.]


        img, reward, done, info = env.step(actions)

        env.render()


        if done:
            break


    controller.reset_states()





def train_controller_openes():

    # env_name = "SonicTheHedgehog-Genesis" # None
    env_name = "SonicTheHedgehog2-Genesis"
    # env_name = "SonicAndKnuckles-Genesis"
    # env_name = "SonicTheHedgehog3-Genesis"
    # env_name = "SonicAndKnuckles3-Genesis"

    env = retro.make(env_name)
    # print(env.observation_space) # Box(224, 320, 3)
    # print(env.action_space) # MultiBinary(12)
    # print(env.action_space.sample()) # [1 1 1 0 1 0 1 0 0 1 1 1]

    # conv_vae_filename = "weights/conv_vae_SonicAndKnuckles.pkl" # 3, 4608
    # lstm_mdn_filename = "weights/lstm_mdn_SonicAndKnuckles.pkl" # 4608
    # controller_filename = "weights/controller_6656_12.pkl"

    conv_vae_filename = "weights/conv_vae_gray.pkl" # 1, 1024
    lstm_mdn_filename = "weights/lstm_mdn_gray.pkl" # 1024
    controller_filename = "weights/controller_rnn_1024_12.pkl"
    evaluator_filename = "weights/evaluator_openes_weights_20_0.499982.npz"


    population_size = 256
    generations = 5000


    # only forward pass
    conv_vae = ConvVAE((1,128,128), 1024)
    if os.path.exists(conv_vae_filename):
        print("loading conv vae weights")
        conv_vae.load_state_dict(torch.load(conv_vae_filename))

    # only forward pass
    lstm_mdn = LSTM(vector_size=1024)
    if os.path.exists(lstm_mdn_filename):
        print("loading lstm mdn weights")
        lstm_mdn.load_state_dict( torch.load(lstm_mdn_filename) )

    controller = Controller_RNN(input_size=1024, batch_size=2) # 6656
    if os.path.exists(controller_filename):
        print("loading controller weights")
        controller.load_state_dict( torch.load(controller_filename) )

    # evaluator restore
    if os.path.exists(evaluator_filename):
        print("loading evaluator data")
        data = np.load(evaluator_filename)
        weights = data["weights"]
        print("inserting weights into controller")
        controller.set_weights(weights)

        evaluator = OpenES(num_params=793612, popsize=population_size, existing_weights=weights)

    else:
        print("extracting controller weights")
        state_dict = controller.state_dict() # 6
        rnn_weight_ih_l0 = state_dict["rnn.weight_ih_l0"] # [512, 1024]  524 288
        rnn_weight_hh_l0 = state_dict["rnn.weight_hh_l0"] # [512, 512]   262 144
        rnn_bias_ih_l0 = state_dict["rnn.bias_ih_l0"] # [512]        512
        rnn_bias_hh_l0 = state_dict["rnn.bias_hh_l0"] # [512]        512
        dence_weight = state_dict["dence.weight"] # [12, 512]    6144
        dence_bias = state_dict["dence.bias"] # [12]         12
        # 793 612

        rnn_weight_ih_l0 = torch.flatten(rnn_weight_ih_l0) # [524288]
        rnn_weight_hh_l0 = torch.flatten(rnn_weight_hh_l0) # [262144]
        dence_weight = torch.flatten(dence_weight) # [6144]

        flattened_controller_weights = torch.cat( (rnn_weight_ih_l0, rnn_weight_hh_l0, rnn_bias_ih_l0, rnn_bias_hh_l0, dence_weight, dence_bias), dim=0) # [793612]
        flattened_controller_weights = flattened_controller_weights.data.numpy()

        evaluator = OpenES(num_params=793612, popsize=population_size, existing_weights=flattened_controller_weights)


    for generation in range(generations):

        solutions = evaluator.ask() # (64, 793612)

        fitness = np.zeros(population_size) # 64

        for i in range(population_size):

            fitness[i] = evaluate(weights=solutions[i],
                                 conv_vae=conv_vae,
                                 lstm_mdn=lstm_mdn,
                                 controller=controller,
                                 env=env)


        evaluator.tell(fitness)

        result = evaluator.result() # first element is the best solution, second element is the best fitness

        best_fitness = result[1]
        best_weights = result[0]

        print(generation, best_fitness)


        ##############################
        ## save evaluator data weights
        print("save evaluator data weights")
        evaluator_weights_filename = "weights/evaluator_openes_weights_%s_%s.npz" %(generation, result[1])
        np.savez(evaluator_weights_filename, weights=best_weights)

        # save controller weights
        # print("save controller weights")
        # torch.save(controller.state_dict(), controller_filename)



def train_controller_cma():
    # env_name = "SonicTheHedgehog-Genesis" # None
    env_name = "SonicTheHedgehog2-Genesis"
    # env_name = "SonicAndKnuckles-Genesis"
    # env_name = "SonicTheHedgehog3-Genesis"
    # env_name = "SonicAndKnuckles3-Genesis"

    env = retro.make(env_name)
    # print(env.observation_space) # Box(224, 320, 3)
    # print(env.action_space) # MultiBinary(12)
    # print(env.action_space.sample()) # [1 1 1 0 1 0 1 0 0 1 1 1]

    # conv_vae_filename = "weights/conv_vae_SonicAndKnuckles.pkl" # 3, 4608
    # lstm_mdn_filename = "weights/lstm_mdn_SonicAndKnuckles.pkl" # 4608
    # controller_filename = "weights/controller_6656_12.pkl"

    conv_vae_filename = "weights/conv_vae_gray.pkl" # 1, 1024
    lstm_mdn_filename = "weights/lstm_mdn_gray.pkl" # 1024
    controller_filename = "weights/controller_cma_1024_12.pkl"

    evaluator_openes_filename = "weights/evaluator_openes_weights_26_0.499966.npz"
    evaluator_cma_filename = "weights/evaluator_cma_weights_0_986.33333333333.npz"


    population_size = 256
    generations = 5000


    # only forward pass
    conv_vae = ConvVAE((1,128,128), 1024)
    if os.path.exists(conv_vae_filename):
        print("loading conv vae weights")
        conv_vae.load_state_dict(torch.load(conv_vae_filename))

    # only forward pass
    lstm_mdn = LSTM(vector_size=1024)
    if os.path.exists(lstm_mdn_filename):
        print("loading lstm mdn weights")
        lstm_mdn.load_state_dict( torch.load(lstm_mdn_filename) )

    controller = Controller_RNN(input_size=1024, batch_size=2) # 6656
    if os.path.exists(controller_filename):
        print("loading controller weights")
        controller.load_state_dict( torch.load(controller_filename) )

    # evaluator openes restore
    if os.path.exists(evaluator_openes_filename):
        print("loading openes evaluator data")
        data = np.load(evaluator_openes_filename)
        weights = data["weights"]
        print("inserting weights into controller")
        controller.set_weights(weights)


    # evaluator cma
    if os.path.exists(evaluator_cma_filename):
        print("loading cma evaluator data")
        data = np.load(evaluator_cma_filename)
        mean_weights = data["mean"]
        # best_weights = data["best"]
        print("inserting dense weights into controller")
        controller.set_dence_weights(mean_weights)

        sigma_init = 0.10 # initial standard deviation
        evaluator = cma.CMAEvolutionStrategy(mean_weights, sigma_init, {"popsize": population_size})

    else:
        print("NO cma filename found, extracting dence weights from controller")
        state_dict = controller.state_dict() # 6
        dence_weight = state_dict["dence.weight"] # [12, 512]    6144
        dence_weight = dence_weight.view(-1)  # [6144]
        dence_bias = state_dict["dence.bias"] # [12]         12

        weights = torch.cat((dence_weight, dence_bias), dim=0) # [6156]
        weights = weights.data.numpy() # (6156,)

        sigma_init = 0.10 # initial standard deviation
        evaluator = cma.CMAEvolutionStrategy(weights, sigma_init, {"popsize": population_size})





    for generation in range(generations):

        solutions = evaluator.ask() # 256, 6156

        fitness = np.zeros(population_size)

        # evaluate solutions
        for index in range(population_size):

            curr_weight = solutions[index] # (6156,)

            fitness[index] = evaluate_cma(curr_weight, conv_vae, lstm_mdn, controller, env)


        # inverse fitness table
        fitness_inverted = -fitness # ЕС похоже ищет минимизатор. Инверсия чтоб самый большой фитнес стал самым малым
        fitness_inverted = fitness_inverted.tolist()

        # передать инвурсию в сма
        evaluator.tell(solutions, fitness_inverted)

        result = evaluator.result

        best_param = result[0] # best evaluated solution [0.03405598 -0.22424321 0.16289401...-0.14126145 -0.04335651 -0.08884694] (6156,)
        curr_reward = -result[1] # инверсия обратно 635.16
        mean_params = result[5] # presumably better with noise [-0.05136075 -0.04151194 -0.01733577...-0.0587192 -0.0432042 -0.0475102 ] (6156,)
        # sigma = result[6] # [0.09907158 0.09907164 0.09907168 ... 0.09907655 0.09907659 0.09907652] (6156,)

        print(generation, curr_reward)

        print("saving cma data")
        evaluator_cma_filename = "weights/evaluator_cma_weights_%s_%s.npz" %(generation, curr_reward)
        np.savez(evaluator_cma_filename, mean=mean_params, best=best_param)






def train_controller_pepg():

    # env_name = "SonicTheHedgehog-Genesis" # None
    env_name = "SonicTheHedgehog2-Genesis"
    # env_name = "SonicAndKnuckles-Genesis"
    # env_name = "SonicTheHedgehog3-Genesis"
    # env_name = "SonicAndKnuckles3-Genesis"

    env = retro.make(env_name)
    # print(env.observation_space) # Box(224, 320, 3)
    # print(env.action_space) # MultiBinary(12)
    # print(env.action_space.sample()) # [1 1 1 0 1 0 1 0 0 1 1 1]


    conv_vae_filename = "weights/conv_vae_SonicAndKnuckles.pkl" # 3, 4608
    lstm_mdn_filename = "weights/lstm_mdn_SonicAndKnuckles.pkl" # 4608
    controller_filename = "weights/controller_6656_12.pkl"

    # conv_vae_filename = "weights/conv_vae_gray_edges.pkl" # 1, 1024
    # lstm_mdn_filename = "weights/lstm_mdn_gray_edges.pkl" # 1024
    # controller_filename = "weights/controller_rnn_1024_12.pkl"


    # only forward pass
    conv_vae = ConvVAE((3,128,128), 4608)
    if os.path.exists(conv_vae_filename):
        print("loading conv vae weights")
        conv_vae.load_state_dict(torch.load(conv_vae_filename))

    # only forward pass
    lstm_mdn = LSTM(vector_size=4608)
    if os.path.exists(lstm_mdn_filename):
        print("loading lstm mdn weights")
        lstm_mdn.load_state_dict( torch.load(lstm_mdn_filename) )

    controller = Controller(input_size=6656, action_size=12) # 6656
    if os.path.exists(controller_filename):
        print("loading controller weights")
        controller.load_state_dict( torch.load(controller_filename) )


    # solver = CMAES(num_params=79884, sigma_init=4, popsize=100)
    solver = PEPG(num_params=79884, sigma_init=4, elite_ratio=0.25, popsize=100, forget_best=False)
    solver_sigma_mu_weights_filename="weights/solver_sigma_mu_weights_34_0.30942985.npz"
    print("load sigma mu to solver")
    data = np.load(solver_sigma_mu_weights_filename)
    solver.mu = data["mu"]
    solver.sigma = data["sigma"]
    ## save sigma mu
    # pepg_mu = solver.mu
    # pepg_sigma = solver.sigma
    # np.savez(solver_sigma_mu_filename, mu=pepg_mu, sigma=pepg_sigma)

    # params = list(controller.parameters())
    # weight = params[0] # [12, 6656]  79 872
    # bias = params[1] # [12]
    # # summ: 79 884
    # weight = weight.view(-1) # 79872
    #
    # weights = torch.cat((weight, bias), dim=0) # [79884]


    generations = 40000
    for generation in range(generations):

        solutions = solver.ask() # (40, 79884)

        fitness_list = np.zeros(solver.popsize)  # (40,)


        for i in range( solver.popsize ):
            fitness_list[i] = evaluate(solutions[i], conv_vae, lstm_mdn, controller, env, n_steps=512)
            print( i, fitness_list[i] )


        solver.tell(fitness_list)

        result = solver.result()
        # first element is the best solution, second element is the best fitness
        # print(result[0]) # (79884,)
        # print(result[1]) # -10732.263849138297

        print(generation, result[1])

        ###############################
        ## save solver sigma mu weights
        print("save pepg data")
        solver_sigma_mu_filename = "weights/solver_sigma_mu_weights_%s_%s.npz" %(generation, result[1])
        pepg_mu = solver.mu
        pepg_sigma = solver.sigma
        weights = result[0]
        np.savez(solver_sigma_mu_filename, mu=pepg_mu, sigma=pepg_sigma, weights=weights)

        # save controller weights
        print("save controller weights")
        torch.save(controller.state_dict(), controller_filename)


def evaluate(weights, conv_vae, lstm_mdn, controller, env):
    # print(weights.shape) # (793612,)

    controller.set_weights(weights)

    reward_dict = {
        "current_score": 0,
        "current_x": 0,
        "current_rings": 0,
        "reward_flow": 0,
        "lives": None
    }
    max_reward = 0
    exit_counter = 0

    img = env.reset()

    while True:

        # img = dumyshape(img) # (3, 128, 128)
        img = dumyshape_gray(img) # (1, 128, 128)
        img = torch.FloatTensor(img) # [3, 128, 128]
        img = img.unsqueeze(0) # [1, 3, 128, 128]

        # latent vector
        z = conv_vae(img, encode=True)  # [1, 4608]
        z = z.detach()
        z = z.unsqueeze(0) # [1, 1, 1024]

        # hidden state, cell state
        z_t = lstm_mdn.predict(z)  # [1, 1, 1024]

        input = torch.cat((z, z_t), dim=1)  # [1, 2, 1024]


        actions = controller(input)  # [1, 12]
        actions = actions.squeeze(0).data.numpy() # [1. 1. 1. 1. 1. 0. 1. 0. 1. 0. 1. 0.]


        img, reward, done, info = env.step(actions)

        reward_calculation(reward_dict, info)

        # env.render()

        if reward_dict["reward_flow"] > max_reward:
            max_reward = reward_dict["reward_flow"]
            exit_counter = 0
        else:
            exit_counter += 1

        # print(max_reward)

        if done or exit_counter > 512:
            break


    controller.reset_states()

    # env.close()

    lstm_mdn.reset_states()

    return max_reward


def evaluate_cma(weight, conv_vae, lstm_mdn, controller, env):
    # print(weight.shape) # (6156,)

    controller.set_dence_weights(weight)

    reward_dict = {
        "current_score": 0,
        "current_x": 0,
        "current_rings": 0,
        "reward_flow": 0,
        "lives": None
    }
    max_reward = 0
    exit_counter = 0

    img = env.reset()

    while True:

        # img = dumyshape(img) # (3, 128, 128)
        img = dumyshape_gray(img) # (1, 128, 128)
        img = torch.FloatTensor(img) # [3, 128, 128]
        img = img.unsqueeze(0) # [1, 3, 128, 128]

        # latent vector
        z = conv_vae(img, encode=True)  # [1, 4608]
        z = z.detach()
        z = z.unsqueeze(0) # [1, 1, 1024]

        # hidden state, cell state
        z_t = lstm_mdn.predict(z)  # [1, 1, 1024]

        input = torch.cat((z, z_t), dim=1)  # [1, 2, 1024]


        actions = controller(input)  # [1, 12]
        actions = actions.squeeze(0).data.numpy() # [1. 1. 1. 1. 1. 0. 1. 0. 1. 0. 1. 0.]


        img, reward, done, info = env.step(actions)

        reward_calculation(reward_dict, info)

        # env.render()

        if reward_dict["reward_flow"] > max_reward:
            max_reward = reward_dict["reward_flow"]
            exit_counter = 0
        else:
            exit_counter += 1

        # print(max_reward)

        if done or exit_counter > 512:
            break


    controller.reset_states()

    # env.close()

    lstm_mdn.reset_states()

    return max_reward











if __name__ == "__main__":

    # controller = Controller()

    # train_controller_openes()
    train_controller_cma()
    # test_controller()

    # print( list(controller.parameters())[0].size() ) # [12, 8896]
    # print( list(controller.parameters())[1].size() ) # [12]

    # optimizer = CMAES(num_params=13000)

    # controller = Controller_RNN()
