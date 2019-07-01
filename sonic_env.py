# coding: utf8

import retro
# print(retro.data.list_games())
import cma
import cv2
import numpy as np
import os, sys, time, copy
import torch
import torch.nn.functional as F
import torch.optim as optim


from conv_b_vae import ConvVAE
from mdn_lstm import LSTM
from controller import Controller
from es import CMAES, PEPG

from helper import dumyshape, dumyshape_gray_edges, dumyshape_gray, reward_calculation, dumyshape_gray_thresh



# env_name = "SonicTheHedgehog-Genesis" # None
# env_name = "SonicTheHedgehog2-Genesis"
# env_name = "SonicAndKnuckles-Genesis"
# env_name = "SonicTheHedgehog3-Genesis"
# env_name = "SonicAndKnuckles3-Genesis"

level_2_name = ["EmeraldHillZone.Act1", "EmeraldHillZone.Act2",
                    "ChemicalPlantZone.Act1", "ChemicalPlantZone.Act2",
                    "AquaticRuinZone.Act1", "AquaticRuinZone.Act2",
                    "CasinoNightZone.Act1", "CasinoNightZone.Act2",
                    "HillTopZone.Act1","HillTopZone.Act2",
                    "OilOceanZone.Act1","OilOceanZone.Act2",
                    "MetropolisZone.Act1", "MetropolisZone.Act2", "MetropolisZone.Act3",
                    "WingFortressZone"]

# env = retro.make(env_name)
# print(env.observation_space) # Box(224, 320, 3)
# print(env.action_space) # MultiBinary(12)
# print(env.action_space.sample()) # [1 1 1 0 1 0 1 0 0 1 1 1]



device = torch.device("cpu")






def train_conv_lstm_on_pics(conv_vae,
                            conv_vae_optimizer,
                            lstm_mdn,
                            lstm_mdn_optimizer,
                            src_dir,
                            batch_list):

    # conv_vae_buffer = []
    # latent_vector = None
    dataset = []

    # prepare dataset for learning
    for short_name in batch_list:

        full_name = os.path.join(src_dir, short_name)

        if os.path.exists(full_name):

            img = cv2.imread(full_name) # (128, 128, 3)
            # img = dumyshape(img) # (3, 128, 128)
            # img = dumyshape_shrink_expand(img, scale) # (3, 128, 128)
            # img = dumyshape_gray_edges(img) # (1, 128, 128)
            # img = dumyshape_gray(img) # (1, 128, 128)
            img = dumyshape_gray_thresh(img) # (1, 128, 128)
            img = torch.FloatTensor(img) # [1, 128, 128]

            dataset.append(img)

    dataset = torch.stack(dataset, dim=0) # [4000, 3, 128, 128]

    # sequence_len = dataset.size(0) - 1 # 3999


    #################
    ## train conv vae
    deconv_img, mu, logvar, z = conv_vae(dataset)
    # [4000, 3, 128, 128] [4000, 4608] [4000, 4608] [4000, 4608]
    # [4000, 1, 128, 128] [4000, 1024] [4000, 1024] [4000, 1024]

    conv_vae_loss = conv_vae.conv_vae_loss(deconv_img, dataset, mu, logvar)
    print("conv vae loss:", conv_vae_loss)
    conv_vae_optimizer.zero_grad()
    conv_vae_loss.backward()
    conv_vae_optimizer.step()

    z.detach_()  # [4000, 4608] [4000, 1024]

    #################
    ## lstm mdn train
    # +1 to represent future step
    # time vector:     [t,t,t,t,t]
    # latent vector: [l,l,l,l,l,l]


    # trunkate 1 element from right, to produce 1000 size time vectors
    # time vector:     [t,t,t,t,t]
    # latent vector: [l,l,l,l,l]
    latent_vector_3999_right = z[:-1, :]  # [3999, 4608]
    latent_vector_3999_right = latent_vector_3999_right.unsqueeze(0)  # [1, 3999, 4608]

    # pi, sigma, mu = lstm_mdn(latent_vector_3999_right)
    # [1, 3999, 8, 4608]
    # [1, 3999, 8, 1024]

    z_t = lstm_mdn.predict(latent_vector_3999_right)


    # trunkate 1 element from left
    # time vector:     [t,t,t,t,t]
    # latent vector:   [l,l,l,l,l]
    # 3999 elements buffer: predictions and actual line up
    target_latent_vector = z[1:, :]  # [3999, 4608]
    target_latent_vector = target_latent_vector.unsqueeze(0)  # [1, 3999, 4608]

    # lstm_mdn_loss = lstm_mdn.mdn_loss_function(pi, sigma, mu, target_latent_vector)
    lstm_mse_loss = lstm_mdn.mse_loss_function(actual=target_latent_vector, prediction=z_t)
    print( "lstm mse loss:", lstm_mse_loss )
    lstm_mdn_optimizer.zero_grad()
    # lstm_mdn_loss.backward()
    lstm_mse_loss.backward()
    lstm_mdn_optimizer.step()




def prepare_list_pics():
    # conv_vae_filename = "weights/conv_vae_SonicAndKnuckles.pkl"
    # conv_vae_filename = "weights/conv_vae_gray_edges.pkl"
    conv_vae_filename = "weights/conv_vae_gray.pkl"
    # lstm_mdn_filename = "weights/lstm_mdn_SonicAndKnuckles.pkl"
    # lstm_mdn_filename = "weights/lstm_mdn_gray_edges.pkl"
    lstm_mdn_filename = "weights/lstm_mdn_gray.pkl"

    base_dir = "/opt/Projects/dataset/sonic"

    batch_size = 4000

    conv_vae = ConvVAE((1,128,128), 1024) # 4608
    conv_vae_optimizer = optim.Adam(conv_vae.parameters(), lr=0.00025)
    if os.path.exists(conv_vae_filename):
        print("loading conv vae weights")
        conv_vae.load_state_dict(torch.load(conv_vae_filename))

    lstm_mdn = LSTM(vector_size=1024)
    lstm_mdn_optimizer = optim.Adam(lstm_mdn.parameters(), lr=0.00025)
    if os.path.exists(lstm_mdn_filename):
        print("loading lstm mdn weights")
        lstm_mdn.load_state_dict( torch.load(lstm_mdn_filename) )


    for subdir in os.listdir(base_dir):
        # print(subdir) # 1_1  1_2  2_3  2_4 ...


        # epoch inside subdir
        for epoch in range(200):

            src_dir = os.path.join(base_dir, subdir)  # /opt/Projects/dataset/sonic/1

            list_of_files = list(os.walk(src_dir))[0][2]

            # full_batches = len(list_of_files) // batch_size
            len_of_files = len(list_of_files)  # 79964
            # print( full_batches ) # 79

            start = 0
            offset = batch_size

            while offset <= (len_of_files - 1):
                batch_list = list_of_files[start:offset]  # 1000

                train_conv_lstm_on_pics(conv_vae,
                                        conv_vae_optimizer,
                                        lstm_mdn,
                                        lstm_mdn_optimizer,
                                        src_dir,
                                        batch_list)

                start += batch_size
                offset += batch_size

            print(epoch)

            print("saving conv vae weights")
            torch.save(conv_vae.state_dict(), conv_vae_filename)

            print("saving lstm mdn weights")
            torch.save(lstm_mdn.state_dict(), lstm_mdn_filename)

            print("\n")

            lstm_mdn.reset_states() # после епоча




def train_conv_vae_lstm_mdn():

    print("\n\n\n\n\n")
    # env_name = "SonicTheHedgehog-Genesis" # None
    env_name = "SonicTheHedgehog2-Genesis"
    # env_name = "SonicAndKnuckles-Genesis"
    # env_name = "SonicTheHedgehog3-Genesis"
    # env_name = "SonicAndKnuckles3-Genesis"

    # conv_vae_filename = "weights/conv_vae_gray_edges.pkl" # 1, 1024
    # lstm_mdn_filename = "weights/lstm_mdn_gray_edges.pkl" # 1024
    conv_vae_filename = "weights/conv_vae_gray.pkl" # 1, 1024
    lstm_mdn_filename = "weights/lstm_mdn_gray.pkl" # 1024


    env = retro.make(env_name)
    # print(env.observation_space) # Box(224, 320, 3)
    # print(env.action_space) # MultiBinary(12)
    # print(env.action_space.sample()) # [1 1 1 0 1 0 1 0 0 1 1 1]

    conv_vae_buffer = []
    latent_vector = None
    batch_size = 50
    sequence_len = batch_size - 1


    conv_vae = ConvVAE((1,128,128), 1024)
    conv_vae_optimizer = optim.Adam(conv_vae.parameters(), lr=0.00025)
    if os.path.exists(conv_vae_filename):
        print("loading conv vae weights")
        conv_vae.load_state_dict(torch.load(conv_vae_filename))

    lstm_mdn = LSTM(vector_size=1024)
    lstm_mdn_optimizer = optim.Adam(lstm_mdn.parameters(), lr=0.00025)
    if os.path.exists(lstm_mdn_filename):
        print("loading lstm mdn weights")
        lstm_mdn.load_state_dict( torch.load(lstm_mdn_filename) )


    for episode in range(1,2):

        img = env.reset() # (224, 320, 3)

        step = 0

        reward_dict = {
            "current_score":0,
            "current_x":0,
            "current_rings":0,
            "reward_flow":0,
            "lives":None
        }

        while True:

            # img = dumyshape_gray_edges(img)  # (1, 128, 128)
            img = dumyshape_gray(img) # (1, 128, 128)
            img = torch.FloatTensor(img) # [1, 128, 128]
            img = img.unsqueeze(0) # [1, 1, 128, 128]

            conv_vae_buffer.append(img)


            ################
            ## cnn_vae train
            if len(conv_vae_buffer) == batch_size: # 1000


                conv_vae_buffer = torch.cat(conv_vae_buffer) # [4000, 1, 128, 128]


                deconv_img, mu, logvar, z = conv_vae(conv_vae_buffer)
                # [1000, 1, 128, 128] [1000, 1024] [1000, 1024] [1000, 1024]


                ##############
                ## lstm buffer
                if latent_vector is None:
                    latent_vector = z
                else:
                    latent_vector = torch.cat( (latent_vector, z), dim=0)


                #################
                ## conv vae train
                # conv_vae_loss = conv_vae.conv_vae_loss(deconv_img, conv_vae_buffer, mu, logvar)
                # print(step, "loss:",conv_vae_loss)
                # conv_vae_optimizer.zero_grad()
                # conv_vae_loss.backward()
                # conv_vae_optimizer.step()

                # zero out conv buffer
                conv_vae_buffer = []




            #################
            ## lstm mdn train
            # +1 to represent future step
            # time vector:     [t,t,t,t,t]
            # latent vector: [l,l,l,l,l,l]
            if latent_vector is not None and latent_vector.size(0) >= sequence_len+1:
                print("lstm mdn training", step, latent_vector.size())

                # cut vector to sequence_len + 1
                latent_vector_1001 = latent_vector[:sequence_len+1,:] # [1001, 4608]


                # trunkate 1 element from right, to produce 1000 size time vectors
                # time vector:     [t,t,t,t,t]
                # latent vector: [l,l,l,l,l]
                latent_vector_1000_right = latent_vector_1001[:-1,:] # [1000, 4608]
                latent_vector_1000_right = latent_vector_1000_right.unsqueeze(0) # [1, 1000, 4608]


                # pi, sigma, mu = lstm_mdn(latent_vector_1000_right)
                # [1, 1000, 5, 4608]

                z_t = lstm_mdn.predict(latent_vector_1000_right) # [1, 49, 1024]

                # trunkate 1 element from left
                # time vector:     [t,t,t,t,t]
                # latent vector:   [l,l,l,l,l]
                # 1000 elements buffer: predictions and actual line up
                target_latent_vector = latent_vector_1001[1:,:] # [1000, 4608]
                target_latent_vector = target_latent_vector.unsqueeze(0) # [1, 1000, 4608]


                # lstm_mdn_loss = lstm_mdn.mdn_loss_function(pi, sigma, mu, target_latent_vector)
                lstm_mse_loss = lstm_mdn.mse_loss_function(actual=target_latent_vector, prediction=z_t)

                lstm_mdn_optimizer.zero_grad()

                # lstm_mdn_loss.backward()
                lstm_mse_loss.backward()

                lstm_mdn_optimizer.step()

                # zero out buffer and states
                latent_vector = None
                lstm_mdn.reset_states()



            action = env.action_space.sample()
            action[7] = 1

            img, reward, done, info = env.step(action)

            #####################
            ## reward calculation
            # reward_flow = reward_calculation(reward_dict, info)

            # print(reward_flow)
            # time.sleep(.025)
            # env.render()

            step += 1



            #################
            ## save weights
            if step >= 8000:
                print("saving weights")
                torch.save(conv_vae.state_dict(), conv_vae_filename)
                # torch.save(lstm_mdn.state_dict(), lstm_mdn_filename)
                step = 0


            if done:
                break

    env.close()





def train_controller():
    print("\n\n\n\n\n")

    # env_name = "SonicTheHedgehog-Genesis" # None
    env_name = "SonicTheHedgehog2-Genesis"
    # env_name = "SonicAndKnuckles-Genesis"
    # env_name = "SonicTheHedgehog3-Genesis"
    # env_name = "SonicAndKnuckles3-Genesis"

    env = retro.make(env_name)
    # print(env.observation_space) # Box(224, 320, 3)
    # print(env.action_space) # MultiBinary(12)
    # print(env.action_space.sample()) # [1 1 1 0 1 0 1 0 0 1 1 1]


    conv_vae_filename = "weights/conv_vae_gray.pkl"
    lstm_mdn_filename = "weights/lstm_mdn_gray.pkl"
    controller_filename = "weights/controller.pkl"

    # only forward pass
    conv_vae = ConvVAE((1,128,128), 1024)
    if os.path.exists(conv_vae_filename):
        print("loading cnn vae weights")
        conv_vae.load_state_dict(torch.load(conv_vae_filename))

    # only forward pass
    lstm_mdn = LSTM(sequence_len=1)
    if os.path.exists(lstm_mdn_filename):
        print("loading lstm mdn weights")
        lstm_mdn.load_state_dict( torch.load(lstm_mdn_filename) )


    controller = Controller(input_size=6656, action_size=12)
    if os.path.exists(controller_filename):
        print("loading controller weights")
        # lstm_mdn.load_state_dict( torch.load(controller_filename) )


    # solver = CMAES(num_params=79884, sigma_init=4, popsize=100)
    solver = PEPG(num_params=79884, sigma_init=4, elite_ratio=0.25, popsize=400, forget_best=False)
    solver_sigma_mu_weights_filename="weights/solver_sigma_mu_weights_0_0.33994994.npz"
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

        solutions = solver.ask() # (400, 79884)

        fitness_list = np.zeros(solver.popsize)  # (400,)


        for i in range( solver.popsize ):
            fitness_list[i] = evaluate(solutions[i], conv_vae, lstm_mdn, controller, env, n_steps=5)
            print( i, fitness_list[i] )

        print(fitness_list)

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

    print()


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
    controller_filename = "weights/controller_dence_cma_2048_12.pkl"
    evaluator_cma_filename = "weights/evaluator_cma_weights_13_11886.000000000055.npz"


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

    controller = Controller(input_size=2048, action_size=12)
    if os.path.exists(controller_filename):
        print("loading controller weights")
        controller.load_state_dict( torch.load(controller_filename) )

    # evaluator openes restore
    # if os.path.exists(evaluator_openes_filename):
    #     print("loading openes evaluator data")
    #     data = np.load(evaluator_openes_filename)
    #     weights = data["weights"]
    #     print("inserting weights into controller")
    #     controller.set_weights(weights)


    # evaluator cma
    if os.path.exists(evaluator_cma_filename):
        print("loading cma evaluator data")
        data = np.load(evaluator_cma_filename)
        mean_weights = data["mean"]
        # best_weights = data["best"]
        print("inserting dense weights into controller")
        controller.set_weights(mean_weights)

        sigma_init = 0.10 # initial standard deviation
        evaluator = cma.CMAEvolutionStrategy(mean_weights, sigma_init, {"popsize": population_size})

    else:
        print("NO cma filename found, extracting dence weights from controller")
        state_dict = controller.state_dict() # 6
        dence_weight = state_dict["fc1.weight"] # [12, 2048] 24576
        dence_weight = dence_weight.view(-1)  # [24576]

        dence_bias = state_dict["fc1.bias"] # [12]         12

        weights = torch.cat((dence_weight, dence_bias), dim=0) # [24588]
        weights = weights.data.numpy() # (24588,)


        sigma_init = 0.10 # initial standard deviation
        evaluator = cma.CMAEvolutionStrategy(weights, sigma_init, {"popsize": population_size})


    for generation in range(generations):

        solutions = evaluator.ask() # 256, 24588 - list

        fitness = np.zeros(population_size)

        # evaluate solutions
        for index in range(population_size):

            curr_weight = solutions[index] # (24588,)

            fitness[index] = evaluate(curr_weight, conv_vae, lstm_mdn, controller, env)

        # print(fitness)

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



def evaluate(weights, conv_vae, lstm_mdn, controller, env):
    # print(weights.shape) # (24588,)

    controller.set_weights(weights)


    reward_dict = {
        "current_score": 0,
        "current_x": 0,
        "current_rings": 0,
        "reward_flow": 0,
        "lives": None
    }
    # reward_calculation(reward_dict, reward_dict, reset=True)

    max_reward = 0
    exit_counter = 0
    img = env.reset()


    while True:

        # img = dumyshape(img) # (1, 3, 128, 128)
        img = dumyshape_gray_thresh(img) # (1, 128, 128)
        img = torch.FloatTensor(img)
        img = img.unsqueeze(0)


        # latent vector
        z = conv_vae(img, encode=True)  # [1, 1024]
        z = z.detach()

        # hidden state, cell state
        # h, c = lstm_mdn.encode(z.unsqueeze(0))  # [2, 1, 512] [2, 1, 512]
        # h = h.view(1, -1)  # [1, 1024]
        # c = c.view(1, -1)  # [1, 1024]
        z_t = lstm_mdn.predict( z.unsqueeze(0) ) # [1, 1, 1024]
        z_t = z_t.squeeze(0) # [1, 1024]



        input = torch.cat((z, z_t), dim=1)  # [1, 2048]


        actions = controller(input)  # [1, 12]
        actions = actions.squeeze(0).cpu().data.numpy()  # (12,)
        actions = actions.astype("int")  # [0 1 0 1 0 1 1 1 1 1 1 1]
        # actions[7] = 1

        img, reward, done, info = env.step(actions)

        reward_calculation(reward_dict, info)

        if reward_dict["reward_flow"] > max_reward:
            max_reward = reward_dict["reward_flow"]
            exit_counter = 0
        else:
            exit_counter += 1

        if exit_counter >= 512:
            break



    return reward_dict["reward_flow"]





def test_conv_vae():
    print("\n\n\n\n\n")

    # env_name = "SonicTheHedgehog-Genesis" # None
    env_name = "SonicTheHedgehog2-Genesis"
    # env_name = "SonicAndKnuckles-Genesis"
    # env_name = "SonicTheHedgehog3-Genesis"
    # env_name = "SonicAndKnuckles3-Genesis"


    env = retro.make(env_name) # level_2_name[1]
    # print(env.observation_space) # Box(224, 320, 3)
    # print(env.action_space) # MultiBinary(12)
    # print(env.action_space.sample()) # [1 1 1 0 1 0 1 0 0 1 1 1]


    # conv_vae_filename = "weights/conv_vae_SonicAndKnuckles.pkl" # 4608
    # conv_vae_filename = "weights/conv_vae_gray_edges.pkl"
    conv_vae_filename = "weights/conv_vae_gray.pkl"


    # only forward pass
    conv_vae = ConvVAE((1,128,128), 1024)
    if os.path.exists(conv_vae_filename):
        print("loading conv vae weights")
        conv_vae.load_state_dict(torch.load(conv_vae_filename))


    for episode in range(1,2):

        img = env.reset()

        while True:

            # img = dumyshape_gray_edges(img) # (1, 128, 128)
            img = dumyshape_gray(img) # (1, 128, 128)
            # img = dumyshape(img) # (3, 128, 128)
            img = torch.FloatTensor(img) # [3, 128, 128]
            img = img.unsqueeze(0) # [1, 3, 128, 128]

            deconv_img, mu, logvar, z = conv_vae(img)
            # [1, 1, 128, 128] [1, 1024] [1, 1024] [1, 1024]

            deconv_img = deconv_img.squeeze(0).cpu().data.numpy() # (1, 128, 128)
            deconv_img = np.transpose(deconv_img, (1,2,0)) # (128, 128, 3)
            # deconv_img *= 255


            env.render()

            cv2.imshow("conv vae deconv", deconv_img)
            # cv2.imshow("conv vae deconv", cv2.cvtColor(deconv_img, cv2.COLOR_BGR2RGB) )
            cv2.waitKey(1)


            action = env.action_space.sample()
            action[7] = 1

            img, reward, done, info = env.step(action)


            if done:
                break

    print()


def test_lstm_mdn():

    print("\n\n\n\n\n")

    # env_name = "SonicTheHedgehog-Genesis" # None
    env_name = "SonicTheHedgehog2-Genesis"
    # env_name = "SonicAndKnuckles-Genesis"
    # env_name = "SonicTheHedgehog3-Genesis"
    # env_name = "SonicAndKnuckles3-Genesis"

    env = retro.make(env_name)
    # print(env.observation_space) # Box(224, 320, 3)
    # print(env.action_space) # MultiBinary(12)
    # print(env.action_space.sample()) # [1 1 1 0 1 0 1 0 0 1 1 1]


    # conv_vae_filename = "weights/conv_vae_SonicAndKnuckles.pkl" # 4608
    # lstm_mdn_filename = "weights/lstm_mdn_SonicAndKnuckles.pkl"
    # conv_vae_filename = "weights/conv_vae_gray_edges.pkl"
    # lstm_mdn_filename = "weights/lstm_mdn_gray_edges.pkl"
    conv_vae_filename = "weights/conv_vae_gray.pkl"
    lstm_mdn_filename = "weights/lstm_mdn_gray.pkl"


    # only forward pass
    conv_vae = ConvVAE((1,128,128), 1024)
    if os.path.exists(conv_vae_filename):
        print("loading cnn vae weights")
        conv_vae.load_state_dict(torch.load(conv_vae_filename))

    # only forward pass
    lstm_mdn = LSTM(vector_size=1024)
    if os.path.exists(lstm_mdn_filename):
        print("loading lstm mdn weights")
        lstm_mdn.load_state_dict( torch.load(lstm_mdn_filename) )

    for episode in range(1,2):

        img = env.reset()

        while True:

            # img = dumyshape(img)
            # img = dumyshape_gray_edges(img)
            img = dumyshape_gray(img)
            img = torch.FloatTensor(img) # [3, 128, 128]
            img = img.unsqueeze(0) # [1, 3, 128, 128]

            z = conv_vae(img, encode=True)
            z = z.unsqueeze(0) # [1, 1, 4608]

            z_t = lstm_mdn.predict(z) # [1, 1, 1024]
            z_t = z_t.squeeze(0) # [1, 1024]

            deconv_img = conv_vae.decode(z_t) # [1, 1, 128, 128]

            deconv_img = deconv_img.squeeze(0).cpu().data.numpy()
            deconv_img = np.transpose(deconv_img, (1,2,0))
            # deconv_img *= 255

            env.render()

            cv2.imshow("lstm_mdn reconstruct", deconv_img)
            # cv2.imshow("lstm_mdn reconstruct", cv2.cvtColor(deconv_img, cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)


            action = env.action_space.sample()
            action[7] = 1

            img, reward, done, info = env.step(action)


            if done:
                break

    print()


def test_controller():
    print("\n\n\n\n\n")

    # env_name = "SonicTheHedgehog-Genesis" # None
    # env_name = "SonicTheHedgehog2-Genesis"
    # env_name = "SonicAndKnuckles-Genesis"
    # env_name = "SonicTheHedgehog3-Genesis"
    env_name = "SonicAndKnuckles3-Genesis"

    env = retro.make(env_name)
    # print(env.observation_space) # Box(224, 320, 3)
    # print(env.action_space) # MultiBinary(12)
    # print(env.action_space.sample()) # [1 1 1 0 1 0 1 0 0 1 1 1]


    conv_vae_filename = "weights/conv_vae_SonicAndKnuckles.pkl"
    lstm_mdn_filename = "weights/lstm_mdn_SonicAndKnuckles.pkl"
    solver_filename = "weights/solver_sigma_mu_weights_22_0.31758243.npz"


    # only forward pass
    conv_vae = ConvVAE((3,128,128), 4608)
    if os.path.exists(conv_vae_filename):
        print("loading cnn vae weights")
        conv_vae.load_state_dict(torch.load(conv_vae_filename))

    # only forward pass
    lstm_mdn = LSTM(sequence_len=1)
    if os.path.exists(lstm_mdn_filename):
        print("loading lstm mdn weights")
        lstm_mdn.load_state_dict( torch.load(lstm_mdn_filename) )


    controller = Controller(input_size=6656, action_size=12)
    if os.path.exists(solver_filename):
        print("loading controller weights")

        data = np.load(solver_filename)
        weights = data["weights"]
        # solver.mu = data["mu"]
        # solver.sigma = data["sigma"]

        controller.set_weights(weights)


    for episode in range(1,2):

        img = env.reset()

        while True:

            img = dumyshape(img)
            img = torch.FloatTensor(img)

            # latent vector
            z = conv_vae(img, encode=True)

            # hidden state, cell state
            h, c = lstm_mdn.encode( z.unsqueeze(0) ) # [2, 1, 512] [2, 1, 512]
            h = h.view(1, -1)  # [1, 1024]
            c = c.view(1, -1)  # [1, 1024]

            input = torch.cat((z, h, c), dim=1) # [1, 6656]

            actions = controller(input) # [[1., 1., 1., 0., 1., 0., 1., 1., 0., 1., 1., 0.]]
            actions = actions.squeeze(0).cpu().data.numpy()
            actions = actions.astype("int") # [1 0 0 0 1 1 0 0 0 1 1 1]


            # decode latent vector
            deconv_img = conv_vae.decode(z)
            deconv_img = deconv_img.squeeze(0).cpu().data.numpy()
            deconv_img = np.transpose(deconv_img, (1,2,0))
            deconv_img *= 255
            deconv_img = cv2.cvtColor(deconv_img, cv2.COLOR_BGR2RGB)

            env.render()

            cv2.imshow("conv vae", deconv_img)
            cv2.waitKey(1)




            img, reward, done, info = env.step(actions)





            if done:
                break



def simple_env_run():
    env_name = "SonicTheHedgehog2-Genesis"
    # env_name = "SonicAndKnuckles-Genesis"
    # env_name = "SonicTheHedgehog3-Genesis"
    # env_name = "SonicAndKnuckles3-Genesis"

    env = retro.make(env_name)
    # print(env.observation_space) # Box(224, 320, 3)
    # print(env.action_space) # MultiBinary(12)
    # print(env.action_space.sample()) # [1 1 1 0 1 0 1 0 0 1 1 1]

    img = env.reset()

    while True:

        img = dumyshape_gray_thresh(img)


        # actions = env.action_space.sample()
        actions = [0] * 12
        actions[7] =1

        img, reward, done, info = env.step(actions)






if __name__ == "__main__":
    # train_conv_vae_lstm_mdn()
    # train_conv_lstm_on_pics()
    # prepare_list_pics()

    # train_controller()
    train_controller_cma()

    # test_conv_vae()
    # test_lstm_mdn()
    # test_controller()

    # simple_env_run()
