# coding: utf8

import numpy as np
import sys

def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    (https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    """
    https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
    """
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y


def compute_weight_decay(weight_decay, model_param_list):
    model_param_grid = np.array(model_param_list)
    return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)


# adopted from:
# https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/optimizers.py

class Optimizer(object):
    def __init__(self, pi, epsilon=1e-08):
        self.pi = pi
        self.dim = pi.num_params
        self.epsilon = epsilon
        self.t = 0

    def update(self, globalg):
        self.t += 1
        step = self._compute_step(globalg)
        theta = self.pi.mu
        ratio = np.linalg.norm(step) / (np.linalg.norm(theta) + self.epsilon)
        self.pi.mu = theta + step
        return ratio

    def _compute_step(self, globalg):
        raise NotImplementedError


class BasicSGD(Optimizer):
    def __init__(self, pi, stepsize):
        Optimizer.__init__(self, pi)
        self.stepsize = stepsize

    def _compute_step(self, globalg):
        step = -self.stepsize * globalg
        return step


class SGD(Optimizer):
    def __init__(self, pi, stepsize, momentum=0.9):
        Optimizer.__init__(self, pi)
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.stepsize, self.momentum = stepsize, momentum

    def _compute_step(self, globalg):
        self.v = self.momentum * self.v + (1. - self.momentum) * globalg
        step = -self.stepsize * self.v
        return step


class Adam(Optimizer):
    def __init__(self, pi, stepsize, beta1=0.99, beta2=0.999):
        Optimizer.__init__(self, pi)
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def _compute_step(self, globalg):
        a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step


class CMAES:
    '''CMA-ES wrapper.'''

    def __init__(self, num_params,  # number of model parameters
                 sigma_init=0.10,  # initial standard deviation
                 popsize=255,  # population size
                 weight_decay=0.01):  # weight decay coefficient

        self.num_params = num_params
        self.sigma_init = sigma_init
        self.popsize = popsize
        self.weight_decay = weight_decay
        self.solutions = None

        import cma
        self.es = cma.CMAEvolutionStrategy(self.num_params * [0],
                                           self.sigma_init,
                                           {'popsize': self.popsize,
                                            })

    def rms_stdev(self):
        sigma = self.es.result[6]
        return np.mean(np.sqrt(sigma * sigma))

    def ask(self):
        '''returns a list of parameters'''

        self.solutions = np.array(self.es.ask())
        return self.solutions

    def tell(self, reward_table_result):
        reward_table = -np.array(reward_table_result)
        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
            reward_table += l2_decay
        self.es.tell(self.solutions, (reward_table).tolist())  # convert minimizer to maximizer.

    def current_param(self):
        return self.es.result[5]  # mean solution, presumably better with noise

    def set_mu(self, mu):
        pass

    def best_param(self):
        return self.es.result[0]  # best evaluated solution

    def result(self):  # return best params so far, along with historically best reward, curr reward, sigma
        r = self.es.result
        return (r[0], -r[1], -r[1], r[6])


class SimpleGA:
    '''Simple Genetic Algorithm.'''

    def __init__(self, num_params,  # number of model parameters
                 sigma_init=0.1,  # initial standard deviation
                 sigma_decay=0.999,  # anneal standard deviation
                 sigma_limit=0.01,  # stop annealing if less than this
                 popsize=256,  # population size
                 elite_ratio=0.1,  # percentage of the elites
                 forget_best=False,  # forget the historical best elites
                 weight_decay=0.01,  # weight decay coefficient
                 ):

        self.num_params = num_params
        self.sigma_init = sigma_init
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.popsize = popsize

        self.elite_ratio = elite_ratio
        self.elite_popsize = int(self.popsize * self.elite_ratio)

        self.sigma = self.sigma_init
        self.elite_params = np.zeros((self.elite_popsize, self.num_params))
        self.elite_rewards = np.zeros(self.elite_popsize)
        self.best_param = np.zeros(self.num_params)
        self.best_reward = 0
        self.first_iteration = True
        self.forget_best = forget_best
        self.weight_decay = weight_decay

    def rms_stdev(self):
        return self.sigma  # same sigma for all parameters.

    def ask(self):
        '''returns a list of parameters'''
        self.epsilon = np.random.randn(self.popsize, self.num_params) * self.sigma
        solutions = []

        def mate(a, b):
            c = np.copy(a)
            idx = np.where(np.random.rand((c.size)) > 0.5)
            c[idx] = b[idx]
            return c

        elite_range = range(self.elite_popsize)
        for i in range(self.popsize):
            idx_a = np.random.choice(elite_range)
            idx_b = np.random.choice(elite_range)
            child_params = mate(self.elite_params[idx_a], self.elite_params[idx_b])
            solutions.append(child_params + self.epsilon[i])

        solutions = np.array(solutions)
        self.solutions = solutions

        return solutions

    def tell(self, reward_table_result):
        # input must be a numpy float array
        assert (len(reward_table_result) == self.popsize), "Inconsistent reward_table size reported."

        reward_table = np.array(reward_table_result)

        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
            reward_table += l2_decay

        if self.forget_best or self.first_iteration:
            reward = reward_table
            solution = self.solutions
        else:
            reward = np.concatenate([reward_table, self.elite_rewards])
            solution = np.concatenate([self.solutions, self.elite_params])

        idx = np.argsort(reward)[::-1][0:self.elite_popsize]

        self.elite_rewards = reward[idx]
        self.elite_params = solution[idx]

        self.curr_best_reward = self.elite_rewards[0]

        if self.first_iteration or (self.curr_best_reward > self.best_reward):
            self.first_iteration = False
            self.best_reward = self.elite_rewards[0]
            self.best_param = np.copy(self.elite_params[0])

        if (self.sigma > self.sigma_limit):
            self.sigma *= self.sigma_decay

    def current_param(self):
        return self.elite_params[0]

    def set_mu(self, mu):
        pass

    def best_param(self):
        return self.best_param

    def result(self):  # return best params so far, along with historically best reward, curr reward, sigma
        return (self.best_param, self.best_reward, self.curr_best_reward, self.sigma)


class OpenES:
    ''' Basic Version of OpenAI Evolution Strategies.'''

    def __init__(self, num_params,  # number of model parameters
                 sigma_init=0.1,  # initial standard deviation
                 sigma_decay=0.999,  # anneal standard deviation
                 sigma_limit=0.01,  # stop annealing if less than this
                 learning_rate=0.01,  # learning rate for standard deviation
                 learning_rate_decay=0.9999,  # annealing the learning rate
                 learning_rate_limit=0.001,  # stop annealing learning rate
                 popsize=256,  # population size
                 antithetic=False,  # whether to use antithetic sampling
                 weight_decay=0.01,  # weight decay coefficient
                 rank_fitness=True,  # use rank rather than fitness numbers
                 forget_best=True, # forget historical best
                 existing_weights = None # pass already existing weights to be the source
                 ):

        self.num_params = num_params
        self.sigma_decay = sigma_decay
        self.sigma = sigma_init
        self.sigma_init = sigma_init
        self.sigma_limit = sigma_limit
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_limit = learning_rate_limit
        self.popsize = popsize
        self.antithetic = antithetic
        if self.antithetic:
            assert (self.popsize % 2 == 0), "Population size must be even"
            self.half_popsize = int(self.popsize / 2)

        self.reward = np.zeros(self.popsize)
        self.mu = np.zeros(self.num_params)
        self.best_mu = np.zeros(self.num_params)
        self.best_reward = 0
        self.first_interation = True
        self.forget_best = forget_best
        self.weight_decay = weight_decay
        self.rank_fitness = rank_fitness
        if self.rank_fitness:
            self.forget_best = True  # always forget the best one if we rank
        # choose optimizer
        self.optimizer = Adam(self, learning_rate)

        self.existing_weights = existing_weights
        if existing_weights is not None:
            self.set_existing_weights()

    def set_existing_weights(self):
        # print( weights.shape ) # 793612

        lst = [ self.existing_weights + ( np.random.randn() * 0.001 )
                for _ in range(self.popsize)  ]

        self.epsilon = np.array( lst ) # (64, 793612)

    def rms_stdev(self):
        sigma = self.sigma
        return np.mean(np.sqrt(sigma * sigma))

    def ask(self):
        '''returns a list of parameters'''
        # antithetic sampling
        if self.existing_weights is None:
            if self.antithetic:
                self.epsilon_half = np.random.randn(self.half_popsize, self.num_params)
                self.epsilon = np.concatenate([self.epsilon_half, - self.epsilon_half])
            else:
                self.epsilon = np.random.randn(self.popsize, self.num_params)

        self.solutions = self.mu.reshape(1, self.num_params) + self.epsilon * self.sigma

        return self.solutions

    def tell(self, reward_table_result):
        # input must be a numpy float array
        assert (len(reward_table_result) == self.popsize), "Inconsistent reward_table size reported."

        reward = np.array(reward_table_result)

        if self.rank_fitness:
            reward = compute_centered_ranks(reward)

        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
            reward += l2_decay

        idx = np.argsort(reward)[::-1]

        best_reward = reward[idx[0]]
        best_mu = self.solutions[idx[0]]

        self.curr_best_reward = best_reward
        self.curr_best_mu = best_mu

        if self.first_interation:
            self.first_interation = False
            self.best_reward = self.curr_best_reward
            self.best_mu = best_mu
        else:
            if self.forget_best or (self.curr_best_reward > self.best_reward):
                self.best_mu = best_mu
                self.best_reward = self.curr_best_reward

        # main bit:
        # standardize the rewards to have a gaussian distribution
        normalized_reward = (reward - np.mean(reward)) / np.std(reward)
        change_mu = 1. / (self.popsize * self.sigma) * np.dot(self.epsilon.T, normalized_reward)

        # self.mu += self.learning_rate * change_mu

        self.optimizer.stepsize = self.learning_rate
        update_ratio = self.optimizer.update(-change_mu)

        # adjust sigma according to the adaptive sigma calculation
        if (self.sigma > self.sigma_limit):
            self.sigma *= self.sigma_decay

        if (self.learning_rate > self.learning_rate_limit):
            self.learning_rate *= self.learning_rate_decay

    def current_param(self):
        return self.curr_best_mu

    def set_mu(self, mu):
        self.mu = np.array(mu)

    def best_param(self):
        return self.best_mu

    def result(self):  # return best params so far, along with historically best reward, curr reward, sigma
        return (self.best_mu, self.best_reward, self.curr_best_reward, self.sigma)


class PEPG:
    '''Extension of PEPG with bells and whistles.'''

    def __init__(self, num_params,  # number of model parameters
                 sigma_init=0.10,  # initial standard deviation
                 sigma_alpha=0.20,  # learning rate for standard deviation
                 sigma_decay=0.999,  # anneal standard deviation
                 sigma_limit=0.01,  # stop annealing if less than this
                 sigma_max_change=0.2,  # clips adaptive sigma to 20%
                 learning_rate=0.01,  # learning rate for standard deviation
                 learning_rate_decay=0.9999,  # annealing the learning rate
                 learning_rate_limit=0.01,  # stop annealing learning rate
                 elite_ratio=0,  # if > 0, then ignore learning_rate
                 popsize=256,  # population size
                 average_baseline=True,  # set baseline to average of batch
                 weight_decay=0.01,  # weight decay coefficient
                 rank_fitness=True,  # use rank rather than fitness numbers
                 forget_best=True):  # don't keep the historical best solution

        self.num_params = num_params
        self.sigma_init = sigma_init
        self.sigma_alpha = sigma_alpha
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.sigma_max_change = sigma_max_change
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_limit = learning_rate_limit
        self.popsize = popsize
        self.average_baseline = average_baseline
        if self.average_baseline:
            assert (self.popsize % 2 == 0), "Population size must be even"
            self.batch_size = int(self.popsize / 2)
        else:
            assert (self.popsize & 1), "Population size must be odd"
            self.batch_size = int((self.popsize - 1) / 2)

        # option to use greedy es method to select next mu, rather than using drift param
        self.elite_ratio = elite_ratio
        self.elite_popsize = int(self.popsize * self.elite_ratio)
        self.use_elite = False
        if self.elite_popsize > 0:
            self.use_elite = True

        self.forget_best = forget_best
        self.batch_reward = np.zeros(self.batch_size * 2)
        self.mu = np.zeros(self.num_params)
        self.sigma = np.ones(self.num_params) * self.sigma_init
        self.curr_best_mu = np.zeros(self.num_params)
        self.best_mu = np.zeros(self.num_params)
        self.best_reward = 0
        self.first_interation = True
        self.weight_decay = weight_decay
        self.rank_fitness = rank_fitness
        if self.rank_fitness:
            self.forget_best = True  # always forget the best one if we rank
        # choose optimizer
        self.optimizer = Adam(self, learning_rate)

    def rms_stdev(self):
        sigma = self.sigma
        return np.mean(np.sqrt(sigma * sigma))



    def ask(self):
        '''returns a list of parameters'''
        # antithetic sampling

        self.epsilon = np.random.randn(self.batch_size, self.num_params) * self.sigma.reshape(1, self.num_params) # (200, 79884)
        self.epsilon_full = np.concatenate([self.epsilon, - self.epsilon]) # (400, 79884)

        """
        self.sigma
        [4. 4. 4. ... 4. 4. 4.] 79884
        """


        if self.average_baseline:
            epsilon = self.epsilon_full
        else:
            # first population is mu, then positive epsilon, then negative epsilon
            epsilon = np.concatenate([np.zeros((1, self.num_params)), self.epsilon_full])


        """
        self.mu
        [0. 0. 0. ... 0. 0. 0.] 79884
        """

        solutions = self.mu.reshape(1, self.num_params) + epsilon
        # self.mu - нули по кол-ву параметров, заполяется по ходу обучения
        # (400, 79884)

        self.solutions = solutions
        return solutions



    def tell(self, reward_table_result):
        # input must be a numpy float array
        assert (len(reward_table_result) == self.popsize), "Inconsistent reward_table size reported."

        reward_table = np.array(reward_table_result)

        if self.rank_fitness:
            # print("rank_fitness")
            reward_table = compute_centered_ranks(reward_table) # (400,)


        """
        reward_table
        [-0.5         0.17418545  0.1716792   0.16917294  0.16666669  0.16416043
          0.16165411  0.15914786  0.1566416   0.15413535  0.15162909  0.14912283
          0.14661652  0.14411026  0.141604    0.13909775  0.1365915   0.13408524
          0.13157892  0.12907267  0.12656641  0.12406015  0.17669171  0.49749374
          """


        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
            reward_table += l2_decay


        """
        reward_table
        [-0.65956575  0.01494922  0.0104529   0.00980288  0.00620125  0.00449006
          0.00263282 -0.00067969 -0.00213331 -0.00558219 -0.00765776 -0.01002224
         -0.01389918 -0.01487547 -0.01853203 -0.02187017 -0.02392901 -0.02571444
         -0.03005311 -0.0309077  -0.03573698 -0.03582077  0.01647482  0.3364253
        """


        reward_offset = 1
        if self.average_baseline:
            b = np.mean(reward_table) # -0.16002512
            reward_offset = 0
        else:
            b = reward_table[0]  # baseline


        reward = reward_table[reward_offset:]
        if self.use_elite:
            idx = np.argsort(reward)[::-1][0:self.elite_popsize] # (100,)
        else:
            idx = np.argsort(reward)[::-1]


        """
        idx - indexes of best scores
        [123  23 274 339  67 134 127 128 130 129 131 132 133 135 125 136 137 138
         139 140 141 142 143 144 145 126 399 124 147 102 103 104 105 106 107 108
         109 110 111 112 113 114 115 116 117 118 119 120 121 122 146 148 100 185
         175 176 177 178 179 180 181 182 183 184 186 149 187 188 189 190 191 192
         193 194 195 196 174 173 172 171 150 151 152 153 154 155 156 157 158 159
         160 161 162 163 164 165 166 167 168 169]
        """


        best_reward = reward[idx[0]]
        if (best_reward > b or self.average_baseline):
            best_mu = self.mu + self.epsilon_full[idx[0]] # складывает нули с лучшим весом
            best_reward = reward[idx[0]] # 0.34101945
        else:
            best_mu = self.mu
            best_reward = b

        """
        self.epsilon_full[idx[0]] - best weights set
        
        """

        self.curr_best_reward = best_reward
        self.curr_best_mu = best_mu


        if self.first_interation:
            self.sigma = np.ones(self.num_params) * self.sigma_init
            self.first_interation = False
            self.best_reward = self.curr_best_reward
            self.best_mu = best_mu
        else:
            if self.forget_best or (self.curr_best_reward > self.best_reward):
                self.best_mu = best_mu
                self.best_reward = self.curr_best_reward


        # short hand
        epsilon = self.epsilon # (200, 79884)
        sigma = self.sigma # (79884,)

        """
        sigma
        [4. 4. 4. ... 4. 4. 4.]
        """


        """
        self.mu
        [0. 0. 0. ... 0. 0. 0.]
        """

        # update the mean

        # move mean to the average of the best idx means
        if self.use_elite:
            self.mu += self.epsilon_full[idx].mean(axis=0) # нули mu сложить с усреднением лучших весов
        else:
            rT = (reward[:self.batch_size] - reward[self.batch_size:])
            change_mu = np.dot(rT, epsilon)
            self.optimizer.stepsize = self.learning_rate
            update_ratio = self.optimizer.update(-change_mu)  # adam, rmsprop, momentum, etc.
            # self.mu += (change_mu * self.learning_rate) # normal SGD method


        """
        self.epsilon_full[idx] - лучшие веса
        (100, 79884)
        [[ 0.02266545 -4.35845032 -0.93838599 ...  2.45283541  5.44484994
          -6.90347066]
         [-1.97980407 -1.60269346 -7.78217363 ... -6.7109332  -3.4263423
          -2.75492042]
         [-3.19258342  0.77048631  5.56987826 ... -5.20690676  3.31503534
           1.92358303]
         ...
         [ 1.10243623 -1.17710795  8.23434168 ...  9.48315662  3.98476007
           0.67712601]
         [-6.48987062  1.56863287 -8.6753437  ... -7.3382564  -4.84010583
           1.77314861]
         [-2.38305014 -5.14370369 -4.37973189 ... -1.88885287 -0.02501935
           0.36819705]]
        
        
        self.epsilon_full[idx].mean(axis=0) - усреднение по 100
        (79884,)
        [ 0.75995355  0.57615031 -0.29691183 ... -0.11813105  0.284917
        -0.03189183]
        """


        # adaptive sigma
        # normalization
        if (self.sigma_alpha > 0):
            stdev_reward = 1.0
            if not self.rank_fitness:
                stdev_reward = reward.std()
            S = ((epsilon * epsilon - (sigma * sigma).reshape(1, self.num_params)) / sigma.reshape(1, self.num_params))
            reward_avg = (reward[:self.batch_size] + reward[self.batch_size:]) / 2.0
            rS = reward_avg - b
            delta_sigma = (np.dot(rS, S)) / (2 * self.batch_size * stdev_reward)

            # adjust sigma according to the adaptive sigma calculation
            # for stability, don't let sigma move more than 10% of orig value
            change_sigma = self.sigma_alpha * delta_sigma
            change_sigma = np.minimum(change_sigma, self.sigma_max_change * self.sigma)
            change_sigma = np.maximum(change_sigma, - self.sigma_max_change * self.sigma)
            self.sigma += change_sigma


        """
        self.sigma
        (79884,)
        [3.99951464 3.99477121 4.00161329 ... 4.0020081  3.99415325 4.00145723]
        """

        if (self.sigma_decay < 1):
            self.sigma[self.sigma > self.sigma_limit] *= self.sigma_decay


        if (self.learning_rate_decay < 1 and self.learning_rate > self.learning_rate_limit):
            self.learning_rate *= self.learning_rate_decay



    def current_param(self):
        return self.curr_best_mu

    def set_mu(self, mu):
        self.mu = np.array(mu)

    def best_param(self):
        return self.best_mu

    def result(self):  # return best params so far, along with historically best reward, curr reward, sigma
        return (self.best_mu, self.best_reward, self.curr_best_reward, self.sigma)



if __name__ == "__main__":

    optimizer = CMAES(num_params=4096)
