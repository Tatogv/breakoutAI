import gym
import random
import numpy as np
import tensorflow as tf
import time
import threading
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from keras import layers
from keras.models import Model
from keras.models import Sequential
from keras.models import clone_model
from keras.models import load_model
from keras.optimizers import RMSprop
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras import backend as K

ATARI_SHAPE = (84, 84, 4)  # input image size to model
ACTION_SIZE = 3
LEARNING_RATE = 0.00025

def huber_loss(y, q_value):
    error = K.abs(y - q_value)
    quadratic_part = K.clip(error, 0.0, 1.0)
    linear_part = error - quadratic_part
    loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)
    return loss

def atari_model():
    # With the functional API we need to define the inputs.
    frames_input = layers.Input(ATARI_SHAPE, name='frames')
    actions_input = layers.Input((ACTION_SIZE,), name='action_mask')

    # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
    normalized = layers.Lambda(lambda x: x / 255.0, name='normalization')(frames_input)

    # "The first hidden layer convolves 16 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity."
    conv_1 = layers.convolutional.Conv2D(
        16, (8, 8), strides=(4, 4), activation='relu'
    )(normalized)
    # "The second hidden layer convolves 32 4×4 filters with stride 2, again followed by a rectifier nonlinearity."
    conv_2 = layers.convolutional.Conv2D(
        32, (4, 4), strides=(2, 2), activation='relu'
    )(conv_1)
    # Flattening the second convolutional layer.hhhhhhhhh
    conv_flattened = layers.core.Flatten()(conv_2)
    # "The final hidden layer is fully-connected and consists of 256 rectifier units."
    hidden = layers.Dense(256, activation='relu')(conv_flattened)
    # "The output layer is a fully-connected linear layer with a single output for each valid action."
    output = layers.Dense(ACTION_SIZE)(hidden)
    # Finally, we multiply the output by the mask!
    filtered_output = layers.Multiply(name='QValue')([output, actions_input])

    model = Model(inputs=[frames_input, actions_input], outputs=filtered_output)
    model.summary()
    optimizer = RMSprop(lr=LEARNING_RATE, rho=0.95, epsilon=0.01)
    # model.compile(optimizer, loss='mse')
    # to changed model weights more slowly, uses MSE for low values and MAE(Mean Absolute Error) for large values
    model.compile(optimizer, loss=huber_loss)
    return model

class DQNA:
    def __init__(self):
        self.epochs = 1000 # Iteraciones
        self.observe = 500 # Timesteps a observar antes de comenzar el entrenamiento
        self.epsilon_step = 10000 # Cuadros para actualizar el epsilon
        self.refresh = 100 # Cuadros para actualizar el modelo
        self.memory = 200000 # Memoria maxima
        self.no_op = 30 # Numero de pasos antes de ejecutar el script
        self.regularizer = 0.01
        self.batch_size = 32 # Tamaño del lote de entrenamiento
        self.init_epsilon = 1.0
        self.final_epsilon = 0.1
        self.gamma = 0.99 # Tasa de descomposición de observaciones pasadas
        self.resume = False # Cargar modelo pasado
        self.render = True # Mostrar el juego
        self.path = './model.h5' # Path para guardar el modelo
        self.model = atari_model()
        self.target_model = atari_model()

    # get action from model using epsilon-greedy policy
    def get_action(self, history, epsilon, step, model):
        if np.random.rand() <= epsilon or step <= self.observe:
            return random.randrange(ACTION_SIZE)
        else:
            # print('A prediction was made!')
            q_value = model.predict([history, np.ones(ACTION_SIZE).reshape(1, ACTION_SIZE)])
            return np.argmax(q_value[0])


    # save sample <s,a,r,s'> to the replay memory
    def store_memory(self, memory, history, action, reward, next_history, dead):
        # print("The memory was appended")
        memory.append((history, action, reward, next_history, dead))


    def get_one_hot(self, targets, nb_classes):
        return np.eye(nb_classes)[np.array(targets).reshape(-1)]


    # train model by radom batch
    def train_memory_batch(self, memory, model):
        mini_batch = random.sample(memory, self.batch_size)
        history = np.zeros((self.batch_size, ATARI_SHAPE[0],
                            ATARI_SHAPE[1], ATARI_SHAPE[2]))
        next_history = np.zeros((self.batch_size, ATARI_SHAPE[0],
                                 ATARI_SHAPE[1], ATARI_SHAPE[2]))
        target = np.zeros((self.batch_size,))
        action, reward, dead = [], [], []

        for idx, val in enumerate(mini_batch):
            history[idx] = val[0]
            next_history[idx] = val[3]
            action.append(val[1])
            reward.append(val[2])
            dead.append(val[4])

        actions_mask = np.ones((self.batch_size, ACTION_SIZE))
        next_Q_values = self.model.predict([next_history, actions_mask])

        # like Q Learning, get maximum Q value at s'
        # But from target model
        for i in range(self.batch_size):
            if dead[i]:
                target[i] = -1
                # target[i] = reward[i]
            else:
                target[i] = reward[i] + self.gamma * np.amax(next_Q_values[i])

        action_one_hot = self.get_one_hot(action, ACTION_SIZE)
        target_one_hot = action_one_hot * target[:, None]

        h = model.fit(
            [history, action_one_hot], target_one_hot, epochs=1,
            batch_size=self.batch_size, verbose=0)
            #batch_size=self.batch_size, verbose=0, callbacks=[tb_callback])

        #if h.history['loss'][0] > 10.0:
        #    print('too large')

        return h.history['loss'][0]

def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe

def train():
        env = gym.make('BreakoutDeterministic-v4')
        agent = DQNA()
        # deque: Once a bounded length deque is full, when new items are added,
        # a corresponding number of items are discarded from the opposite end
        memory = deque(maxlen=agent.memory)
        episode_number = 0
        epsilon = agent.init_epsilon
        epsilon_decay = (agent.init_epsilon - agent.final_epsilon) / agent.epsilon_step
        global_step = 0

        if agent.resume:
            model = load_model(agent.path, custom_objects={'huber_loss': huber_loss})  # load model with customized loss func
            # Assume when we restore the model, the epsilon has already decreased to the final value
            epsilon = agent.final_epsilon
        else:
            model = agent.model

        model_target = clone_model(model)
        model_target.set_weights(model.get_weights())

        while episode_number < agent.epochs:

            done = False
            dead = False
            # 1 episode = 5 lives
            step, score, start_life = 0, 0, 5
            loss = 0.0
            observe = env.reset()

            # this is one of DeepMind's idea.
            # just do nothing at the start of episode to avoid sub-optimal
            for _ in range(random.randint(1, agent.no_op)):
                observe, _, _, _ = env.step(1)
            # At start of episode, there is no preceding frame
            # So just copy initial states to make history
            state = pre_processing(observe)
            history = np.stack((state, state, state, state), axis=2)
            history = np.reshape([history], (1, 84, 84, 4))

            while not done:
                if agent.render:
                    env.render()
                    time.sleep(0.01)

                # get action for the current history and go one step in environment
                action = agent.get_action(history, epsilon, global_step, model_target)
                # change action to real_action
                real_action = action + 1

                # scale down epsilon, the epsilon only begin to decrease after observe steps
                if epsilon > agent.final_epsilon and global_step > agent.observe:
                    epsilon -= epsilon_decay

                observe, reward, done, info = env.step(real_action)
                # pre-process the observation --> history
                next_state = pre_processing(observe)
                next_state = np.reshape([next_state], (1, 84, 84, 1))
                next_history = np.append(next_state, history[:, :, :, :3], axis=3)

                # if the agent missed ball, agent is dead --> episode is not over
                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']

                # TODO: may be we should give negative reward if miss ball (dead)
                # reward = np.clip(reward, -1., 1.)  # clip here is not correct

                # save the statue to memory, each replay takes 2 * (84*84*4) bytes = 56448 B = 55.125 KB
                agent.store_memory(memory, history, action, reward, next_history, dead)  #

                # check if the memory is ready for training
                if global_step > agent.observe:
                    loss = loss + agent.train_memory_batch(memory, model)
                    # if loss > 100.0:
                    #    print(loss)
                    if global_step % agent.refresh == 0:  # update the target model
                        # print('The model was refreshed. Global_step: {}, rrefresh: {}'.format(global_step, agent.refresh))
                        model_target.set_weights(model.get_weights())

                score += reward

                # If agent is dead, set the flag back to false, but keep the history unchanged,
                # to avoid to see the ball up in the sky
                if dead:
                    dead = False
                else:
                    history = next_history

                #print("step: ", global_step)
                global_step += 1
                step += 1

                if done:
                    if global_step <= agent.observe:
                        state = "observe"
                    elif agent.observe < global_step <= agent.observe + agent.epsilon_step:
                        state = "explore"
                    else:
                        state = "train"
                    print('state: {}, episode: {}, score: {}, global_step: {}, avg loss: {}, step: {}, memory length: {}, epsilon: {}'
                          .format(state, episode_number, score, global_step, loss / float(step), step, len(memory), epsilon))

                    if episode_number % 100 == 0 or (episode_number + 1) == agent.epochs:
                    #if episode_number % 1 == 0 or (episode_number + 1) == FLAGS.num_episode:  # debug
                        print("The model was saved")
                        model.save(agent.path)

                    episode_number += 1

def test():
    env = gym.make('BreakoutDeterministic-v4')
    agent = DQNA()
    episode_number = 0
    epsilon = 0.001
    global_step = agent.observe + 1
    # model = load_model(FLAGS.restore_file_path)
    model = load_model(agent.path, custom_objects={'huber_loss': huber_loss})  # load model with customized loss func

    # test how to deep copy a model

    while episode_number < agent.epochs:
        done = False
        dead = False
        # 1 episode = 5 lives
        score, start_life = 0, 5
        observe = env.reset()

        observe, _, _, _ = env.step(1)
        # At start of episode, there is no preceding frame
        # So just copy initial states to make history
        state = pre_processing(observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        while not done:
            env.render()
            time.sleep(0.1)

            # get action for the current history and go one step in environment
            action = agent.get_action(history, epsilon, global_step, model)
            # change action to real_action
            real_action = action + 1

            observe, reward, done, info = env.step(real_action)
            # pre-process the observation --> history
            next_state = pre_processing(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            # if the agent missed ball, agent is dead --> episode is not over
            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            # TODO: may be we should give negative reward if miss ball (dead)
            reward = np.clip(reward, -1., 1.)

            score += reward

            # If agent is dead, set the flag back to false, but keep the history unchanged,
            # to avoid to see the ball up in the sky
            if dead:
                dead = False
            else:
                history = next_history

            # print("step: ", global_step)
            global_step += 1

            if done:
                episode_number += 1
                print('episode: {}, score: {}'.format(episode_number, score))


def main(argv=None):
    #train(1)
    tr_gen = train()
    model.fit_generator(generator=tr_gen, steps_per_epoch=20, max_queue_size=10,
    workers=3, use_multiprocessing=True)
    #test()
    #threads = []
    #for i in range(4):
    #    t = threading.Thread(target=train, args=(i,))
    #    threads.append(t)
    #    t.start()




if __name__ == '__main__':
    #tf.app.run()
    main()
