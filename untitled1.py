#!apt-get install -y xvfb python-opengl x11-utils > /dev/null 2>&1
#!pip3 install gym pyvirtualdisplay scikit-video > /dev/null 2>&1
#!pip3 install gym[atari]
#!pip3 install mitdeeplearning
#!pip3 install tqdm
import tensorflow as tf

import numpy as np
import base64, io, time, gym
import IPython, functools
import matplotlib.pyplot as plt

from tqdm import tqdm


import mitdeeplearning as mdl
def choose_action(model, observation):
  observation = np.expand_dims(observation, axis=0)

  logits = model.predict(observation)
  prob_weights = tf.nn.softmax(logits).numpy()
  
  action = np.random.choice(n_actions, size=1, p= prob_weights.flatten() )[0]

  return action
class Memory:
  def __init__(self): 
      self.clear()

  # Resets/restarts the memory buffer
  def clear(self): 
      self.observations = []
      self.actions = []
      self.rewards = []

  # Add observations, actions, rewards to memory
  def add_to_memory(self, new_observation, new_action, new_reward): 
      self.observations.append(new_observation)
      self.actions.append(new_action)
      self.rewards.append(new_reward)
memory = Memory()
def compute_loss(logits, actions, rewards): 
  neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)
  loss = tf.reduce_mean( neg_logprob * rewards)
  return loss
def train_step(model, optimizer, observations, actions, discounted_rewards):
  with tf.GradientTape() as tape:
      # Forward propagate through the agent network
      logits = model(observations)
      loss = compute_loss(logits, actions,discounted_rewards)
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
env = gym.make("Pong-v0", frameskip=5)
env.seed(1); # for reproducibility
print("Environment has observation space =", env.observation_space)
n_actions = env.action_space.n
print("Number of possible actions that the agent can choose from =", n_actions)
Conv2D = functools.partial(tf.keras.layers.Conv2D, padding='same', activation='relu')
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense
def create_pong_model():
  model = tf.keras.models.Sequential([
    Conv2D(filters=16, kernel_size=7, strides=4),
    Conv2D(filters=32, kernel_size=5, strides=2),
    Conv2D(filters=48, kernel_size=3, strides=2),
    Flatten(),
    Dense(units=64, activation='relu'),
    Dense(units=n_actions, activation=None)
  
  ])
  return model
pong_model = create_pong_model()
def discount_rewards(rewards, gamma=0.99): 
  discounted_rewards = np.zeros_like(rewards)
  R = 0
  for t in reversed(range(0, len(rewards))):
      # NEW: Reset the sum if the reward is not 0 (the game has ended!)
      if rewards[t] != 0:
        R = 0
      # update the total discounted reward as before
      R = R * gamma + rewards[t]
      discounted_rewards[t] = R     
  return normalize(discounted_rewards)
def normalize(x):
  x -= np.mean(x)
  x /= np.std(x)
  return x.astype(np.float32)
observation = env.reset()
for i in range(30):
  observation, _,_,_ = env.step(0)
learning_rate=1e-4
MAX_ITERS = 100 # increase the maximum number of episodes, since Pong is more complex!
pong_model = create_pong_model()
optimizer = tf.keras.optimizers.Adam(learning_rate)
smoothed_reward = mdl.util.LossHistory(smoothing_factor=0.9)
plotter = mdl.util.PeriodicPlotter(sec=5, xlabel='Iterations', ylabel='Rewards')
memory = Memory()
for i_episode in range(MAX_ITERS):
  plotter.plot(smoothed_reward.get())
  observation = env.reset()
  previous_frame = mdl.lab3.preprocess_pong(observation)
  while True:
      # Pre-process image 
      current_frame = mdl.lab3.preprocess_pong(observation)
      obs_change = current_frame- previous_frame
      action = choose_action(pong_model, obs_change)
      next_observation, reward, done, info = env.step(action)
      memory.add_to_memory(obs_change,action, reward)
      if done:
          total_reward = sum(memory.rewards)
          smoothed_reward.append( total_reward )
          train_step(pong_model, 
                     optimizer, 
                     observations = np.stack(memory.observations, 0), 
                     actions = np.array(memory.actions),
                     discounted_rewards = discount_rewards(memory.rewards))
          memory.clear()
          break

      observation = next_observation
      previous_frame = current_frame
saved_pong = mdl.lab3.save_video_of_model(
    pong_model, "Pong-v0", obs_diff=True, 
    pp_fn=mdl.lab3.preprocess_pong)
mdl.lab3.play_video(saved_pong)


