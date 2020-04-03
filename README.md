# All about Deep Reinforcement Learning

I work in TensorFlow2 Framework

## Environments
- Snake_v1
    - Observation space: (RGB array or Extracted features)
    - Action space: Discrete(3), [Go Straight / Turn Left / Turn Right]
    - Reward scheme: (Dense or Sparse)
    
![PPO agent after 50000 episodes](https://github.com/sunghoonhong/DeepRL_tf1/blob/master/tensorflow1/snake_feature/gifs/ppo%20after%2050000%20episode.gif)
    
- Snake_v2
    - Observation space: (RGB array)
    - Action space: Box(2), [Speed(0.0 ~ 1.0), Angle(-1.0 ~ 1.0)]
    - Reward scheme: (Dense or Sparse)


## RL Agents
- Randomly
- TODOs
    - PPO (Proximal Policy Optimization)
    - DDPG (Deep Deterministic Policy Gradient)
    - A3C (Asynchronous Advantage Actor Critic)
    - DQN (Deep Q-Network)

## IL Agents
- TODOs
    - BC (Behavior Cloning)
    - GAIL (Generative Adversarial Imitation Learning)
    - VAIL (Variational GAIL)
    - MAIL (Mature GAIL)
    - DI-GAIL (Directed-Info GAIL)
    - DI-MAIL (Mature DI-GAIL)
