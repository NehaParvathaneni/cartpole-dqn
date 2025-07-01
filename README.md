# Deep Reinforcement Learning - DQN with CartPole in PyTorch

## 📌 About
This project implements a **Deep Q-Network (DQN)** agent using **PyTorch** to solve the **CartPole-v1** environment from OpenAI Gym.  
The objective is for the agent to learn to balance a pole on a moving cart for as long as possible by taking discrete actions (move left or right).

---

## 🚀 Model Architecture
The DQN agent uses a simple feedforward neural network:
- **Input:** 4 features (cart position, cart velocity, pole angle, pole angular velocity)
- **Hidden Layers:** 
  - Layer 1: 128 neurons + ReLU
  - Dropout: 0.7
  - Layer 2: 64 neurons + ReLU
- **Output:** Q-values for two possible actions (left or right)

The agent is trained using the Mean Squared Error (MSE) loss between predicted and target Q-values, with experience replay.

---

## 🏗 Code Structure
- `DQN_cartpole.py` : Main Python script containing
  - Model definition
  - Training loop
  - Evaluation / Running loop
  - Plotting of training metrics
- Outputs generated during training:
  - Plots of Loss and Rewards over Episodes (`plot_loss.png`, `plot_rewards.png`)
  - Trained model weights (`model.pt`)

---

## ⚙️ How to Run
The project uses **command-line arguments** for flexibility.

### ✅ Key arguments
| Argument    | Description                                       | Default   |
|-------------|---------------------------------------------------|-----------|
| `--device`  | CUDA device if available                          | `0`       |
| `--verbose` | Print detailed progress logs                      | `1`       |
| `--load`    | Load a previously trained model                   | `False`   |
| `--save`    | Save the trained model weights                    | `False`   |
| `--plot`    | Plot and save training graphs                     | `True`    |
| `--model`   | Path to load or save model weights                | `'reinforce_cartpole/model.pt'` |
| `--runtype` | `'train_run'`, `'train'`, or `'run'`              | `'train_run'` |
| `--lr`      | Learning rate                                     | `0.001`   |
| `--episodes`| Number of training episodes                       | `500`     |
| `--gamma`   | Discount factor for future rewards                | `0.99`    |

---

### 🏃 Examples

#### 🔥 Train and Run (500 episodes), save the model
python DQN_cartpole.py --save=True --episodes=500
#### 🛠 Train Only (no rendering, for servers)
python DQN_cartpole.py --runtype=train --episodes=500

#### 🚀 Run only using a saved model
python DQN_cartpole.py --runtype=run --load=True --model=path/to/model.pt

---

## 🎯 Environment
- **CartPole-v1** from [OpenAI Gym](https://gym.openai.com/envs/CartPole-v1/)

### ⚙ State Space (4 features)
1. Cart Position
2. Cart Velocity
3. Pole Angle
4. Pole Angular Velocity

### 🎮 Action Space (Discrete)
- `0`: Move cart to the left
- `1`: Move cart to the right

---

## 🖼 Outputs
- 📈 `plot_loss.png` : Shows how the loss changes over episodes.
- 🏆 `plot_rewards.png` : Shows episode rewards and moving average.
- 🧠 `model.pt` : Saved trained model weights.

These are saved under automatically created timestamped folders in `dqn_cartpole/`.

---

## 📚 References
- [PyTorch Q-learning Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [Towards Data Science - Actor Critic Methods](https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f)

---

✅ **This repository focuses only on DQN with CartPole.**  
Future work may explore Policy Gradients (REINFORCE) or Actor-Critic (A2C).

---

### 🚀 Author
> Neha Parvathaneni
