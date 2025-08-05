import random

import numpy as np
import torch


class MolDQN(torch.nn.Module):
    def __init__(
        self,
        num_input: int,
        num_output: int,
        hidden_state_neurons: list | None = None,
    ):
        super(MolDQN, self).__init__()

        if hidden_state_neurons is None:
            hidden_state_neurons = [1024, 512, 128, 32]
        self.layers = torch.nn.ModuleList([])

        hs = hidden_state_neurons

        N = len(hs)

        for i in range(N - 1):
            h, h_next = hs[i], hs[i + 1]
            dim_input = num_input if i == 0 else h

            self.layers.append(torch.nn.Linear(dim_input, h_next))

        self.out = torch.nn.Linear(hs[-1], num_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.nn.functional.relu(layer(x))
        x = self.out(x)
        return x

class ReplayMemory:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = args
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class MegAgent:
    def __init__(
        self,
        num_input: int = 5,
        num_output: int = 10,
        lr: float = 1e-3,
        replay_buffer_size: int = 10,
        device: str | None = None
    ):
        self.num_input = num_input
        self.num_output = num_output
        self.replay_buffer_size = replay_buffer_size

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.dqn, self.target_dqn = (
            MolDQN(num_input, num_output).to(self.device),
            MolDQN(num_input, num_output).to(self.device),
        )

        for p in self.target_dqn.parameters():
            p.requires_grad = False

        self.replay_buffer = ReplayMemory(replay_buffer_size)

        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=lr)

    def action_step(self, observations, epsilon_threshold):
        if np.random.uniform() < epsilon_threshold:
            action = np.random.randint(0, observations.shape[0])
        else:
            q_value = self.dqn(observations.to(self.device)).cpu()
            action = torch.argmax(q_value).detach().numpy()

        return action

    def train_step(self, batch_size: int, gamma: float, polyak: float) -> torch.Tensor:
        experience = self.replay_buffer.sample(batch_size)
        states_ = torch.stack([S for S, *_ in experience])
        next_states_ = [S for *_, S, _ in experience]
        q = self.dqn(states_)
        q_target = torch.stack(
            [self.target_dqn(S).max(dim=0).values.detach() for S in next_states_]
        )

        rewards = (
            torch.stack([R for _, R, *_ in experience])
            .reshape((1, batch_size))
            .to(self.device)
        )
        dones = (
            torch.tensor([D for *_, D in experience])
            .reshape((1, batch_size))
            .to(self.device)
        )

        q_target = rewards + gamma * (1 - dones) * q_target
        td_target = q - q_target

        loss = torch.where(
            torch.abs(td_target) < 1.0,
            0.5 * td_target * td_target,
            1.0 * (torch.abs(td_target) - 0.5),
        ).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            for param, target_param in zip(
                self.dqn.parameters(), self.target_dqn.parameters()
            ):
                target_param.data.mul_(polyak)
                target_param.data.add_((1 - polyak) * param.data)

        return loss