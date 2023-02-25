import numpy as np

from typing import List, Any
from tqdm import trange


class TableCrossEntopyMethod:
    def __init__(
        self,
        env: Any
    ) -> None:
        self.env = env
        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.n
        self.policy = self.initialize_policy(self.n_states, self.n_actions)

    def generate_session(self, policy: np.ndarray, t_max: int = 10**4):
        states, actions = [], []
        total_reward = 0.
        state = self.env.reset()

        for _ in range(t_max):
            action = np.random.choice(
                np.arange(policy.shape[1]),
                p=policy[state]
            )
            new_state, r, done, info = self.env.step(action)
            states.append(state)
            actions.append(action)
            total_reward += r

            state = new_state
            if done:
                break

        return states, actions, total_reward

    def initialize_policy(self, n_states: int, n_actions: int):
        policy = np.ones((n_states, n_actions))
        policy /= n_actions
        return policy

    def select_elites(
        self,
        states_batch: List[List[Any]],
        actions_batch: List[List[Any]],
        rewards_batch: List[Any],
        percentile
    ):
        reward_threshold = np.percentile(rewards_batch, percentile)
        elite_states = []
        elite_actions = []
        for batch_idx in range(len(rewards_batch)):
            if rewards_batch[batch_idx] >= reward_threshold:
                elite_states.extend(states_batch[batch_idx])
                elite_actions.extend(actions_batch[batch_idx])
        return elite_states, elite_actions

    def get_new_policy(self, elite_states, elite_actions):
        new_policy = np.zeros([self.n_states, self.n_actions])
        for i in range(len(elite_states)):
            new_policy[elite_states[i], elite_actions[i]] += 1

        for i in range(self.n_states):
            if np.sum(new_policy[i]) > 0:
                new_policy[i] /= new_policy[i].sum()
            else:
                new_policy[i] = np.ones(self.n_actions) / self.n_actions
        return new_policy

    def fit(
        self,
        num_epoch: int,
        n_sessions: int,
        percentile: int,
        learning_rate: float
    ):
        rewards = []
        current_policy = self.policy
        with trange(num_epoch) as pbar:
            for _ in pbar:
                sessions = [
                    self.generate_session(current_policy)
                    for _ in range(n_sessions)
                ]
                states_batch, actions_batch, rewards_batch = zip(*sessions)
                elite_states, elite_actions = self.select_elites(
                    states_batch,
                    actions_batch,
                    rewards_batch,
                    percentile
                )
                new_policy = self.get_new_policy(elite_states, elite_actions)
                current_policy = learning_rate * new_policy \
                    + (1 - learning_rate) * current_policy

                rewards.append(np.mean(rewards_batch))
                pbar.set_postfix_str(f"mean reward: {np.mean(rewards_batch)}")
        self.policy = current_policy
        return rewards
