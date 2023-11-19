from typing import Optional
import torch
from core.train.collector import Collector
from core.algorithms.evaluator import Evaluator

class ASMRCollector(Collector):
    def __init__(self,
        evaluator: Evaluator,
        episode_memory_device: torch.device
    ) -> None:
        super().__init__(evaluator, episode_memory_device)

    def assign_rewards(self, terminated_episodes, terminated):
        episodes = []
        for episode in terminated_episodes:
            episode_with_rewards = []
            moves = len(episode)
            for (inputs, visits, legal_actions) in episode:
                episode_with_rewards.append((inputs, visits, torch.tensor(moves, dtype=torch.float32, requires_grad=False, device=inputs.device), legal_actions))
                moves -= 1
            episodes.append(episode_with_rewards)
        return episodes

    def postprocess(self, terminated_episodes):
        inputs, probs, rewards, legal_actions = zip(*terminated_episodes)
        return list(zip(inputs, probs, rewards, legal_actions))
