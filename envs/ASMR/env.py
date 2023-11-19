
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from core.env import Env, EnvConfig
from .torchscripts import push_actions, get_legal_actions, get_rewards, generate_random_games

@dataclass
class ASMREnvConfig(EnvConfig):
    # The dimension of one of the matrices in the set
    # (matrices are matrix_dim x matrix_dim)
    matrix_dim: int = 2
    # The minimum number of matrices in the starting set
    min_initial_set_size: int = 3
    # The maximum number of matrices in the starting set
    max_initial_set_size: int = 7
    # The standard deviation of the normal distribution which
    # is used to generate elements of the initial and
    # target matrices
    normal_std_dev: float = 0.1
    # The maximum number of turns to take
    max_num_turns: int = 6
    # The discount factor
    discount_factor: float = 0.01

class ASMREnv(Env):
    def __init__(self,
        parallel_envs: int,
        config: ASMREnvConfig,
        device: torch.device,
        debug=False
    ) -> None:
        self.matrix_dim = config.matrix_dim
        self.min_initial_set_size = config.min_initial_set_size
        self.max_initial_set_size = config.max_initial_set_size
        self.normal_std_dev = config.normal_std_dev
        self.max_num_turns = config.max_num_turns
        self.discount_factor = config.discount_factor

        # Derived information

        self.max_ending_set_size = self.max_num_turns + self.max_initial_set_size
        self.max_policy_dim_size = self.max_ending_set_size - 1

        # The total number of matrices, including the active
        # set and the target matrix
        self.total_num_matrices = self.max_ending_set_size + 1

        # The size of the state vector
        self.state_vector_size = self.total_num_matrices * self.matrix_dim * self.matrix_dim
        self.state_vector_size += 1 # for the current set size tracker
        self.state_vector_size += 1 # for the current turn counter

        # The policy encompasses every index-pair, together with a
        # probability of stopping
        self.policy_vector_size = (self.max_policy_dim_size ** 2) + 1

        super().__init__(
            parallel_envs=parallel_envs,
            config=config,
            device=device,
            num_players=1,
            state_shape=torch.Size((self.state_vector_size, )),
            policy_shape=torch.Size((self.policy_vector_size,)),
            value_shape=torch.Size((1,)),
            debug=debug
        )

        if self.debug:
            self.get_legal_actions_ts = get_legal_actions
            self.push_actions_ts = push_actions
            self.get_rewards_ts = get_rewards
            self.generate_random_games_ts = generate_random_games
        else:
            self.get_legal_actions_ts = torch.jit.trace(get_legal_actions, ( # type: ignore
                self.states,
                self.max_policy_dim_size,
            ))

            self.push_actions_ts = torch.jit.trace(push_actions, ( # type: ignore
                self.states,
                torch.zeros((self.parallel_envs, ), dtype=torch.int64, device=device)
            ))

            self.get_rewards_ts = torch.jit.trace(get_rewards, ( # type: ignore
                self.states,
                self.discount_factor
            ))
            self.generate_random_games_ts = torch.jit.trace(generate_random_games, ( # type: ignore
                self.parallel_envs,
                self.matrix_dim,
                self.min_initial_set_size,
                self.max_initial_set_size,
                self.normal_std_dev,
                self.total_num_matrices
            ))

        self.saved_states = self.states.clone()

    def reset(self, seed: Optional[int] = None) -> int:
        self.states.zero_()
        self.terminated.fill_(True)
        return self.reset_terminated_states(seed)

    def reset_terminated_states(self, seed: Optional[int] = None) -> int:
        if seed is not None:
            torch.manual_seed(seed)
        else:
            seed = 0
        # Zeros the states which are terminated
        self.states *= torch.logical_not(self.terminated).view(self.parallel_envs, 1)

        # Find the total number of terminated states
        num_terminated_states = torch.sum(self.terminated)

        if num_terminated_states > 0:
            # Re-initialize the terminated states
            random_games = self.generate_random_games_ts(
                num_terminated_states,
                self.matrix_dim,
                self.min_initial_set_size,
                self.max_initial_set_size,
                self.normal_std_dev,
                self.total_num_matrices,
                self.states.get_device()
            )

            self.states[self.terminated] = random_games

        # Clears the terminated mask, since presumably, all states have
        # been correctly reset
        self.terminated.zero_()
        return seed

    def next_turn(self):
        # Apply updates to the state for the next turn
        # I think (?) nothing really needs to be done here
        return

    def get_rewards(self, player_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.get_rewards_ts(self.states, self.total_num_matrices, self.matrix_dim, self.discount_factor)

    def update_terminated(self) -> None:
        self.terminated = self.is_terminal()

    def is_terminal(self):
        # The leaf nodes [not counting the ordinary nodes with the
        # "terminate" action taken.]
        out_of_turns = self.states[:, -1] >= self.max_num_turns
        return torch.logical_or(self.terminated, out_of_turns)

    def get_legal_actions(self) -> torch.Tensor:
        # Gets the legal actions for the current state
        return self.get_legal_actions_ts(self.states, self.max_policy_dim_size) # type: ignore

    def push_actions(self, actions) -> None:
        # Updates the state in response to an action
        self.states, terminate_actions = self.push_actions_ts(self.states, actions,
            self.total_num_matrices, self.matrix_dim,
            self.max_policy_dim_size) # type: ignore
        self.terminated = torch.logical_or(self.terminated, terminate_actions)

    def save_node(self) -> torch.Tensor:
        return self.states.clone()

    def load_node(self, load_envs: torch.Tensor, saved: torch.Tensor):
        load_envs_expnd = load_envs.view(self.parallel_envs, 1)
        self.states = saved.clone() * load_envs_expnd + self.states * (~load_envs_expnd)
        self.update_terminated()

    def print_state(self, action=None) -> None:
        assert self.parallel_envs == 1
        self.states, action,
        print("testing")
