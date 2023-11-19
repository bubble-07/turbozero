
from typing import Tuple
import torch

#TODO: Find a way to restore these as torch-scripts
#Currently there's the issue of all these dang extra parameters
#could probably use something like https://pytorch.org/docs/stable/generated/torch.jit.script.html

def push_actions(states, actions, total_num_matrices, matrix_dim, max_policy_dim_size) -> (torch.Tensor, torch.Tensor):
    max_policy_dim_size_sq = max_policy_dim_size ** 2
    # Find all the "terminate" actions
    terminate_actions = torch.eq(actions, max_policy_dim_size_sq)
    continue_actions = ~terminate_actions
    actions_rows, actions_cols = actions // max_policy_dim_size, actions % max_policy_dim_size
    # Put terminate actions back in-bounds. We'll still compute matrix
    # products for them, but eh, whatever.
    actions_rows *= continue_actions
    actions_cols *= continue_actions
    # Select the matrices from the states
    matrix_states = states[:, :-2]
    matrix_states = matrix_states.reshape(-1, total_num_matrices, matrix_dim, matrix_dim)

    actions_rows = actions_rows.reshape(-1, 1, 1, 1)
    actions_rows = actions_rows.expand(-1, 1, matrix_dim, matrix_dim)
    left_matrices = torch.gather(matrix_states, 1, actions_rows)
    left_matrices = torch.squeeze(left_matrices, dim=1)

    actions_cols = actions_cols.reshape(-1, 1, 1, 1)
    actions_cols = actions_cols.expand(-1, 1, matrix_dim, matrix_dim)
    right_matrices = torch.gather(matrix_states, 1, actions_cols)
    right_matrices = torch.squeeze(right_matrices, dim=1)

    # This is now of shape parallel_envs * matrix_dim * matrix_dim
    multiplied_matrices = torch.matmul(left_matrices, right_matrices)
    multiplied_matrices = multiplied_matrices.reshape(-1, 1, matrix_dim, matrix_dim)

    # Now append the newly-multiplied matrices
    current_num_matrices = states[:, -2]
    current_num_matrices = current_num_matrices.type(torch.int64)
    current_num_matrices = current_num_matrices.reshape(-1, 1, 1, 1)
    current_num_matrices = current_num_matrices.expand(-1, 1, matrix_dim, matrix_dim)
    matrix_states.scatter_(1, current_num_matrices, multiplied_matrices)

    # After appending the newly-multiplied matrices, update the states
    current_num_matrices = states[:, -2] + 1
    num_turns_taken = states[:, -1] + 1

    current_num_matrices = current_num_matrices.reshape(-1, 1)
    num_turns_taken = num_turns_taken.reshape(-1, 1)
    matrix_states = matrix_states.reshape(-1, total_num_matrices * matrix_dim * matrix_dim)

    states = torch.cat((matrix_states, current_num_matrices, num_turns_taken), dim=1)

    # Update the state in response to an action
    return (states, terminate_actions)

def get_legal_actions(states, max_policy_dim_size):
    num_parallel_envs = states.size(dim=0)
    # Get the current number of matrices for each environment
    current_num_matrices = states[:, -2]
    # Construct 1d index masks for valid single-indices for every env
    index_masks = leading_ones(max_policy_dim_size, current_num_matrices - 1)

    # Inflate the 1d index masks to 2d index-pair masks
    index_pair_masks = torch.logical_and(
            index_masks.reshape(-1, max_policy_dim_size, 1),
            index_masks.reshape(-1, 1, max_policy_dim_size))

    max_policy_dim_size_sq = max_policy_dim_size ** 2

    # Flatten the index-pair masks
    index_pair_masks = index_pair_masks.reshape(-1, max_policy_dim_size_sq)

    # Incorporate them into a result tensor, which additionally
    # has the last index ("terminate") selected.
    result = torch.full((num_parallel_envs, max_policy_dim_size_sq + 1), True, device=states.get_device())
    result[:, :max_policy_dim_size_sq] = index_pair_masks
    return result

# Assumes that the state is terminal
def get_rewards(states: torch.Tensor, total_num_matrices: int, matrix_dim: int, discount_factor: float) -> torch.Tensor:
    num_turns_taken = states[:, -1]
    current_num_matrices = states[:, -2]
    # Select the matrices from the states
    matrix_states = states[:, :-2]
    matrix_states = matrix_states.reshape(-1, total_num_matrices, matrix_dim, matrix_dim)

    # Extract target matrices
    target_matrices = matrix_states[:, -1, :, :]
    target_matrices = target_matrices.reshape(-1, 1, matrix_dim, matrix_dim)

    # Pull out the matrix sets and insert +inf in positions
    # which don't correspond to actual matrices in each set.
    matrix_sets = matrix_states[:, :-1, :, :]
    matrix_sets_mask = leading_ones(total_num_matrices - 1, current_num_matrices - 1)
    matrix_sets_mask = ~matrix_sets_mask
    matrix_sets_mask = matrix_sets_mask.reshape(-1, total_num_matrices - 1, 1, 1)
    matrix_sets_mask = matrix_sets_mask.expand_as(matrix_sets)
    matrix_sets = matrix_sets.masked_fill(matrix_sets_mask, float('inf'))

    # Compute squared distance between all matrices and the target
    diffs = target_matrices - matrix_sets
    sq_diffs = diffs * diffs
    sq_dists = torch.sum(sq_diffs, dim=[2, 3])
    # Find the min dist for each env
    min_sq_dists, _ = torch.min(sq_dists, dim=1)

    # Discount by the number of turns taken
    loss = torch.log(min_sq_dists)
    # We do this twice to get to negative largest float values for nans
    loss = torch.nan_to_num(loss, nan=float('-inf'))
    loss = torch.nan_to_num(loss)
    loss = discount_factor * num_turns_taken + loss
    return -loss

# Returns a matrix with a binary mask of the form
# 1 1 1 0 0 0 0
# 1 0 0 0 0 0 0
# 1 1 0 0 0 0 0
# where the number of leading ones in each row is given
# by a passed-in `indices` tensor
def leading_ones(num_cols: int, index_vector: torch.Tensor) -> torch.Tensor:
    index_vector_len = index_vector.size(dim=0)
    index_vector = index_vector.reshape((-1, 1))
    counting_vector = torch.arange(num_cols, dtype=torch.uint8, device=index_vector.get_device())
    counting_matrix = counting_vector.reshape((1, num_cols))
    counting_matrix = counting_matrix.expand(index_vector_len, -1)
    return counting_matrix <= index_vector


# Generates a collection of random matrices + game-states
def generate_random_games(parallel_envs: int, matrix_dim: int,
        min_initial_set_size: int, max_initial_set_size: int,
        normal_std_dev: float, total_num_matrices: int,
        device: torch.device) -> torch.Tensor:
    # First, generate a bunch o' random matrices with the appropriate sizing
    matrices = torch.randn(parallel_envs, total_num_matrices, matrix_dim * matrix_dim, device=device)
    matrices *= normal_std_dev
    # Then, generate the initial set size for every environment
    initial_set_sizes = torch.randint(min_initial_set_size, max_initial_set_size + 1,
                                      (parallel_envs,), device=device)
    # Generate a mask matrix for the matrices of each size to keep,
    # which includes the initial set [leading indices] and the target
    # matrices [final index]
    mask = leading_ones(total_num_matrices, initial_set_sizes)
    mask = torch.roll(mask,
                      -1, 1)
    mask = mask.reshape((parallel_envs, -1, 1))

    # Mask non-selected matrices out [become zeros]
    matrices *= mask

    # Assemble the result
    matrix_total_dims = total_num_matrices * matrix_dim * matrix_dim
    matrices = matrices.reshape((parallel_envs, matrix_total_dims))

    current_num_matrices = initial_set_sizes.reshape((-1, 1))
    # Start out with zero turns taken
    num_turns_taken = torch.zeros_like(current_num_matrices)

    result = torch.cat((matrices, current_num_matrices, num_turns_taken), dim=1)

    return result
