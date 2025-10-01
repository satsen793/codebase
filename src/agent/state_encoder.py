# state_encoder.py
# Shared encoder for student state
import numpy as np

class StateEncoder:
    def __init__(self, state_dim):
        self.state_dim = state_dim

    def encode(self, mastery_vec, latents, history, context):
        # Concatenate all state features into a single vector
        # mastery_vec: np.array (L,)
        # latents: np.array (7,)
        # history: np.array (history_dim,)
        # context: np.array (context_dim,)
        return np.concatenate([mastery_vec, latents, history, context])
