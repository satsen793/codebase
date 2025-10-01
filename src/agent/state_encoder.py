# state_encoder.py
# Shared encoder for student state
import numpy as np


class StateEncoder:
    def __init__(self, L, error_dim=32, hist_dim=64, context_dim=6):
        self.L = L  # Number of LOs
        self.error_dim = error_dim
        self.hist_dim = hist_dim
        self.context_dim = context_dim

    def encode(self, state):
        # state: dict with all required fields
        # Extract features (use zeros for missing fields as placeholder)
        mastery_vec = np.array(state.get('mastery_vec', np.zeros(self.L)))
        mastery_vel = np.array(state.get('mastery_vel', np.zeros(3*self.L)))
        latents = np.array(state.get('latents', np.zeros(7)))
        uncertainty = np.array(state.get('uncertainty', np.zeros(7)))
        error_embed = np.array(state.get('error_embed', np.zeros(self.error_dim)))
        hist_embed = np.array(state.get('hist_embed', np.zeros(self.hist_dim)))
        # Curriculum position: current LO index, unlock state (L,)
        c_pos = np.array(state.get('curriculum_pos', [0]))
        unlocks = np.array(state.get('unlocks', np.zeros(self.L)))
        jump_vel = np.array([state.get('jump_vel', 0)])
        context = np.array(state.get('context', np.zeros(self.context_dim)))
        # Concatenate all
        return np.concatenate([
            mastery_vec, mastery_vel, latents, uncertainty, error_embed, hist_embed,
            c_pos, unlocks, jump_vel, context
        ])
