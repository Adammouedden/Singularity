import random
import time
from typing import Any
import numpy as np
import sys
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import hashlib
from schemas.schemas import CandidateActions, ActionCandidate
from agents.structs import FrameData, GameAction, GameState

"""
Action Learner - Learns to predict which actions cause frame changes for efficient exploration.

Architecture:
- CNN with 16 input channels (one-hot encoded colors 0-15)
- Two-headed output: action head (6 logits for ACTION1-ACTION6) + click head (4096 logits for 64x64 positions)
- Binary classification: predicts if each action will change the current frame

Training:
- Supervised learning on (state, action) -> frame_changed labels
- Action head always trained, click head only trained when ACTION6 is selected
- Experience buffer cleared when score increases (new level)

Sampling:
- Hierarchical: first sample action type using softmax over action logits
- If ACTION6 selected, then sample click position using softmax over click logits
- Stochastic exploration biased toward actions predicted to cause changes

This enables more efficient exploration than random, especially for coordinate-based actions.
"""

class ActionModel(nn.Module):
    """CNN that predicts which actions will result in new frames with shared conv backbone."""
    
    def __init__(self, input_channels=16, grid_size=64):
        super().__init__()
        self.grid_size = grid_size
        self.num_action_types = 5  # ACTION1-ACTION5
        
        # Shared convolutional backbone
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Action head - preserve spatial information
        self.action_pool = nn.MaxPool2d(4, 4)  # Reduce to 16x16 like coordinates
        action_flattened_size = 256 * 16 * 16  # 65,536
        self.action_fc = nn.Linear(action_flattened_size, 512)
        self.action_head = nn.Linear(512, self.num_action_types)
        
        # Coordinate head - enhanced spatial reasoning (64x64 action space)
        self.coord_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)  # Spatial processing
        self.coord_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)   # More spatial processing
        self.coord_conv3 = nn.Conv2d(64, 32, kernel_size=1)               # Channel reduction
        self.coord_conv4 = nn.Conv2d(32, 1, kernel_size=1)                # Final logits
        
        # No special initialization - let coordinates start naturally
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        
        # Shared convolutional backbone
        x = F.relu(self.conv1(x))  # (batch, 32, 64, 64)
        x = F.relu(self.conv2(x))  # (batch, 64, 64, 64)
        x = F.relu(self.conv3(x))  # (batch, 128, 64, 64)
        conv_features = F.relu(self.conv4(x))  # (batch, 256, 64, 64)
        
        # Action head - preserve spatial information (5 actions)
        action_features = self.action_pool(conv_features)  # (batch, 256, 16, 16)
        action_features = action_features.view(action_features.size(0), -1)  # (batch, 65536)
        action_features = F.relu(self.action_fc(action_features))  # (batch, 512)
        action_features = self.dropout(action_features)
        action_logits = self.action_head(action_features)  # (batch, 5)
        
        # Coordinate head - enhanced 64x64 action space
        coord_features = F.relu(self.coord_conv1(conv_features))  # (batch, 128, 64, 64)
        coord_features = F.relu(self.coord_conv2(coord_features))  # (batch, 64, 64, 64)
        coord_features = F.relu(self.coord_conv3(coord_features))  # (batch, 32, 64, 64)
        coord_logits = self.coord_conv4(coord_features)            # (batch, 1, 64, 64)
        coord_logits = coord_logits.view(coord_logits.size(0), -1) # (batch, 4096)
        
        # Return combined logits: [5 action logits, 4096 coordinate logits]
        combined_logits = torch.cat([action_logits, coord_logits], dim=1)  # (batch, 5 + 4096)
        
        return combined_logits


class CNNWorldModel():
    """Agent using action model to predict which actions lead to new frames."""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed % (2**32 - 1))
        self.start_time = time.time()
        
        # No max action limit.
        self.MAX_ACTIONS = float('inf')
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Action agent using device: {self.device}")
                
        self.current_score = -1
        
        # Setup logger
        self.logger = logging.getLogger(f"ActionAgent_{self.game_id}")
        
        # Initialize action model
        self.grid_size = 64
        self.num_coordinates = self.grid_size * self.grid_size
        self.num_colours = 16
        self.action_model = None
        self.optimizer = None

        # Experience buffer for training with uniqueness tracking
        self.experience_buffer = deque(maxlen=200000)
        self.experience_hashes = set()  # Track unique frame+action combinations
        self.batch_size = 64
        # TODO: Update this to a smaller value?
        self.train_frequency = 5  # Train every N actions
        
        # Track previous state/action for experience creation
        self.prev_frame = None
        self.prev_action_idx = None
        
        # Action mapping: ACTION1-ACTION5
        self.action_list = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, 
                           GameAction.ACTION4, GameAction.ACTION5]
        
        # Action semantics for proposer
        self.action_semantics = {
            "ACTION1": "up",
            "ACTION2": "down",
            "ACTION3": "left",
            "ACTION4": "right",
            "ACTION5": "special/interact/select/rotate/execute",
            "ACTION6": "coordinate action requiring x,y in [0,63]",
            "ACTION7": "undo",
        }
        
        print(f"Action agent initialized for game_id: {self.game_id}")
        self.logger.info(f"Action agent initialized for game_id: {self.game_id}")

    def propose_actions(
        self, frame: FrameData, available_actions: list
    ) -> CandidateActions:
        """
        Generate ranked candidate actions from CNN predictions.
        
        Args:
            frame: Current FrameData object
            available_actions: List of ActionCandidate objects representing available actions
            
        Returns:
            CandidateActions: Ranked list of candidate next actions with probabilities
        """
        # get the action names
        available_action_strings = {ac.action for ac in available_actions}
        
        # Force ACTION7 to have 0 probability by excluding it from available set
        # (ACTION7 is not in CNN output, but we explicitly ensure it in the filtering)
        available_action_strings.discard("ACTION7")
        
        # Convert frame to tensor
        current_frame = self._frame_to_tensor(frame)
        
        # Get CNN predictions -> first 5 are probability for actions, sixth one is coordinates for action 6
        # uses frame to return probability each action has to lead to frame change
        with torch.no_grad():
            combined_logits = self.action_model(current_frame.unsqueeze(0))
            combined_logits = combined_logits.squeeze(0)  # (5 + 4096,)
            
            # Split logits into action and coordinate spaces
            action_logits = combined_logits[:5]  # ACTION1-ACTION5
            coord_logits = combined_logits[5:]   # ACTION6 coordinates (4096)
            
            # Convert logits to probabilities using sigmoid
            action_probs = torch.sigmoid(action_logits).cpu().numpy()  # Shape: (5,)
            coord_probs = torch.sigmoid(coord_logits).cpu().numpy()    # Shape: (4096,)
        
        # Build list of candidate actions with scores
        candidates_with_scores = []
        
        # ACTION1-ACTION5 candidates
        action_names = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"]
        for action_idx, action_name in enumerate(action_names):
            if action_name in available_action_strings:
                prob = float(action_probs[action_idx])
                candidate = {
                    'action': ActionCandidate(
                        action=action_name,
                        x=None,
                        y=None,
                        rationale=f"{self.action_semantics[action_name]} (confidence: {prob:.3f})"
                    ),
                    'score': prob
                }
                candidates_with_scores.append(candidate)
        
        # ACTION6 candidates (coordinate-based actions)
        if "ACTION6" in available_action_strings:
            # Find top coordinate positions by probability
            top_coord_indices = np.argsort(coord_probs)[-10:][::-1]  # Top 10 coordinates
            
            # we are adding 10 possible action 6 per CandidateActions
            for coord_idx in top_coord_indices:
                prob = float(coord_probs[coord_idx])
                y = coord_idx // self.grid_size
                x = coord_idx % self.grid_size
                
                candidate = {
                    'action': ActionCandidate( # actual candidation obj
                        action="ACTION6",
                        x=x,
                        y=y,
                        rationale=f"coordinate action at ({x}, {y}) (confidence: {prob:.3f})"
                    ),
                    'score': prob # probability of frame change -> used to get top k probabilities
                }
                candidates_with_scores.append(candidate)
        
        # Sort by score (descending) and take top-K
        candidates_with_scores.sort(key=lambda x: x['score'], reverse=True)
        top_k = 3  # Return top 3 candidates
        
        # Extract ActionCandidate objects
        result_candidates = [c['action'] for c in candidates_with_scores[:top_k]]
        
        # If no candidates (all actions filtered), return at least the best available
        if not result_candidates and candidates_with_scores:
            result_candidates = [candidates_with_scores[0]['action']]
        
        return CandidateActions(candidates=result_candidates)

    def _sample_from_combined_output(self, combined_logits: torch.Tensor, available_actions: list[int] = None) -> tuple[int, int, int, np.ndarray]:
        """Sample from combined 5 + 64x64 action space with masking for invalid actions."""
        # Split logits
        action_logits = combined_logits[:5]  # First 5
        coord_logits = combined_logits[5:]   # Remaining 4096
        
        # Apply masking based on available_actions if provided
        if available_actions is not None and len(available_actions) > 0:
            # Create mask for action logits (ACTION1-ACTION5 = indices 0-4)
            action_mask = torch.full_like(action_logits, float('-inf'))
            action6_available = False
            
            for action in available_actions:
                # Extract action value if it's a GameAction enum
                action_id = action.value
                
                if 1 <= action_id <= 5:  # ACTION1-ACTION5
                    action_mask[action_id - 1] = 0.0  # Unmask valid actions
                elif action_id == 6:  # ACTION6
                    action6_available = True
            
            # Apply mask to action logits
            action_logits = action_logits + action_mask
            
            # If ACTION6 (coordinate action) is not available, mask all coordinate logits
            if not action6_available:
                coord_mask = torch.full_like(coord_logits, float('-inf'))
                coord_logits = coord_logits + coord_mask
        
        # Apply sigmoid
        action_probs = torch.sigmoid(action_logits)
        coord_probs_raw = torch.sigmoid(coord_logits)
        
        # For fair sampling: treat coordinates as one action type with total prob divided by 4096
        coord_probs_scaled = coord_probs_raw / self.num_coordinates
        
        # Combine for sampling (normalize)
        all_probs_sampling = torch.cat([action_probs, coord_probs_scaled])
        all_probs_sampling = all_probs_sampling / all_probs_sampling.sum()
        all_probs_sampling_np = all_probs_sampling.cpu().numpy()
        
        # Sample from normalized space
        selected_idx = np.random.choice(len(all_probs_sampling_np), p=all_probs_sampling_np)
        
        # Return unnormalized sigmoid values for visualization
        coord_probs_viz = torch.sigmoid(coord_logits)  # Raw sigmoid for visualization
        all_probs_viz = torch.cat([action_probs, coord_probs_viz])
        all_probs_viz_np = all_probs_viz.cpu().numpy()
        
        if selected_idx < 5:
            # Selected one of ACTION1-ACTION5
            return selected_idx, None, None, all_probs_viz_np
        else:
            # Selected a coordinate (index 5-4100)
            coord_idx = selected_idx - 5
            y_idx = coord_idx // self.grid_size
            x_idx = coord_idx % self.grid_size
            return 5, (y_idx, x_idx), coord_idx, all_probs_viz_np

    def _frame_to_tensor(self, frame_data: FrameData) -> torch.Tensor:
        """Convert frame data to tensor format for the model."""
        # Convert frame to numpy array with color indices 0-15
        frame = np.array(frame_data.frame, dtype=np.int64)
        
        # Take the last frame (in case of an animation of frames)
        frame = frame[-1]
        
        assert frame.shape == (self.grid_size, self.grid_size)
        
        # One-hot encode: (64, 64) -> (16, 64, 64)
        tensor = torch.zeros(self.num_colours, self.grid_size, self.grid_size, dtype=torch.float32)
        tensor.scatter_(0, torch.from_numpy(frame).unsqueeze(0), 1)
        
        return tensor.to(self.device)

    def _compute_experience_hash(self, frame: np.array, action_idx: int) -> str:
        """Compute hash for frame+action combination to ensure uniqueness."""
        assert frame.shape == (self.num_colours, self.grid_size, self.grid_size)
        frame_bytes = frame.tobytes()
        
        # Create hash from frame + action combination
        hash_input = frame_bytes + str(action_idx).encode('utf-8')
        return hashlib.md5(hash_input).hexdigest()

    def _train_action_model(self):
        """Train the action model on collected experiences with hierarchical click selection."""
        if len(self.experience_buffer) < self.batch_size:
            return
        
        # Sample batch from experience buffer
        batch_indices = np.random.choice(len(self.experience_buffer), self.batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in batch_indices]
        
        # Prepare batch data - convert numpy arrays to tensors and move to GPU
        states = torch.stack([torch.from_numpy(exp['state']).float().to(self.device) for exp in batch])
        action_indices = torch.tensor([exp['action_idx'] for exp in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([exp['reward'] for exp in batch], dtype=torch.float32, device=self.device)
        
        self.optimizer.zero_grad()
        
        # Forward pass - unified action space
        combined_logits = self.action_model(states)  # (batch, 4101)
        
        # Single unified loss - only if there's at least one positive reward
        selected_logits = combined_logits.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        main_loss = F.binary_cross_entropy_with_logits(selected_logits, rewards)
        
        # Light entropy regularization - encourage exploration via high sigmoid values
        all_probs = torch.sigmoid(combined_logits)
        
        # Split into action and coordinate spaces
        action_probs = all_probs[:, :5]
        coord_probs = all_probs[:, 5:]
        
        # Calculate entropy bonus based on raw sigmoid values (higher values = more confident exploration)
        action_entropy = action_probs.mean()  # Mean sigmoid activation (0-1 range)
        coord_entropy = coord_probs.mean()    # Mean sigmoid activation (0-1 range)
        
        # Get dynamic entropy coefficients from scheduler
        action_coeff=0.0001
        coord_coeff=0.00001
        
        # Apply dynamic entropy regularization
        total_loss = main_loss - action_coeff * action_entropy - coord_coeff * coord_entropy
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        
        # Clean up GPU memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def _has_time_elapsed(self) -> bool:
        """Check if 8 hours have elapsed since start."""
        elapsed_hours = time.time() - self.start_time
        return elapsed_hours >= 8 * 3600 - 5 * 60 # 8 hours with a 5 minute safety buffer.

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing or not."""
        return any([
            latest_frame.state is GameState.WIN,
            self._has_time_elapsed(),
        ])

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:

        """Choose action using action model predictions."""
        # Check if score has changed and log score at action count
        if latest_frame.score != self.current_score:
            self.logger.info(f"Score changed from {self.current_score} to {latest_frame.score} at action {self.action_counter}")
            print(f"Score changed from {self.current_score} to {latest_frame.score} at action {self.action_counter}")
            
            # Clear experience buffer when reaching new level
            self.experience_buffer.clear()
            self.experience_hashes.clear()
            self.logger.info(f"Cleared experience buffer - new level reached")
            print("Cleared experience buffer - new level reached")
            
            self.logger.info(f"Reset entropy scheduler for new level - starting with high exploration")
            print("Reset entropy scheduler for new level - starting with high exploration")
            
            # Reset network and optimizer for new level
            # TODO: Try not resetting the networks here. Perhaps it performs even better.
            self.action_model = ActionModel(input_channels=self.num_colours, grid_size=self.grid_size).to(self.device)
            self.optimizer = optim.Adam(self.action_model.parameters(), lr=0.0001)
            self.logger.info(f"Reset action model and optimizer for new level")
            print("Reset action model and optimizer for new level")
            
            # Reset previous tracking
            self.prev_frame = None
            self.prev_action_idx = None
            
            
            self.current_score = latest_frame.score
        
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            # Reset previous tracking on game reset
            self.prev_frame = None
            self.prev_action_idx = None
            action = GameAction.RESET
            action.reasoning = "Game needs reset."
            return action


        # Convert current frame to tensor
        current_frame = self._frame_to_tensor(latest_frame)
        
        # If frame processing failed, reset tracking and return random action
        if current_frame is None:
            print("Error detected!")
            self.prev_frame = None
            self.prev_action_idx = None
            
            action = random.choice(self.action_list[:5])  # Random ACTION1-ACTION5
            action.reasoning = f"Skipped weird frame, random {action.value}"
            return action
        
        # Create experience from previous action if we have previous data
        if self.prev_frame is not None:
            # Compute hash for uniqueness check
            experience_hash = self._compute_experience_hash(self.prev_frame, self.prev_action_idx)
            
            # Only store if unique
            if experience_hash not in self.experience_hashes:
                # Convert current frame to numpy bool for comparison
                current_frame_np = current_frame.cpu().numpy().astype(bool)
                frame_changed = not np.array_equal(self.prev_frame, current_frame_np)
                # if frame_changed:
                    # print(f"Action: {self.prev_action_idx} got a new positive reward!")
                experience = {
                    'state': self.prev_frame,  # Already numpy bool
                    'action_idx': self.prev_action_idx,  # Unified action index
                    'reward': 1.0 if frame_changed else 0.0
                }
                self.experience_buffer.append(experience)
                self.experience_hashes.add(experience_hash)
                
        # Get action predictions from action model
        with torch.no_grad():
            combined_logits = self.action_model(current_frame.unsqueeze(0))
            combined_logits = combined_logits.squeeze(0)  # (5 + 4096,)
            
            # Sample from combined action space
            action_idx, coords, coord_idx, all_probs = self._sample_from_combined_output(combined_logits, latest_frame.available_actions)
            
            if action_idx < 5:
                # Selected ACTION1-ACTION5
                selected_action = self.action_list[action_idx]
                selected_action.reasoning = f"{selected_action.name} (prob: {all_probs[action_idx]:.3f})"
            else:
                # Selected a coordinate - treat as ACTION6
                selected_action = GameAction.ACTION6
                y, x = coords
                selected_action.set_data({"x": x, "y": y})
                selected_action.reasoning = f"ACTION6 at ({x}, {y}) (prob: {all_probs[coord_idx]:.3f})"
                
        
        # Store current frame and action for next experience creation
        self.prev_frame = current_frame.cpu().numpy().astype(bool)
        # Store unified action index: 0-4 for ACTION1-5, 5+ for coordinates
        if action_idx < 5:
            self.prev_action_idx = action_idx
        else:
            self.prev_action_idx = 5 + coord_idx  # Unified action space
        
        
        # Train model periodically
        if self.action_counter % self.train_frequency == 0:
            self._train_action_model()
        
        
        return selected_action