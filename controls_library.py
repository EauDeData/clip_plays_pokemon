import annoy
import json
import random
import torch 
import torch.nn as nn
import torch.optim as optim
import open_clip
import numpy as np
from PIL import Image
from collections import deque


def correct_prior_shift(p_biased, p_true_prior, p_train_prior, eps=1e-12):
    """
    Correct posterior probabilities under label shift.

    Parameters
    ----------
    p_biased : array-like, shape (..., K)
        Biased posterior probabilities P_train(x | image).
    p_true_prior : array-like, shape (K,)
        True class prior probabilities P_true(x).
    p_train_prior : array-like, shape (K,)
        Training class prior probabilities P_train(x).
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    p_unbiased : ndarray, shape (..., K)
        Corrected posterior probabilities.
    """

    p_biased = np.asarray(p_biased, dtype=float)
    p_true_prior = np.asarray(p_true_prior, dtype=float)
    p_train_prior = np.asarray(p_train_prior, dtype=float)

    # Avoid divide-by-zero
    weights = p_true_prior / (p_train_prior + eps)

    # Reweight
    weighted = p_biased * weights

    # Renormalize
    normalization = np.sum(weighted, axis=-1, keepdims=True) + eps
    p_unbiased = weighted / normalization

    return p_unbiased


class ControlCollection:
    def __init__(self, device='cuda', use_rl_projection=True, buffer_size=100):
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.device = device
        self.model.to(device)
        
        # Get embedding dimension (512 for ViT-B-32)
        self.embed_dim = 512
        
        # Trainable linear projection for image features
        self.use_rl_projection = use_rl_projection
        if use_rl_projection:
            self.image_projection = nn.Linear(self.embed_dim, self.embed_dim, bias=False).to(device)
            # Initialize as identity
            nn.init.eye_(self.image_projection.weight)
            
            # RL optimizer
            self.optimizer = optim.Adam(self.image_projection.parameters(), lr=1e-4)
            
            # Screenshot buffer for diversity computation
            self.screenshot_buffer = deque(maxlen=buffer_size)
            self.feature_buffer = deque(maxlen=buffer_size)
        
        with open('control_mappings.json') as fh:
            controls_mapping = json.load(fh)

        self.sentence2control = {}
        self.idx2sentence = []
        
        for control in controls_mapping:
            for sentence in controls_mapping[control]:
                self.sentence2control[sentence] = control
                self.idx2sentence.append(sentence)
        
        self.indices = list(range(len(self.idx2sentence)))
        self.prior_counts = np.ones(len(self.idx2sentence))
        self.compute_embeddings()
        
        # RL tracking
        self.episode_rewards = []
        self.step_count = 0
        
    def sample_controls(self, sample_size=10):
        elements = random.sample(self.indices, k=sample_size)
        sentences = [self.idx2sentence[i] for i in elements]
        buttons = [self.sentence2control[i] for i in sentences]
        embeddings = self.text_embeddings[elements]
        return sentences, embeddings, buttons
    
    def compute_embeddings(self):
        ### Here we create an index
        with torch.no_grad():
            text = self.tokenizer(self.idx2sentence).to(self.device)
            text_features = self.model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            self.text_embeddings = text_features

    def compute_diversity_reward(self, current_features):
        """
        Compute reward based on diversity of screenshots.
        Higher reward for more diverse/novel content.
        
        Returns
        -------
        float
            Diversity reward
        """
        if len(self.feature_buffer) < 2:
            return 0.0
        
        # Convert buffer to tensor
        buffer_features = torch.stack(list(self.feature_buffer))  # [N, embed_dim]
        
        # Compute similarity between current and all buffered features
        similarities = torch.matmul(current_features, buffer_features.T)  # [1, N]
        
        # Reward is inversely proportional to max similarity
        # (lower similarity = more novel = higher reward)
        max_similarity = similarities.max().item()
        
        # Diversity reward: 1.0 for completely novel, 0.0 for identical
        diversity_reward = 1.0 - max_similarity
        
        # Also add entropy-based reward for overall buffer diversity
        if len(self.feature_buffer) >= 10:
            # Compute pairwise distances in buffer
            pairwise_sim = torch.matmul(buffer_features, buffer_features.T)
            # Average dissimilarity (excluding diagonal)
            mask = ~torch.eye(len(buffer_features), dtype=bool, device=self.device)
            avg_dissimilarity = (1.0 - pairwise_sim[mask]).mean().item()
            diversity_reward += 0.5 * avg_dissimilarity
        
        return diversity_reward
    def update_rl(self, reward):
        """
        Update the projection layer using policy gradient.
        
        Parameters
        ----------
        reward : float
            Diversity reward from current state
        """
        if not self.use_rl_projection:
            return
        
        # Store reward
        self.episode_rewards.append(reward)
        
        # Update every N steps
        if len(self.episode_rewards) >= 10:
            # Compute average reward (no gradient needed here)
            avg_reward = np.mean(self.episode_rewards)
            
            # Reset episode
            self.episode_rewards = []
            
            return avg_reward
        
        return None

    def get_next_action(self, image, temperature=1.0, train_rl=True):
        """
        Get the next action based on image input with optional RL training.
        
        Parameters
        ----------
        image : PIL.Image
            Input image
        temperature : float
            Temperature for softmax (default: 1.0)
        train_rl : bool
            Whether to train the RL projection (default: True)
        
        Returns
        -------
        tuple
            (selected_sentence, selected_button, probabilities, diversity_reward)
        """
        # Sample 10 random controls
        sentences, embeddings, buttons = self.sample_controls(sample_size=10)
        
        # Preprocess image and compute image features
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        diversity_reward = 0.0
        
        if train_rl and self.use_rl_projection:
            # Enable gradients for projection layer
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Apply trainable projection
            projected_features = self.image_projection(image_features)
            projected_features = projected_features / projected_features.norm(dim=-1, keepdim=True)
            
            # Compute diversity reward (detached for reward calculation)
            with torch.no_grad():
                diversity_reward = self.compute_diversity_reward(projected_features.detach())
            
            # Store in buffer (detached)
            self.feature_buffer.append(projected_features.detach().clone().squeeze())
            
            # Calculate similarity through the gradient graph
            similarity = (projected_features @ embeddings.T) / temperature
            probs = similarity.softmax(dim=-1)
            
            # RL loss: negative diversity reward weighted by action probability
            # This creates a policy gradient signal
            selected_idx = torch.argmax(probs).item()
            action_prob = probs[0, selected_idx]
            
            # Policy gradient loss
            loss = -action_prob * diversity_reward
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Convert to numpy for return
            probs_biased = probs.detach().cpu().numpy().squeeze()
            
            # Track rewards for monitoring
            avg_reward = self.update_rl(diversity_reward)
            if avg_reward is not None:
                print(f"RL Update - Avg Diversity Reward: {avg_reward:.4f}")
            
        else:
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                if self.use_rl_projection:
                    image_features = self.image_projection(image_features)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity and apply softmax
                similarity = (image_features @ embeddings.T) / temperature
                probs_biased = similarity.softmax(dim=-1).cpu().numpy().squeeze()
                
                selected_idx = np.argmax(probs_biased)
        
        # Get selected action
        selected_sentence = sentences[selected_idx]
        selected_button = buttons[selected_idx]
        
        # Update prior counter for the selected sentence
        global_idx = self.idx2sentence.index(selected_sentence)
        self.prior_counts[global_idx] += 1
        
        self.step_count += 1
        
        return selected_sentence, selected_button, probs_biased, diversity_reward
    def save_projection(self, path='projection_weights.pt'):
        """Save the trained projection layer."""
        if self.use_rl_projection:
            torch.save({
                'projection_state_dict': self.image_projection.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'step_count': self.step_count,
            }, path)
            print(f"Projection saved to {path}")
    
    def load_projection(self, path='projection_weights.pt'):
        """Load a trained projection layer."""
        if self.use_rl_projection:
            checkpoint = torch.load(path, map_location=self.device)
            self.image_projection.load_state_dict(checkpoint['projection_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.step_count = checkpoint.get('step_count', 0)
            print(f"Projection loaded from {path}")