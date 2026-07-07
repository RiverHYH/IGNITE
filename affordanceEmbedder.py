import torch
import torch.nn as nn
from typing import List, Dict, Any

class GeometricPredicateExtractor(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.d_dim = embedding_dim # 64 dims per coordinate coordinate projection = 256 total dims
        self.__predicates = ['is_in', 'is_on', 'is_near']

        # Elements correspond to: [delta_x, delta_y, delta_w, delta_h]
        self.__ideal_coordinates = {
            'is_in':   torch.tensor([ 0.0,  0.0, -0.7, -0.7]), # Centered inside
            'is_on':   torch.tensor([ 0.0, -0.6, -0.4, -0.4]), # Directly stacked on top
            'is_near': torch.tensor([ 1.0,  0.0,  0.0,  0.0])  # Displaced horizontally next to
        }
        
        # 2. Pre-calculate and cache the static anchor embeddings on instantiation
        self.anchors: torch.Tensor  # type hint for Pylance
        self.register_buffer('anchors', torch.stack([
            self._compute_wave_embedding(self.__ideal_coordinates[p]) for p in self.predicates
        ])) # Shape: [3, 256]
    
    
    @property
    def predicates(self):
        """
        Predicates available in the model
        """
        return self.__predicates
    
    @predicates.setter
    def predicates(self, new_predicates:list|str):
        """
        Setter of Predicates. Takes in new predicates
        Args:
            new_predicates (list|str): list of new predicates to add to the model
        Returns:
            None
        """
        if type(new_predicates) != list:
            new_predicates = [new_predicates]
        self.__predicates+=new_predicates
        self.anchors = torch.stack([
            self._compute_wave_embedding(self.__ideal_coordinates[p]) for p in self.predicates
        ])
    
    @property
    def dimension(self):
        return self.d_dim
    @dimension.setter
    def dimension(self,new_dim):
        self.d_dim = new_dim
        # Recompute anchors with new dimension
        self.anchors = torch.stack([
            self._compute_wave_embedding(self.__ideal_coordinates[p]) for p in self.predicates
        ])
    
    def _compute_wave_embedding(self, t_vector):
        """
        Maps a 4D spatial layout vector into a 256D continuous wave signature.
        Args:
            t_vector: Tensor [x_center, y_center, width, height]
        
        Returns:
            embedding: Tensor [256] - Continuous wave signature
        """
        device = t_vector.device
        # Compute multi-frequency scaling denominators
        feat_range = torch.arange(self.d_dim, dtype=torch.float32, device=device)
        dim_mat = 10000.0 ** (2.0 * (feat_range // 2) / self.d_dim)
        
        embeddings = []
        for val in t_vector:
            # Broadcast scalar across all frequencies
            sinusoid_input = val / dim_mat
            # Interleave sine and cosine waves
            emb = torch.zeros(self.d_dim, device=device)
            emb[0::2] = torch.sin(sinusoid_input[0::2])
            emb[1::2] = torch.cos(sinusoid_input[1::2])
            embeddings.append(emb)
            
        return torch.cat(embeddings) # Concatenate 4 channels of 64 dims -> 256D Vector

    def get_predicate(self, box_f, box_o):
        """
        Inputs:
            box_f: Tensor [x_center, y_center, width, height]
            box_o:  Tensor [x_center, y_center, width, height]
        Returns:
            winning_predicate: String token matching the closest physical topology layout
        """
        xs, yf, wf, hf = box_f
        xo, yo, wo, ho = box_o
        
        # Step 1: Extract normalized scale-invariant relative vector
        t_live = torch.tensor([
            (xs - xo) / wo,
            (yf - yo) / ho,
            torch.log(wf / wo),
            torch.log(hf / ho)
        ], device=box_f.device)
        
        # Step 2: Project to 256D wave continuous space
        v_live = self._compute_wave_embedding(t_live) # Shape: [256]
        
        # Step 3 & 4: Compute continuous cosine attention matching against the templates
        # Normalize vectors to calculate dot product cosine similarities
        v_live_norm = v_live / torch.norm(v_live, p=2)
        anchors_norm = self.anchors / torch.norm(self.anchors, p=2, dim=1, keepdim=True)
        
        similarities = torch.mv(anchors_norm, v_live_norm) # Matrix-Vector dot product (Shape: [3])
        winning_index = int(torch.argmax(similarities).item())
        
        return self.predicates[winning_index]
    
    
    
def concat(sub,predicate,obj):
    """
    structral text aggregator
    Args:
    sub(str): subject label
    predicate(str): predicate
    obj(str): object label
    
    Returns:
        str: Concatenated sentence in the form "subject predicate object"
    """
    
    return f"{sub} {predicate} {obj}"