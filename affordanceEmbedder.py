import torch
import torch.nn as nn
from typing import List, Dict, Any

import torch
import torch.nn as nn
from IO_Module.logger import Logger

from triplet2natural import PREDICATE_TEMPLATES

class GeometricPredicateExtractor(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.d_dim = embedding_dim 
        self.__predicates = ['is_in', 'is_on', 'is_near']

        # Elements correspond to ideal CXCYWH offsets: [delta_cx, delta_cy, delta_log_w, delta_log_h]
        self.__ideal_coordinates = {
            'is_in':   torch.tensor([ 0.0,  0.0, -0.7, -0.7]), 
            'is_on':   torch.tensor([ 0.0, -0.6, -0.4, -0.4]), 
            'is_near': torch.tensor([ 1.0,  0.0,  0.0,  0.0])  
        }
        
        # Initialize and register buffers correctly
        self._recompute_anchors()

    def _recompute_anchors(self):
        """Helper to compute, register, and normalize anchors upon structural changes."""
        raw_anchors = torch.stack([
            self._compute_wave_embedding(self.__ideal_coordinates[p]) for p in self.__predicates
        ])
        
        # Register raw anchors as buffer (moves with module device automatically)
        self.register_buffer('anchors', raw_anchors)
        
        # Pre-compute and cache normalized anchors to save GPU cycles in live loops
        norm_anchors = raw_anchors / torch.norm(raw_anchors, p=2, dim=1, keepdim=True)
        self.register_buffer('anchors_norm', norm_anchors)

    @property
    def predicates(self):
        return self.__predicates
    
    @predicates.setter
    def predicates(self, new_predicates: list | str):
        if not isinstance(new_predicates, list):
            new_predicates = [new_predicates]
        self.__predicates += new_predicates
        self._recompute_anchors()
    
    @property
    def dimension(self):
        return self.d_dim
        
    @dimension.setter
    def dimension(self, new_dim):
        self.d_dim = new_dim
        self._recompute_anchors()
    
    def _compute_wave_embedding(self, t_vector):
        device = t_vector.device
        feat_range = torch.arange(self.d_dim, dtype=torch.float32, device=device)
        dim_mat = 10000.0 ** (2.0 * (feat_range // 2) / self.d_dim)
        
        embeddings = []
        for val in t_vector:
            sinusoid_input = val / dim_mat
            emb = torch.zeros(self.d_dim, device=device)
            emb[0::2] = torch.sin(sinusoid_input[0::2])
            emb[1::2] = torch.cos(sinusoid_input[1::2])
            embeddings.append(emb)
            
        return torch.cat(embeddings) 

    def get_predicate(self, box_f, box_o):
        """
        Args:
            box_f: List or Tensor [x1, y1, x2, y2]
            box_o: List or Tensor [x1, y1, x2, y2]
        Returns:
            winning_predicate: String token matching closest layout template
        """
        # 1. Extract device once from your trusted cached buffer.
        # This acts as the single source of truth and silences Pylance.
        device = self.anchors.device

        # Ensure conversion to tensor on correct device if input is a python list
        if not isinstance(box_f, torch.Tensor):
            box_f = torch.tensor(box_f, dtype=torch.float32, device=device)
        if not isinstance(box_o, torch.Tensor):
            box_o = torch.tensor(box_o, dtype=torch.float32, device=device)

        # Convert XYXY format to CXCYWH format
        wf, hf = box_f[2] - box_f[0], box_f[3] - box_f[1]
        cxf, cyf = box_f[0] + wf / 2.0, box_f[1] + hf / 2.0

        wo, ho = box_o[2] - box_o[0], box_o[3] - box_o[1]
        cxo, cyo = box_o[0] + wo / 2.0, box_o[1] + ho / 2.0

        # Prevent division-by-zero or negative log issues
        wo = max(wo, 1e-6)
        ho = max(ho, 1e-6)
        wf = max(wf, 1e-6)
        hf = max(hf, 1e-6)
        
        # 2. Use the local 'device' variable here. Pylance knows this is perfectly safe.
        t_live = torch.tensor([
            (cxf - cxo) / wo,
            (cyf - cyo) / ho,
            torch.log(torch.tensor(wf / wo, device=device)),
            torch.log(torch.tensor(hf / ho, device=device))
        ], dtype=torch.float32, device=device)
        
        # Step 2: Project runtime layout to 256D wave continuous signature space
        v_live = self._compute_wave_embedding(t_live)
        
        # Step 3: Run runtime unit-normalization
        v_live_norm = v_live / torch.norm(v_live, p=2)
        
        # Step 4: Parallel Matrix-Vector dot product utilizing cached template matrices
        similarities = torch.mv(self.anchors_norm, v_live_norm) 
        winning_index = int(torch.argmax(similarities).item())
        
        return self.predicates[winning_index]
    
    
    
def generate_semantic_prompt(subject: str, predicate: str, obj: str) -> str:
    """
    Transforms raw spatial triplets into fluent natural language sentences.
    Example: ("flame", "is_in", "Furniture_Fabric") -> "A flame is located inside and surrounded by the furniture fabric."
    """
    # 1. Clean up snake_case class names from your YOLO/Object detection labels
    clean_subject = subject.replace("_", " ").lower()
    if obj.split("_")[0]=='Appliance':
        clean_object=obj.split("_")[1].lower()
    else:
        clean_object=obj.lower().replace("_", " made of ")
    
    # 2. Extract the descriptive natural language predicate phrase
    # Fallback to a basic string replacement if the predicate isn't explicitly mapped
    natural_predicate = PREDICATE_TEMPLATES.get(predicate, predicate.replace("_", " "))
    
    # 3. Assemble into a grammatically structured sentence
    # We use indefinite/definite articles to help the embedder process it like normal text
    natural_sentence = f"A {clean_subject} {natural_predicate} the {clean_object}."
    
    return natural_sentence