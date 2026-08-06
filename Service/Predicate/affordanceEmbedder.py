import torch
import torch.nn as nn
from typing import List, Dict, Any

import torch
import torch.nn as nn
from IO_Module.logger import Logger

from Service.Predicate.triplet2natural import PREDICATE_TEMPLATES

class GeometricPredicateExtractor(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.d_dim:int = embedding_dim 
        self.__predicates:list[str] = ['is_in', 'is_on', 'is_near']

        # Elements correspond to ideal CXCYWH offsets: [delta_cx, delta_cy, delta_log_w, delta_log_h]
        self.__ideal_coordinates = {
            'is_in':   torch.tensor([ 0.0,  0.0, -0.6, -0.6]),
            'is_on':   torch.tensor([ 0.0, -0.85, -0.2, -0.2]), # Increased negative cy offset
            'is_near': torch.tensor([ 1.2,  0.0,  0.0,  0.0])
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

    def get_predicate(self, box_f, box_o, gt_predicate: str | None = None,Tempreture:float=0.02) -> tuple[str, float, dict]:
        """
        Args:
            box_f: Tensor or List [x1, y1, x2, y2]
            box_o: Tensor or List [x1, y1, x2, y2]
            gt_predicate (str, optional): Ground-truth predicate to evaluate absolute target margin.

        Returns:
            winning_predicate (str): Name of predicted geometric predicate template.
            margin (float): Computed Delta_margin confidence score.
            sim_dict (dict): Full dictionary of cosine similarities across all anchors.
        """
        device = self.anchors.device

        if not isinstance(box_f, torch.Tensor):
            box_f = torch.tensor(box_f, dtype=torch.float32, device=device)
        if not isinstance(box_o, torch.Tensor):
            box_o = torch.tensor(box_o, dtype=torch.float32, device=device)

        # Convert XYXY to CXCYWH
        wf, hf = max(box_f[2] - box_f[0], 1e-6), max(box_f[3] - box_f[1], 1e-6)
        cxf, cyf = box_f[0] + wf / 2.0, box_f[1] + hf / 2.0

        wo, ho = max(box_o[2] - box_o[0], 1e-6), max(box_o[3] - box_o[1], 1e-6)
        cxo, cyo = box_o[0] + wo / 2.0, box_o[1] + ho / 2.0

        # Layout offsets
        t_live = torch.tensor([
            (cxf - cxo) / wo,
            (cyf - cyo) / ho,
            torch.log(wf / wo),
            torch.log(hf / ho)
        ], dtype=torch.float32, device=device)

        # Continuous wave projection and unit normalization
        v_live = self._compute_wave_embedding(t_live)
        v_live_norm = v_live / torch.norm(v_live, p=2)

        # Parallel dot-product similarity against template anchors
        # Parallel dot-product similarity against template anchors
        similarities = torch.mv(self.anchors_norm, v_live_norm)

        # Apply Temperature Scaling (T = 0.02) to un-squash cosine similarities
        temperature = Tempreture
        scaled_logits = similarities / temperature
        probs = torch.softmax(scaled_logits, dim=-1)

        # 1. Evaluate Margin against Ground-Truth reference if provided
        if gt_predicate is not None and gt_predicate in self.predicates:
            gt_idx = self.predicates.index(gt_predicate)
            s_target = probs[gt_idx]
            
            mask = torch.ones(len(self.predicates), dtype=torch.bool, device=device)
            mask[gt_idx] = False
            s_comp_max = probs[mask].max()
            
            margin = (s_target - s_comp_max).item()
            winning_idx = int(torch.argmax(probs).item())
        
        # 2. Otherwise compute top-1 vs top-2 competitive margin
        else:
            top_vals, top_idxs = torch.topk(probs, k=min(2, len(self.predicates)))
            margin = (top_vals[0] - top_vals[1]).item()
            winning_idx = int(top_idxs[0].item())

        sim_dict = {p: probs[i].item() for i, p in enumerate(self.predicates)}
        return self.predicates[winning_idx], margin, sim_dict
    
    
    
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