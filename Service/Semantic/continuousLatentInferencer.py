import importlib.util
import pathlib
import pickle
from typing import Union, List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
import sys,os
from utils.logger import Logger

import pprint


class StandardEncoder:
  def __init__(self,model_name="BAAI/bge-small-en-v1.5"):
    self.model=AutoModel.from_pretrained(model_name)
    self.tokenizer=AutoTokenizer.from_pretrained(model_name)
  def encode(self,text):
    inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = self.model(**inputs)
    # Use the CLS token embedding
    return outputs.last_hidden_state[:, 0, :]

class ContinuousLatentInferencer(StandardEncoder):
    def __init__(
        self,
        ref_triplets="Service//Semantic//common_knowledge.py",
        cache_file="Service//Semantic//triplets_cache.pkl",
    ):
        super().__init__()

        self.cache_path = pathlib.Path(cache_file)
        self.file_path = pathlib.Path(ref_triplets)
        self.__ref= []
        self.ref_matrix= None

        # Initialise data loading and matrix generation
        self._initialise_data()

    def reset(self, dump_to_py: bool = True) -> None:
        """
        Private Method.
        Ignores the existing pickle cache file, reloads the original reference triplets from the
        source file, recalculates embeddings for all entries, updates the cache, and reinitialises
        the GPU matrix.
        Args:
            dump_to_py (bool): Whether to dump the computed embeddings to a pickle cache file. Defaults to True.
        """
        Logger.info(f"Resetting cache. Bypassing existing pickle file and reloading source: {self.file_path}")
        
        # 1. Read directly from the source Python file
        raw_data = self._resolve_and_validate_triplets(self.file_path)
        
        # 2. Force recalculation of all embeddings and serialise to cache
        self.__ref = self._compute_and_cache_embeddings(
            raw_data, dump=dump_to_py, force_recompute=True
        )
        
        # 3. Synchronise the updated GPU matrix
        self._update_ref_matrix()
        Logger.info("Reset complete and cache updated successfully.")

    def _initialise_data(self) -> None:
        """Private Method. Initialises data from binary cache if available, otherwise parses source triplets."""
        if self.cache_path.exists():
            Logger.info(f"Loading pre-computed embeddings from cache: {self.cache_path}")
            with open(self.cache_path, "rb") as f:
                self.__ref = pickle.load(f)
        else:
            Logger.critical(f"No cache found. Processing source: {self.file_path}")
            raw_data = self._resolve_and_validate_triplets(self.file_path)
            self.__ref = self._compute_and_cache_embeddings(raw_data)

        self._update_ref_matrix()

    def _update_ref_matrix(self) -> None:
        """Extracts embeddings and stacks them into a FP16 2D tensor on GPU."""
        all_embeddings = [item["embedding"].squeeze() for item in self.__ref]
        self.ref_matrix = torch.stack(all_embeddings).cuda().half()

    def _resolve_and_validate_triplets(
        self, source: Union[str, pathlib.Path, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Loads 'ref_triplets' dynamically from a file path or validates a direct list.
        """
        if isinstance(source, (str, pathlib.Path)):
            file_path = pathlib.Path(source)
            if not file_path.exists():
                raise FileNotFoundError(f"Source file '{file_path}' not found.")

            spec = importlib.util.spec_from_file_location("external_triplets", file_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not read layout of {file_path}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if not hasattr(module, "ref_triplets"):
                raise AttributeError(f"Could not find 'ref_triplets' inside {file_path.name}")

            data = getattr(module, "ref_triplets")
        elif isinstance(source, list):
            data = source
        else:
            raise TypeError("ref_triplets must be a file path string, Path object, or a list of dicts.")

        if not self._validate_structure(data):
            raise ValueError("Invalid ref_triplets schema. Required keys: status, triplet, embedding")

        return data

    def _validate_structure(self, data: Any) -> bool:
        """Enforces schema: {'status': int, 'triplet': str, 'embedding': Any}"""
        if not isinstance(data, list):
            return False
        required_keys = {"status", "triplet", "embedding"}
        for item in data:
            if not isinstance(item, dict) or not required_keys.issubset(item.keys()):
                return False
        return True

    def _compute_and_cache_embeddings(
        self,
        data: List[Dict[str, Any]],
        dump: bool = True,
        force_recompute: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Computes missing vectors (or all vectors if force_recompute=True) and
        serialises the list to cache.
        """
        updated = False

        for item in data:
            if force_recompute or item.get("embedding") is None:
                print(f"Generating embedding for: '{item['triplet']}'...")
                # Call parent class encoding feature
                item["embedding"] = self.encode(item["triplet"])
                updated = True

        # Save to disk so future executions avoid recalculation
        if updated or force_recompute:
            with open(self.cache_path, "wb") as f:
                pickle.dump(data, f)
            print(f"Successfully saved computed embeddings to cache: {self.cache_path}")

        if dump:
            self._write_back_to_py_file(data)

        return data

    def _write_back_to_py_file(self, data: List[Dict[str, Any]]):
        """Formats the live list back into clean Python code and overwrites the source file."""
        formatted_list = pprint.pformat(data, indent=4, sort_dicts=False)
        code_content = (
            "# Generated automatically by ContinuousLatentInferencer\n\n"
            "from torch import tensor\n\n"
            f"ref_triplets = {formatted_list}\n"
        )

        with open(self.file_path, "w", encoding="utf-8") as f:
            f.write(code_content)
        Logger.info(f"Successfully updated source text file: {self.file_path.name}")

    @property
    def references(self):
        """Public getter to inspect the loaded data safely."""
        return self.__ref
    def embed(self,triplet):
        return self.encode(triplet)
    
    def _cosine_similarity(self, runtime_triplet):
        """
        Parallel computation of cosine similarity scores against the entire matrix.
        """
        # Encode live text generated from Stage 3
        qe = self.encode(runtime_triplet).cuda().half()
        
        # Normalize the live vector
        query_embedding = qe / qe.norm(dim=-1, keepdim=True)
        
        # Parallel Matrix Multiplication: (M, 384) x (384, 1) -> (M,)
        scores = torch.matmul(self.ref_matrix, query_embedding.T).squeeze(-1)
        return scores

    def get_top_k(self, runtime_triplet, k=10):
        # Guard rail against k requests larger than your reference inventory
        k = min(k, len(self.__ref))
        
        # Execute vectorized similarity lookup
        scores = self._cosine_similarity(runtime_triplet)
        
        # Compute top k values and their structural position indexes on the GPU
        top_k = torch.topk(scores, k=k)
        
        indices = top_k.indices.tolist()
        confidences = top_k.values.tolist()
        
        # 3. Construct rich, auditable outputs using matching indices
        matched_triplets = []
        matched_statuses = []
        
        for idx in indices:
            matched_triplets.append(self.__ref[idx]['triplet'])
            matched_statuses.append(self.__ref[idx]['status'])
            
        # Returns corresponding texts, numerical alert rules, and matching confidence scores
        return matched_triplets, matched_statuses, confidences
    
    
    
    
    
if __name__ == "__main__":
    inferencer = ContinuousLatentInferencer()
    inferencer.reset()
    triplets, statuses, confidences = inferencer.get_top_k("Flame inside gaslamp", k=1)
    
    print("Matched Reference Rules:", triplets)
    print("Associated Action Statuses:", statuses)
    print("Latent Confidence Metric:", confidences)