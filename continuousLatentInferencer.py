import importlib.util
import pathlib
import pickle
from typing import Union, List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
from IO_Module.logger import Logger

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
    def __init__(self, ref_triplets="common_knowledge.py", cache_file="triplets_cache.pkl"):
        super().__init__()
        
        self.cache_path = pathlib.Path(cache_file)
        self.file_path=pathlib.Path(ref_triplets)
        
        
        # 2. Try to load from binary cache first for maximum speed
        if self.cache_path.exists():
            print(f"Loading pre-computed embeddings from cache: {self.cache_path}")
            with open(self.cache_path, "rb") as f:
                self.__ref = pickle.load(f)
        else:
            # 3. If no cache exists, parse the source file and generate them
            print(f"No cache found. Processing source: {ref_triplets}")
            raw_data = self._resolve_and_validate_triplets(ref_triplets)
            self.__ref = self._compute_and_cache_embeddings(raw_data)

    def _resolve_and_validate_triplets(self, source: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Loads 'ref_triplets' dynamically from a file or validates a direct list."""
        if isinstance(source, str):
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
            raise TypeError("ref_triplets must be a file path string or a list of dicts.")

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

    def _compute_and_cache_embeddings(self, data: List[Dict[str, Any]],dump=True) -> List[Dict[str, Any]]:
        """Computes missing vectors and serializes the list to a file."""
        updated = False
        
        for item in data:
            if item['embedding'] is None:
                print(f"Generating embedding for: '{item['triplet']}'...")
                # Call the Father Class' encoding feature
                item['embedding'] = self.encode(item['triplet'])
                item['status'] = 1
                updated = True
                
        # Save to disk so this heavy work never runs again for these triplets
        if updated:
            with open(self.cache_path, "wb") as f:
                pickle.dump(data, f)
            print(f"Successfully saved computed embeddings to cache: {self.cache_path}")
        if dump:
            self._write_back_to_py_file(data)
            
        return data

    def _write_back_to_py_file(self, data: List[Dict[str, Any]]):
        """Formats the live list back into clean Python code and overwrites the file."""
        
            
        # Use pretty-print to format the list of dicts cleanly
        formatted_list = pprint.pformat(data, indent=4, sort_dicts=False)
            
        # Reconstruct the file content
        code_content = f"# Generated automatically by ContinuousLatentInferencer\n\nref_triplets = {formatted_list}\n"
            
        with open(self.file_path, "w", encoding="utf-8") as f:
            f.write(code_content)
        Logger.info(f"Successfully updated source text file: {self.file_path.name}")

    @property
    def references(self):
        """Public getter to inspect the loaded data safely."""
        return self.__ref
    def embed(self,triplet):
        return self.encode(triplet)
    
    
    
    
    
    
if __name__ == "__main__":
    inferencer = ContinuousLatentInferencer()