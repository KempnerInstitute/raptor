import glob
import os
import sys

def find_raptor_checkpoint(variant: str, seed: int, base_dir: str) -> str:
    """
    Find the raptor checkpoint file for a given variant and seed.
    
    Args:
        variant: 'raptor2', 'raptor3', or 'raptor4'
        seed: The seed number (e.g., 1001)
        base_dir: The directory where runs/models is located.
                  Typically '/path/to/project/result_dir'
    
    Returns:
        Absolute path to the checkpoint file.
    
    Raises:
        FileNotFoundError: If no matching file is found.
        ValueError: If multiple matching files are found.
    """
    models_dir = os.path.join(base_dir, "runs", "models")
    
    # Pattern to match the file naming convention observed:
    # final_{variant}_..._seed_{seed}_step_*.pt
    # The variant name appears near the front.
    # We use a wildcard for the middle parameters.
    
    pattern = f"final_{variant}_*_seed_{seed}_step_*.pt"
    search_path = os.path.join(models_dir, pattern)
    
    matches = glob.glob(search_path)
    
    if len(matches) == 0:
        raise FileNotFoundError(f"No checkpoint found for variant '{variant}' and seed '{seed}' in {models_dir} with pattern {pattern}")
    
    if len(matches) > 1:
        # If we have multiple matches, we might need more specific logic. 
        # For now, let's print them and raise an error to be safe.
        raise ValueError(f"Multiple checkpoints found for variant '{variant}' and seed '{seed}': {matches}")
        
    return matches[0]
