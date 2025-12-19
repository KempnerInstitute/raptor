import argparse
import sys
import torch
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import find_raptor_checkpoint
from raptor_wrapper import RaptorWrapper
from dino_wrapper import DinoModelWrapper
from trainer import run_train

from paths import PRJ_DIR, NYUD_DIR as DATA_FOLDER
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description="Train Raptor/Dino Probe on NYUD")
    parser.add_argument("--variant", type=str, required=True, choices=["raptor2", "raptor3", "raptor4", "dino_s", "dino_b"], help="Model variant")
    parser.add_argument("--seed", type=int, default=42, help="Seed for probe training (initialization, shuffling)")
    parser.add_argument("--model_seed", type=int, help="Seed of the pre-trained Raptor model (required for Raptor variants)")
    parser.add_argument("--output_dir", type=str, default="probes", help="Directory to save the classifier")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    # Configuration
    NYU_MIN, NYU_MAX = 0.001, 10.0
    GRID_SIZE = (35, 46)
    IMG_SIZE = (480, 640)
    
    # Generate output filename - include both seeds if relevant
    if args.variant.startswith("raptor"):
        if args.model_seed is None:
            parser.error("--model_seed is required for Raptor variants")
        output_filename = f"{args.variant}_classifier_modelseed_{args.model_seed}_probeseed_{args.seed}.pt"
    else:
        output_filename = f"{args.variant}_classifier_probeseed_{args.seed}.pt"
        
    classifier_save_path = os.path.join(args.output_dir, output_filename)
    print(f"Saving classifier to: {classifier_save_path}")

    if args.variant.startswith("raptor"):
        # Raptor Training
        try:
            raptor_model_path = find_raptor_checkpoint(args.variant, args.model_seed, BASE_DIR)
            print(f"Loading Raptor model from: {raptor_model_path}")
        except Exception as e:
            print(f"Error finding model: {e}")
            sys.exit(1)
            
        dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        model = torch.load(raptor_model_path, map_location=device)
        model = RaptorWrapper(model, dino_model, GRID_SIZE, NYU_MIN, NYU_MAX)
        
        for p in model.dino.parameters():
            p.requires_grad_(False)
        for p in model.raptor.parameters():
            p.requires_grad_(False)
        model.dino.eval()
        model.raptor.eval()
        model.classifier.requires_grad_(True)
        
    elif args.variant.startswith("dino"):
        # Dino Training
        dino_name = "dinov2_vits14_reg" if args.variant == "dino_s" else "dinov2_vitb14_reg"
        print(f"Loading Dino model: {dino_name}")
        model = DinoModelWrapper(dino_model=dino_name, device=device, grid_size=GRID_SIZE,
                                 min_depth=NYU_MIN, max_depth=NYU_MAX).to(device)
        
        for p in model.dino.parameters():
            p.requires_grad_(False)
        model.dino.eval()
        model.classifier.requires_grad_(True)
        
    else:
        print("Unknown variant")
        sys.exit(1)
            
    model = model.to(device).float()
    
    run_train(model, device, DATA_FOLDER, classifier_save_path, IMG_SIZE, NYU_MIN, NYU_MAX, args.seed, args.variant, model_seed=args.model_seed)

if __name__ == "__main__":
    main()
