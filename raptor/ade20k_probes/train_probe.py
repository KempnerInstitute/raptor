import argparse
import sys
import torch
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import find_raptor_checkpoint
from dino_wrapper import DinoModelWrapper
from raptor_wrapper import RaptorWrapper
from trainer import run_train

from paths import ADE20K_DIR, MODEL_DIR
DATA_PATH = ADE20K_DIR

def main():
    parser = argparse.ArgumentParser(description="Train Raptor/Dino Probe on ADE20k")
    parser.add_argument("--variant", type=str, required=True, choices=["raptor2", "raptor3", "raptor4", "dino_s", "dino_b"], help="Model variant")
    parser.add_argument("--seed", type=int, default=42, help="Seed for probe training (initialization, shuffling)")
    parser.add_argument("--model_seed", type=int, help="Seed of the pre-trained Raptor model (required for Raptor variants)")
    parser.add_argument("--output_dir", type=str, default="probes", help="Directory to save the classifier")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
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
            raptor_model_path = find_raptor_checkpoint(args.variant, args.model_seed, MODEL_DIR)
            print(f"Loading Raptor model from: {raptor_model_path}")
        except Exception as e:
            print(f"Error finding model: {e}")
            sys.exit(1)
            
        # Helper to load dino for raptor wrapper
        dino_wrapper = DinoModelWrapper(device=device)
        for p in dino_wrapper.dino.parameters():
            p.requires_grad_(False)
        dino_wrapper.classifier.requires_grad_(False)
        dino_wrapper = dino_wrapper.to(device)
        
        model = torch.load(raptor_model_path, map_location=device)
        model = RaptorWrapper(model, dino_wrapper.dino)
        
        for p in model.dino.parameters():
            p.requires_grad_(False)
        for p in model.raptor.parameters():
            p.requires_grad_(False)
        model.classifier.requires_grad_(True)
        model.dino.eval()
        model.raptor.eval()
        
    elif args.variant.startswith("dino"):
        # Dino Training
        dino_name = "dinov2_vits14_reg" if args.variant == "dino_s" else "dinov2_vitb14_reg"
        print(f"Loading Dino model: {dino_name}")
        model = DinoModelWrapper(dino_model=dino_name, device=device)
        for p in model.dino.parameters():
            p.requires_grad_(False)
        model.classifier.requires_grad_(True)
        model = model.to(device)
        model.dino.eval()
        
    else:
        print("Unknown variant")
        sys.exit(1)
    
    model = model.to(device).float()
    
    run_train(model, device, DATA_PATH, classifier_save_path, args.seed, args.variant, model_seed=args.model_seed)

if __name__ == "__main__":
    main()
