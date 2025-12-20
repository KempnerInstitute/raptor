PRJ_DIR = 'absolute_path/to/raptor/'
IMAGENET_TRAIN_DIR = "/absolute_path/to/imagenet/train" # in this dir, there should be subdir like n02086240 filled with images
IMAGENET_VAL_DIR = "/absolute_path/to/imagenet/val" # in this dir, there should be subdir like n02086240 filled with images
NYUD_DIR = "/absolute_path/to/nyu_data/" # should be kaggle: kagglehub/datasets/soumikrakshit/nyu-depth-v2/versions/1/nyu_data/
ADE20K_DIR = "/absolute_path/to/ADEChallengeData2016"
DATA_DIR = f"absolute_path/to/dinov2_b_reg4_float_zarr/activations.zarr" # where precomputed activations will be stored
MODEL_DIR = f'{PRJ_DIR}src/runs/models' # where models will be stored