import os
import random
import numpy as np
import torch
from PIL import Image

import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset

import sys
sys.path.append("../src/")
from paths import IMAGENET_TRAIN_DIR, DATA_DIR

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
transform = transforms.Compose([
    transforms.Resize(256, interpolation=Image.BICUBIC, antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
])

# Create output directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
dino = dino.eval().cuda().float()

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

batch_size = 256
dataset = datasets.ImageFolder(root=IMAGENET_TRAIN_DIR, transform=transform)

shuffled_indices = list(range(len(dataset)))
random.seed(seed)
random.shuffle(shuffled_indices)
shuffled_dataset = Subset(dataset, shuffled_indices)

dataloader = DataLoader(
    shuffled_dataset, batch_size=batch_size, shuffle=False,
    num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2
)

def inference(dino, x):
    with torch.no_grad():
        x = x.cuda().float()
        a = dino.prepare_tokens_with_masks(x, None)

        activations = torch.empty(13, *a.shape, device=a.device, dtype=a.dtype)
        activations[0] = a

        for i in range(12):
            a = dino.blocks[i](a)
            activations[i + 1] = a

        return activations.transpose(0, 1)

import os
import queue
import threading
import numpy as np
import torch
from tqdm import tqdm
import zarr
from numcodecs import Blosc

# Zarr parameters
num_samples = len(dataset)
num_layers = 13
num_tokens = 261
dim = 768
chunk_batch = 128

store_path = os.path.join(DATA_DIR, 'activations.zarr')

z = zarr.open(
    store=store_path,
    mode='w',
    shape=(num_samples, num_layers, num_tokens, dim),
    chunks=(chunk_batch, 1, num_tokens, dim),
    dtype='float32',
    compressor=Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE),
    zarr_version=2
)

# Writer queue
write_queue = queue.Queue(maxsize=24)

def writer_thread(zarr_array, q):
    while True:
        item = q.get()
        if item is None:
            q.task_done()
            break
        index, acts_np = item
        try:
            assert acts_np.shape[1:] == (num_layers, num_tokens, dim), f"Wrong shape: {acts_np.shape}"
            assert acts_np.dtype == np.float32, f"Wrong dtype: {acts_np.dtype}"
            assert index + acts_np.shape[0] <= num_samples, "Index overflow"

            zarr_array[index: index + acts_np.shape[0]] = acts_np
        except Exception as e:
            print(f"[Writer Error] At index {index}: {e}")
        q.task_done()

# Launch background writer thread
writer = threading.Thread(target=writer_thread, args=(z, write_queue))
writer.start()

# Main loop
index = 0
for batch in tqdm(dataloader, desc="Writing activations to Zarr"):
    x, _ = batch
    x = x.cuda(non_blocking=True)

    with torch.no_grad():
        acts = inference(dino, x)  # shape: (B, 13, 261, 768)
        acts_np = acts.to('cpu', non_blocking=True).float().numpy()

    write_queue.put((index, acts_np))
    index += acts_np.shape[0]

# Flush and stop
write_queue.join()
write_queue.put(None)
writer.join()

print("✅ All activations written successfully.")

#import zarr
#z = zarr.open(DATA_DIR + '/activations.zarr', mode='r')
#print(z.shape)  # should match expected total samples