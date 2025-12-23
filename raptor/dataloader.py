import threading
import queue
import time
import numpy as np
import torch
import zarr
from PIL import Image
import torchvision.transforms as transforms


class AsynchZarrLoader:
    def __init__(self, zarr_path, layer_start, layer_end, batch_size=128, num_workers=8, queue_size=30, device='cuda'):
        self.data = zarr.open(zarr_path, mode='r')
        self.layer_slice = slice(layer_start, layer_end + 1)  # end of slice is exclusive
        self.batch_size = batch_size
        self.device = device
        self.queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.num_workers = num_workers
        self.n_samples = self.data.shape[0]
        self.indices = np.arange(0, self.n_samples, self.batch_size)

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                idx = self.indices_q.get_nowait()
            except queue.Empty:
                break
            start = idx
            end = min(start + self.batch_size, self.n_samples)
            batch = self.data[start:end, self.layer_slice]
            tensor = torch.from_numpy(batch.astype(np.float32)).pin_memory().to(self.device, non_blocking=True)
            self.queue.put(tensor)

    def __iter__(self):
        while True:
            np.random.shuffle(self.indices)
            self.indices_q = queue.Queue()
            for idx in self.indices:
                self.indices_q.put(idx)

            threads = []
            self.stop_event.clear()
            for _ in range(self.num_workers):
                t = threading.Thread(target=self._worker)
                t.start()
                threads.append(t)

            try:
                active = True
                while active or not self.queue.empty():
                    # check if we should stop early (e.g. from close())
                    if self.stop_event.is_set():
                        break

                    try:
                        # periodically check for stop_event if get blocks
                        yield self.queue.get(timeout=0.1)
                    except queue.Empty:
                        active = any(t.is_alive() for t in threads)
            finally:
                self.stop_event.set()
                # drain queue to unblock any workers waiting on put()
                # we loop for a bit to ensure all workers have a chance to unblock
                shutdown_start = time.time()
                while any(t.is_alive() for t in threads):
                    while not self.queue.empty():
                        try:
                            self.queue.get_nowait()
                        except queue.Empty:
                            break

                    # wait a bit for threads to progress
                    time.sleep(0.05)

                    if time.time() - shutdown_start > 5.0:
                        print("WARNING: AsynchZarrLoader threads did not exit within 5 seconds.")
                        break

                # wait for threads to finish (with timeout to avoid hanging)
                for t in threads:
                    t.join(timeout=1.0)

    def close(self):
        self.stop_event.set()


def imagenet_transform():
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    return transforms.Compose([
        transforms.Resize(256, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN,
                             std=IMAGENET_DEFAULT_STD),
    ])
