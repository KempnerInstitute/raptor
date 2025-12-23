import numpy as np
import pytest
import torch

zarr = pytest.importorskip("zarr")
pytest.importorskip("torchvision")
pytest.importorskip("PIL")

from dataloader import AsynchZarrLoader, imagenet_transform


def test_asynch_zarr_loader_yields_batches(tmp_path):
    data_path = tmp_path / "data.zarr"
    data = zarr.open(str(data_path), mode="w", shape=(10, 3, 2, 4), dtype="f4")
    data[:] = np.random.rand(10, 3, 2, 4).astype("f4")

    loader = AsynchZarrLoader(
        zarr_path=str(data_path),
        layer_start=0,
        layer_end=1,
        batch_size=4,
        num_workers=1,
        queue_size=2,
        device="cpu",
    )

    iterator = iter(loader)
    batch = next(iterator)
    loader.close()
    iterator.close()

    assert batch.shape == (4, 2, 2, 4)
    assert batch.dtype == torch.float32


def test_imagenet_transform_output_shape():
    from PIL import Image

    img = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8"))
    tensor = imagenet_transform()(img)

    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
