from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data(
    *, data_dir, batch_size, slice_size, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param slice_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files, all_embeds = _list_files_recursively(data_dir)

    # classes = None
    # if class_cond:
    #     # Assume classes are the first part of the filename,
    #     # before an underscore.
    #     class_names = [bf.basename(path).split("_")[0] for path in all_files]
    #     sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
    #     classes = [sorted_classes[x] for x in class_names]

    dataset = ImageDataset(
        slice_size,
        all_files,
        embed_paths=all_embeds,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader

def _list_files_recursively(data_dir):
    results = []
    embeds = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["npy"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_files_recursively(full_path))
    return results, embeds

# TODO: get audio and corresponding CLIP embeddings npy files
def _list_files(data_dir):
    results = []
    embeds = []
    all_dirs = sorted(bf.listdir(data_dir))
    print(f'Found {len(all_dirs)} pairs of audio and frames.')
    # try catch for audio and frames
    try:
        for entry in all_dirs:
            curr_dir = bf.join(data_dir, entry)
            if bf.isdir(curr_dir):
                all_subdirs = sorted(bf.listdir(curr_dir))
                frame_dirs = list(filter(lambda x: bf.isdir(bf.join(curr_dir, x)), all_subdirs))
                assert len(all_subdirs) - len(frame_dirs) == 1
                audio_path = list(filter(lambda x: not bf.isdir(x), all_subdirs))[0]
                assert audio_path.split(".")[-1] == "npy"

                frame_dir = np.random.choice(frame_dirs)
                frame_path = bf.join(curr_dir, frame_dir, f"{frame_dir}_features.npy")
                audio_path = bf.join(curr_dir, audio_path)

                results.append(audio_path)
                embeds.append(frame_path)
    except Exception as e:
        print(f'Error reading files: {e}')

    print(f'Loaded {len(results)} pairs of audio and frames.')
    return results, embeds


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, embed_paths=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_embeds = None if embed_paths is None else embed_paths[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        spectrogram = np.load(path, dtype=np.float32) # (H, W)
        
        # Randomly extract a [R,R] slice (R=64)
        assert spectrogram.shape[1] == self.resolution
        if spectrogram.shape[0] >= self.resolution:
            start_x = np.random.randint(0, spectrogram.shape[0] - self.resolution + 1)
            spectrogram_slice = spectrogram[start_x:start_x + self.resolution, :]
        else:
            raise ValueError("Spectrogram is too small to extract a slice.")

        # Normalize values from [-12, 3] to [-1, 1]
        spectrogram_slice = (spectrogram_slice + 12) / 15.0 * 2 - 1

        out_dict = {}
        if self.local_embeds is not None:
            out_dict["y"] = np.load(self.local_embeds[idx], dtype=np.float32) # (D,)
            # out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.expand_dims(spectrogram_slice, axis=0), out_dict # (1, R, R), (D,)

if __name__ == "__main__":
    data_dir = "/home/ywn1043/audio-diff/sample-data"
    _list_files(data_dir)