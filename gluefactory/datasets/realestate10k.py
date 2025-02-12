"""
RealEstate10K Dataset

This dataloader loads sequences from the RealEstate10K dataset.
Each sequence is stored under {data_dir}/{split}/<sequence_id>/data.npz.
The .npz file contains multiple frames (with sorted keys). Two
pairing strategies are provided via the configuration:
    1. "first_all": always pair the first frame against every subsequent frame.
    2. "skip": pair frame i with frame i+n, where n is set by "frame_gap".

The output from __getitem__ is a dictionary with keys 'name', 'sequence', 'view0', and 'view1',
consistent with the format of other dataloaders (e.g. for homographies and MegaDepth).
"""

# When running this file directly, set the package so that relative imports work
if __name__ == "__main__" and __package__ is None:
    __package__ = "gluefactory.datasets"

import os
from pathlib import Path
import numpy as np
import torch

from omegaconf import OmegaConf
from .base_dataset import BaseDataset
from .utils import numpy_image_to_torch  # helper to convert image to tensor
from ..utils.image import ImagePreprocessor

class RealEstate10K(BaseDataset, torch.utils.data.Dataset):
    default_conf = {
        "data_root": "data",
        "data_dir": "realestate",       # Directory containing npz sequences in DATA_PATH/{data_root}/{data_dir}/...
        "cam_txt_dir": "RealEstate10K", # Directory containing txt files in DATA_PATH/{data_root}/{cam_txt_dir}/...
        # Old behavior: if split_percentages is not provided, the data is expected under DATA_PATH/{data_root}/{data_dir}/{split}
        "split": "test",
        "pairing": "first_all",          # Options: "first_all" or "skip"
        "frame_gap": 5,                  # Used for both pairing and single mode
        "mode": "single",                # New flag: "pair" or "single". "pair" returns paired views; "single" returns single (consecutive) frames.
        "preprocessing": ImagePreprocessor.default_conf,
        "num_items": 0,                # How many video sequences (.npz files) to load. 0 or None means load all.
        "split_percentages": None,     # e.g. {"train": 0.8, "val": 0.2}. If provided, the dataset will partition all sequences into train/val splits.
    }

    def _init(self, conf):
        self.preprocessor = ImagePreprocessor(conf.preprocessing)
        self.cam_params = {}     # dict: sequence name -> list of camera parameters per frame
        self.sequence_keys = {}  # dict: sequence name -> list of frame keys (sorted)
        self.sequence_data = []  # list to hold sequences and their frame pairs

        # New mode: if split_percentages is provided, ignore conf.split and partition sequences
        if conf.split_percentages is not None:
            seq_root = Path(conf.data_root) / conf.data_dir
            all_sequences = sorted([seq for seq in seq_root.iterdir() if seq.is_dir()])
            if conf.num_items and conf.num_items > 0:
                all_sequences = all_sequences[:conf.num_items]
            self.all_sequences = all_sequences
        else:
            # Old mode: use split subfolder.
            self.root = Path(conf.data_root) / conf.data_dir / conf.split
            if not self.root.exists():
                raise FileNotFoundError(f"Dataset directory {self.root} not found.")
            self.sequences = sorted([seq for seq in self.root.iterdir() if seq.is_dir()])
            if conf.num_items and conf.num_items > 0:
                self.sequences = self.sequences[:conf.num_items]

        for seq in self.sequences:
            npz_path = seq / "data.npz"
            if not npz_path.exists():
                continue
            try:
                data = np.load(npz_path)
            except Exception as e:
                print(f"Could not load {npz_path}: {e}")
                continue
            frame_keys = sorted(data.files)
            # Store the frame keys for later lookup (to map key to frame index)
            self.sequence_keys[seq.name] = frame_keys

            # NEW: try to load camera parameters from a corresponding txt file.
            txt_path = Path(conf.data_root) / conf.cam_txt_dir / conf.split / f"{seq.name}.txt"
            if txt_path.exists():
                try:
                    with open(txt_path, "r") as f:
                        _ = f.readline()
                        cam_data = np.loadtxt(f, dtype=str)
                        if cam_data.ndim == 1:
                            cam_data = np.expand_dims(cam_data, axis=0)
                        self.cam_params[seq.name] = cam_data.tolist()
                except Exception as e:
                    print(f"Could not load camera parameters from {txt_path}: {e}")
            # End NEW

            if conf.mode == "pair":
                if len(frame_keys) < 2:
                    continue  # skip sequences with less than 2 frames for pairing mode

                # Compute the list of pairs for the current sequence using the selected pairing strategy
                if conf.pairing == "first_all":
                    gap = conf.frame_gap
                    pairs = [(frame_keys[0], frame_keys[i]) for i in range(gap, len(frame_keys), gap)]
                elif conf.pairing == "skip":
                    gap = conf.frame_gap
                    pairs = []
                    for i in range(0, len(frame_keys) - gap, gap):
                        pairs.append((frame_keys[i], frame_keys[i + gap]))
                else:
                    raise ValueError(f"Unknown pairing strategy: {conf.pairing}")

                if pairs:
                    self.sequence_data.append({
                        "sequence": seq.name,
                        "npz_path": npz_path,
                        "pairs": pairs,
                    })
            elif conf.mode == "single":
                # In single mode, simply sample frames using the frame_gap (starting at index 0)
                frames = [frame_keys[i] for i in range(0, len(frame_keys), conf.frame_gap)]
                if frames:
                    self.sequence_data.append({
                        "sequence": seq.name,
                        "npz_path": npz_path,
                        "frames": frames,
                    })
            else:
                raise ValueError(f"Unknown mode: {conf.mode}")

    def get_dataset(self, split):
        # If split_percentages is provided, partition self.all_sequences using the given percentages.
        if self.conf.split_percentages is not None:
            total = len(self.all_sequences)
            train_pct = float(self.conf.split_percentages.get("train", 0))
            train_count = int(total * train_pct)
            if split == "train":
                selected_sequences = self.all_sequences[:train_count]
            elif split == "val":
                selected_sequences = self.all_sequences[train_count:]
            else:
                raise ValueError(f"Split {split} not recognized when using split_percentages.")
        else:
            selected_sequences = self.sequences

        # Instead of rebuilding self.items, now rebuild self.sequence_data.
        self.sequence_data = []
        for seq in selected_sequences:
            npz_path = seq / "data.npz"
            if not npz_path.exists():
                continue
            try:
                data = np.load(npz_path)
            except Exception as e:
                print(f"Could not load {npz_path}: {e}")
                continue
            frame_keys = sorted(data.files)
            self.sequence_keys[seq.name] = frame_keys

            # Load camera parameters from corresponding txt file.
            txt_path = Path(self.conf.data_root) / self.conf.cam_txt_dir / split / f"{seq.name}.txt"
            if txt_path.exists():
                try:
                    with open(txt_path, "r") as f:
                        _ = f.readline()  # skip YouTube URL
                        cam_data = np.loadtxt(f, dtype=str)
                        if cam_data.ndim == 1:
                            cam_data = np.expand_dims(cam_data, axis=0)
                        self.cam_params[seq.name] = cam_data.tolist()
                except Exception as e:
                    print(f"Could not load camera parameters from {txt_path}: {e}")

            if self.conf.mode == "pair":
                if len(frame_keys) < 2:
                    continue  # skip sequences with less than 2 frames for pairing mode

                # Compute the list of pairs for the current sequence using the selected pairing strategy
                if self.conf.pairing == "first_all":
                    gap = self.conf.frame_gap
                    pairs = [(frame_keys[0], frame_keys[i]) for i in range(gap, len(frame_keys), gap)]
                elif self.conf.pairing == "skip":
                    gap = self.conf.frame_gap
                    pairs = []
                    for i in range(0, len(frame_keys) - gap, gap):
                        pairs.append((frame_keys[i], frame_keys[i + gap]))
                else:
                    raise ValueError(f"Unknown pairing strategy: {self.conf.pairing}")

                if pairs:
                    self.sequence_data.append({
                        "sequence": seq.name,
                        "npz_path": npz_path,
                        "pairs": pairs,
                    })
            elif self.conf.mode == "single":
                frames = [frame_keys[i] for i in range(0, len(frame_keys), self.conf.frame_gap)]
                if frames:
                    self.sequence_data.append({
                        "sequence": seq.name,
                        "npz_path": npz_path,
                        "frames": frames,
                    })
            else:
                raise ValueError(f"Unknown mode: {self.conf.mode}")
        return self

    def __getitem__(self, idx):
        # Now, idx corresponds to one video sequence
        seq_entry = self.sequence_data[idx]
        seq_name = seq_entry["sequence"]
        npz_path = seq_entry["npz_path"]
        if self.conf.mode == "pair":
            data_list = seq_entry["pairs"]
        elif self.conf.mode == "single":
            data_list = seq_entry["frames"]
        else:
            raise ValueError(f"Unknown mode: {self.conf.mode}")

        try:
            data = np.load(npz_path)
        except Exception as e:
            raise IOError(f"Could not load {npz_path}: {e}")

        def parse_cam_line(row):
            """
            Parse a line from the camera TXT file.
            Expected format (19 tokens):
              row[0]: timestamp (ignored)
              row[1:5]: [focal_length_x, focal_length_y, principal_point_x, principal_point_y]
              row[7:]: camera pose (12 numbers reshaped to 3x4)
            """
            if len(row) != 19:
                raise ValueError(f"Expected 19 columns in camera file, got {len(row)}")
            fx, fy, pp_x, pp_y = map(float, row[1:5])
            pose_vals = list(map(float, row[7:]))
            return fx, fy, pp_x, pp_y, np.array(pose_vals).reshape(3, 4)

        if self.conf.mode == "pair":
            pair_outputs = []
            for key0, key1 in data_list:
                try:
                    img0 = data[key0]
                    img1 = data[key1]
                except Exception as e:
                    raise IOError(f"Could not load frame data from {npz_path}: {e}")
                # Convert images to torch tensors via the designated preprocessor.
                view0 = self.preprocessor(numpy_image_to_torch(img0))
                view1 = self.preprocessor(numpy_image_to_torch(img1))

                cam_info = {}
                if seq_name in self.cam_params and seq_name in self.sequence_keys:
                    keys = self.sequence_keys[seq_name]
                    try:
                        idx_frame0 = keys.index(key0)
                        idx_frame1 = keys.index(key1)
                    except ValueError:
                        idx_frame0, idx_frame1 = None, None

                    if idx_frame0 is not None and idx_frame1 is not None:
                        row0 = self.cam_params[seq_name][idx_frame0]
                        row1 = self.cam_params[seq_name][idx_frame1]
                        try:
                            fx0, fy0, pp_x0, pp_y0, P0 = parse_cam_line(row0)
                            fx1, fy1, pp_x1, pp_y1, P1 = parse_cam_line(row1)
                            h0, w0 = img0.shape[0:2]
                            h1, w1 = img1.shape[0:2]
                            K0 = np.array([[w0 * fx0, 0,       w0 * pp_x0],
                                           [0,       h0 * fy0, h0 * pp_y0],
                                           [0,       0,        1]])
                            K1 = np.array([[w1 * fx1, 0,       w1 * pp_x1],
                                           [0,       h1 * fy1, h1 * pp_y1],
                                           [0,       0,        1]])
                            cam_info = {"K0": K0, "P0": P0, "K1": K1, "P1": P1}
                        except Exception as e:
                            print(f"Error parsing camera parameters for sequence {seq_name}: {e}")
                out_pair = {
                    "name": f"{seq_name}/{key0}_{key1}",
                    "view0": view0,
                    "view1": view1,
                }
                out_pair.update(cam_info)
                pair_outputs.append(out_pair)

            return {"sequence": seq_name, "pairs": pair_outputs}
        elif self.conf.mode == "single":
            frames_list = seq_entry["frames"]
            frame_outputs = []
            for key in frames_list:
                try:
                    img = data[key]
                except Exception as e:
                    raise IOError(f"Could not load frame data from {npz_path}: {e}")
                frame = self.preprocessor(numpy_image_to_torch(img))

                cam_info = {}
                if seq_name in self.cam_params and seq_name in self.sequence_keys:
                    keys = self.sequence_keys[seq_name]
                    try:
                        idx_frame = keys.index(key)
                    except ValueError:
                        idx_frame = None

                    if idx_frame is not None:
                        row = self.cam_params[seq_name][idx_frame]
                        try:
                            fx, fy, pp_x, pp_y, P = parse_cam_line(row)
                            h, w = img.shape[0:2]
                            K = np.array([[w * fx, 0,       w * pp_x],
                                          [0,       h * fy, h * pp_y],
                                          [0,       0,      1]])
                            cam_info = {"K": K, "P": P}
                        except Exception as e:
                            print(f"Error parsing camera parameters for sequence {seq_name}: {e}")
                out_frame = {"name": f"{seq_name}/{key}", "frame": frame}
                out_frame.update(cam_info)
                frame_outputs.append(out_frame)

            return {"sequence": seq_name, "frames": frame_outputs}
        else:
            raise ValueError(f"Unknown mode: {self.conf.mode}")

    def __len__(self):
        # Return number of sequences rather than number of individual pair items
        return len(self.sequence_data)

if __name__ == "__main__":
    #   Example usage:
    #     python gluefactory/datasets/realestate10k.py --dotlist data_root=your_path/realestate split=test pairing=first_all num_items=50
    #     python gluefactory/datasets/realestate10k.py --dotlist data_root=your_path/realestate split=train pairing=first_all num_items=100 split_percentages.train=0.8 split_percentages.val=0.2
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--dotlist", nargs="*", default=[], help="dotlist configuration overrides")
    args = parser.parse_args()
    # Merge default config with command-line dotlist arguments
    conf = OmegaConf.merge(RealEstate10K.default_conf, OmegaConf.from_dotlist(args.dotlist))
    dataset = RealEstate10K(conf)
    print(f"Loaded {len(dataset)} video sequences.")
    sample = dataset[0]
    print("Sample video sequence info:")
    print(f"Name: {sample['sequence']}")
    if conf.mode == "pair":
        print(f"Number of pairs: {len(sample['pairs'])}")
        print(f"Camera intrinsics: {sample['pairs'][0].get('K0', 'N/A')}")
        print(f"Camera pose: {sample['pairs'][0].get('P0', 'N/A')}")
        print(f"Camera intrinsics: {sample['pairs'][0].get('K1', 'N/A')}")
        print(f"Camera pose: {sample['pairs'][0].get('P1', 'N/A')}")
    elif conf.mode == "single":
        print(f"Number of frames: {len(sample['frames'])}")
        print(f"Camera intrinsics: {sample['frames'][0].get('K', 'N/A')}")
        print(f"Camera pose: {sample['frames'][0].get('P', 'N/A')}")

    # Print the length of all video sequences
    for seq in dataset:
        if conf.mode == "pair":
            print(f"Sequence: {seq['sequence']}, Num pairs: {len(seq['pairs'])}")
        elif conf.mode == "single":
            print(f"Sequence: {seq['sequence']}, Num frames: {len(seq['frames'])}")

