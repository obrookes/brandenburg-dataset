import os
import json
import mmcv
import torch
import pickle
import torchaudio
import torchvision
from tqdm import tqdm
from PIL import Image
from glob import glob
from torch.utils.data import Dataset
from typing import Callable, Optional
from torchvision.transforms.functional import resize


class BrandenburgDataset(Dataset):
    """
    Base class for Brandenburg Dataset
    """

    def __init__(
        self,
        data_dir: str = None,
        ann_dir: str = None,
        sequence_len: int = None,
        sample_itvl: int = None,
        stride: int = None,
        split: str = None,
        transform: Optional[Callable] = None,
    ):
        super(BrandenburgDataset, self).__init__()

        self.data_path = data_dir
        self.ann_path = ann_dir
        self.data = glob(f"{data_dir}/**/*.MP4", recursive=True)

        # Frame offset
        self.offset = 5

        # Number of frames in sequence
        self.sequence_len = sequence_len * self.offset

        # Sample every i'th frame
        self.sample_itvl = sample_itvl

        # Frames required to build samples
        self.total_seq_len = self.sequence_len * self.sample_itvl

        # Frames between samples
        if stride is None:
            # This leaves no overlap between samples
            self.stride = self.total_seq_len
        else:
            self.stride = stride

        self.split = split
        self.transform = transform
        self.samples = {}
        self.labels = 0
        self.initialise_dataset()
        self.samples_by_video = {}

    def check_animal_exists(self, ann, frame_no, current_animal):
        animal = False
        frame = self.get_frame(ann, frame_no)
        for d in frame["detections"]:
            if d["id"] == current_animal:
                animal = True
        return animal

    def check_sufficient_animals(self, ann, current_ape, frame_no):
        for look_ahead_frame_no in range(frame_no, frame_no + self.total_seq_len):
            animal = self.check_animal_exists(ann, look_ahead_frame_no, current_ape)
            if not animal:
                return False
        return True

    def print_samples(self):
        print(self.samples)

    def __len__(self):
        samples = 0
        for video in self.samples.keys():
            samples += len(self.samples[video])
        return samples

    def load_annotation(self, base_path):
        with open(f"{base_path}/results.json", "rb") as handle:
            annotation = json.load(handle)
            return annotation

    def get_animal_ids(self, annotation):
        animal_ids = []
        for frame in annotation["images"]:
            for d in frame["detections"]:
                if d["id"] not in animal_ids:
                    animal_ids.append(d["id"])
        return max(animal_ids)

    def get_label(self, base_path):
        """split path and extract label as lowercase."""
        label = "_".join([x.lower() for x in base_path.split("/")[1].split(" ")])
        return label

    def get_valid_frames(self, ann, current_animal, frame_no, no_of_frames):
        valid_frames = 0

        for look_ahead_frame_no in range(frame_no, no_of_frames + 1, self.offset):
            animal = self.check_animal_exists(ann, look_ahead_frame_no, current_animal)
            if animal:
                valid_frames += self.offset
            else:
                return valid_frames

        return valid_frames

    def initialise_dataset(self):
        for data in tqdm(self.data):

            video = mmcv.VideoReader(data)
            base_path = "/".join(data.split("/")[:-1])
            annotation = self.load_annotation(base_path)
            label = self.get_label(base_path)

            # Check no of frames match
            no_of_frames = len(video)

            animal_ids = self.get_animal_ids(annotation)
            no_of_animals = animal_ids

            if not animal_ids:
                break

            for current_animal in range(0, no_of_animals + 1):

                frame_no = 0

                while frame_no <= len(video):
                    if (
                        len(video) - frame_no
                    ) < self.total_seq_len:  # TODO: check equality symbol is correct
                        break

                    animal = self.check_animal_exists(
                        annotation, frame_no, current_animal
                    )

                    if not animal:
                        frame_no += self.offset
                        continue

                    valid_frames = self.get_valid_frames(
                        annotation, current_animal, frame_no, no_of_frames
                    )

                    last_valid_frame = frame_no + valid_frames

                    for valid_frame_no in range(
                        frame_no, last_valid_frame, self.stride
                    ):
                        if (
                            valid_frame_no + max(self.total_seq_len, self.stride)
                            >= last_valid_frame
                        ):
                            correct_animal = False

                            for temporal_frame in range(
                                valid_frame_no,
                                self.total_seq_len,
                            ):
                                animal = self.check_ape_exists(
                                    annotation, temporal_frame, current_animal
                                )

                                if not animal:
                                    correct_animal = False
                                    break

                            if not correct_animal:
                                break

                        if (no_of_frames - valid_frame_no) >= self.total_seq_len:

                            if data not in self.samples.keys():
                                self.samples[data] = []

                            self.labels += 1
                            self.samples[data].append(
                                {
                                    "id": current_animal,
                                    "species": label,
                                    "start_frame": valid_frame_no,
                                }
                            )

                    frame_no = last_valid_frame

    # Get the ith sample from the dataset
    def find_sample(self, index):
        current_index = 0

        for key in self.samples.keys():
            for i, value in enumerate(self.samples[key]):
                if current_index == index:
                    return (
                        self.samples[key][i]["id"],
                        self.samples[key][i]["species"],
                        self.samples[key][i]["start_frame"],
                        key,
                    )
                current_index += 1

    def get_frame(self, annotation, frame_idx):
        for frame in annotation["images"]:
            if frame["frame_id"] == frame_idx:
                return frame

    def get_animal(self, frame, id):
        for d in frame["detections"]:
            if d["id"] == id:
                return d

    def get_coords(self, annotation, animal_id, frame_idx):
        frame = self.get_frame(annotation, frame_idx)
        animal = self.get_animal(frame, animal_id)
        return animal["bbox"]

    def build_spatial_sample(self, video_path, animal_id, frame_idx):

        video = mmcv.VideoReader(video_path)
        dims = (video.width, video.height)

        base_path = "/".join(video_path.split("/")[:-1])
        annotation = self.load_annotation(base_path)

        spatial_sample = []
        for i in range(0, self.total_seq_len, 5):
            frame = video[frame_idx + i]
            spatial_img = Image.fromarray(frame)
            coords = self.normalised_xywh_to_xywh(
                self.get_coords(annotation, animal_id, frame_idx), dims
            )
            coords = list(map(int, coords))
            cropped_img = spatial_img.crop(coords)
            spatial_data = self.transform(cropped_img)

            spatial_sample.append(spatial_data.squeeze_(0))
        spatial_sample = torch.stack(spatial_sample, dim=0)
        spatial_sample = spatial_sample.permute(0, 1, 2, 3)

        # Check frames in sample match sequence length
        return spatial_sample

    def normalised_xywh_to_xywh(self, bbox, dims):
        """Normalised [x, y, w, h] from Megadetector to [x1, y1, x2, y2]"""
        width = dims[0]
        height = dims[1]

        x1 = bbox[0] * width
        y1 = bbox[1] * height

        x2 = (bbox[0] + bbox[2]) * width
        y2 = (bbox[1] + bbox[3]) * height

        return x1, y1, x2, y2

    def resize_stack_seq(self, sequence, size):
        sequence = [resize(x, size=size).squeeze(dim=0) for x in sequence]
        return torch.stack(sequence, dim=0)

    def build_sample(self, name, ape_id, frame_idx):
        sample = dict()
        if "r" in self.type:
            video = self.get_video(name)
            sample["spatial_sample"] = self.build_spatial_sample(
                video, name, ape_id, frame_idx
            )
        return sample

    def __getitem__(self, index):
        id, species, start_frame, video = self.find_sample(index)
        sample = self.build_spatial_sample(video, id, start_frame)
        return sample, species
