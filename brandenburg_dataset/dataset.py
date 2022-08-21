import json
import mmcv
import torch
from tqdm import tqdm
from PIL import Image
from glob import glob
from torch.utils.data import Dataset
from typing import Callable, Optional
from torchvision.transforms.functional import resize


class BrandenburgDataset(Dataset):
    """
    Base class for Brandenburg Dataset

    Args:

     Paths:
         data_dir: Path to parent dir containing video files and annotations.

     Sample building:
         sequence_len: Number of frames in each sample. The output tensor
         will have shape (B x C x T x W x H) where B = batch_size, C = channels,
         T = sequence_len, W = width and H = height.

         sample_itvl: Number of frames between each sample frame i.e., if
         sample_itvl = 1 consecutive frames are sampled, if sample_itvl = i
         every i'th frame is sampled.

         *Note if sequence_len = 5 and sample_itvl = 2, the output tensor will
         be of shape (B x C x 5 x H x W).

         stride: Number of frames between samples. By default, this is
         sequence_len x sample_itvl. This means samples are built consecutively
         and without overlap. If the stride is manually set to a lower value
         samples will be generated with overlapping frames i.e., samples built
         with sequence_len = 20 and stride = 10 will have a 10-frame overlap.

     Transform:
         transform: List of transforms to be applied.
    """

    def __init__(
            self,
            data_dir: str = None,
            sequence_len: int = None,
            sample_itvl: int = None,
            stride: int = None,
            transform: Optional[Callable] = None,
    ):
        super(BrandenburgDataset, self).__init__()

        self.data_path = data_dir
        self.annotation_path = f"{data_dir.replace('/data/', '/annotations/')}"
        mp4 = glob(f"{data_dir}/**/*.MP4", recursive=True)
        avi = glob(f"{data_dir}/**/*.AVI", recursive=True)
        self.data = mp4 + avi

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
            self.stride = stride * self.offset

        self.transform = transform
        self.samples = {}
        self.class_dict = {}
        self.labels = 0
        self.initialise_dataset()
        self.samples_by_video = {}

    def check_animal_exists(self, ann, frame_no, current_animal):
        animal = False
        frame = self.get_frame(ann, frame_no)
        if frame is not None:
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

    def load_annotation(self, path):
        with open(path, "rb") as handle:
            annotation = json.load(handle)
            return annotation

    def get_animal_ids(self, annotation):
        animal_ids = []
        for frame in annotation["annotations"]:
            for d in frame["detections"]:
                if d["id"] not in animal_ids:
                    animal_ids.append(d["id"])
        if animal_ids:
            return max(animal_ids)
        else:
            return []

    def get_label(self, path):
        """split path and extract label as lowercase."""
        animal = path.split(self.data_path + '/')[-1].split('/')[0].lower()
        if animal not in self.class_dict.keys():
            self.class_dict[animal] = len(self.class_dict)
        return self.class_dict[animal]

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
            title = data.split('/')[-1].split('.')[0]
            annotation_path = f"{self.annotation_path}/{title}_track.json"
            annotation = self.load_annotation(annotation_path)
            label = self.get_label(annotation_path)

            # Check no of frames match
            no_of_frames = len(video)

            animal_ids = self.get_animal_ids(annotation)
            no_of_animals = animal_ids

            if animal_ids == []:
                continue

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
                                    valid_frame_no, self.total_seq_len, self.offset
                            ):
                                animal = self.check_animal_exists(
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
        for frame in annotation["annotations"]:
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
        title = video_path.split('/')[-1].split('.')[0]
        annotation_path = f"{self.annotation_path}/{title}_track.json"
        annotation = self.load_annotation(annotation_path)
        annotation = self.load_annotation(annotation_path)

        spatial_sample = []
        for i in range(0, self.total_seq_len, self.sample_itvl * self.offset):
            frame = video[frame_idx + i]
            spatial_img = Image.fromarray(frame)
            coords = list(
                map(float, self.get_coords(annotation, animal_id, frame_idx + i))
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
        sample = dict()
        id, species, start_frame, video = self.find_sample(index)
        sample["spatial_sample"] = self.build_spatial_sample(video, id, start_frame)
        return sample, species
