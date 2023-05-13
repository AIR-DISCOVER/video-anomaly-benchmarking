"""
Given a generated `train.csv` or `valid.csv`, generate file for further evaluation on a sequence basis.
Output eval_helper.json
"""

import os
import argparse
import json
import re


class ImageLabelPair:
    def __init__(self, dataset_root: str, image_pair: str, global_idx: int):
        # Example: train/seq05-2/rgb_v/157.png,train/seq05-2/mask_id_v/157.png
        image, label = image_pair.split(",")
        self.image_path = os.path.join(dataset_root, image)
        self.label_path = os.path.join(dataset_root, label)
        self.depth_path = self.label_path.replace('mask_id_v', 'depth_v')
        self.global_idx = global_idx
        
        # Line k:
        # k: Transform(Location(x=182.332687, y=52.207020, z=1.806737), Rotation(pitch=0.273139, yaw=-0.060333, roll=-0.000549))
        extrinsics_filepath = os.path.join(os.path.dirname(os.path.dirname(self.image_path)), "path.txt")
        if not os.path.exists(extrinsics_filepath):
            extrinsics_filepath = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(self.image_path)), "../../path.txt"))
        
        # extract k-th line of path.txt
        with open(extrinsics_filepath, 'r') as f:
            extrinsics_line = f.readlines()[self.get_seq_idx() - 1].strip()
        
        # include k in extraction pattern for verification
        # note for the negative sign when extracting elements
        pattern = r"{}: Transform\(Location\(x=(-?\d+\.\d+), y=(-?\d+\.\d+), z=(-?\d+\.\d+)\), Rotation\(pitch=(-?\d+\.\d+), yaw=(-?\d+\.\d+), roll=(-?\d+\.\d+)\)\)".format(self.get_seq_idx())
        match = re.match(pattern, extrinsics_line)
        assert match is not None
        
        x = float(match.group(1))
        y = float(match.group(2))
        z = float(match.group(3))
        pitch = float(match.group(4))
        yaw = float(match.group(5))
        roll = float(match.group(6))
        self.extrinsics = [x, y, z, pitch, yaw, roll]
        
        assert os.path.exists(self.image_path)
        assert os.path.exists(self.depth_path)
        assert os.path.exists(self.label_path)
    
    def get_seq_name(self):
        # Example: seq05-2
        return os.path.basename(os.path.dirname(os.path.dirname(self.label_path)))

    def get_seq_idx(self):
        # Example: 157
        return int(os.path.splitext(os.path.basename(self.image_path))[0])

    def to_dict(self):
        return {
            "image_path": self.image_path,
            "depth_path": self.depth_path,  # used for consistency check
            "label_path": self.label_path,
            "global_idx": self.global_idx,
            "extrinsics": self.extrinsics,
        }
    
    def __str__(self):
        return self.get_seq_name() + "_" + str(self.get_seq_idx())


class Sequence:
    def __init__(self, name: str) -> None:
        self.name = name
        self.image_label_pairs = []
    
    def add_image_label_pair(self, image_label_pair: ImageLabelPair):
        self.image_label_pairs.append(image_label_pair)
    
    def sort_image_label_pairs(self):
        self.image_label_pairs.sort(key=lambda x: x.get_seq_idx())
    
    def check_contiguous(self):
        seq_idx = [pair.get_seq_idx() for pair in self.image_label_pairs]
        return all([seq_idx[i] + 1 == seq_idx[i + 1] for i in range(len(seq_idx) - 1)])

    def to_dict(self):
        return {
            self.name: [pair.to_dict() for pair in self.image_label_pairs]
        }

    def __str__(self):
        return self.name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str)
    parser.add_argument("--out", type=str, default="eval_helper.json")
    opt = parser.parse_args()
    
    assert opt.csv.endswith(".csv")
    assert os.path.exists(opt.csv)
    
    dataset_root = os.path.dirname(opt.csv)
    
    path2seq_mapping = {}
    with open(opt.csv, "r") as f:
        current_line_idx = 0
        for line in f:
            image_label_pair = ImageLabelPair(dataset_root, line.strip(), current_line_idx)
            seq_name = image_label_pair.get_seq_name()
            if seq_name not in path2seq_mapping:
                path2seq_mapping[seq_name] = Sequence(seq_name)
            path2seq_mapping[seq_name].add_image_label_pair(image_label_pair)
            current_line_idx += 1
    
    for seq in path2seq_mapping.values():
        seq.sort_image_label_pairs()
        assert seq.check_contiguous()
        
    path2seq_mapping = {k: v.to_dict()[k] for k, v in path2seq_mapping.items()}

    with open(opt.out, "w") as f:
        f.write(json.dumps(path2seq_mapping, indent=4, ensure_ascii=False))
