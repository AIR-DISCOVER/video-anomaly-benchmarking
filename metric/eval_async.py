import os
import json
import argparse
import numpy as np
from PIL import Image
from p_tqdm import p_map

from utils.measures import get_measures

train_id_in = 0
train_id_out = 1
anomaly_id = 254
ignore_id = 255
frame_time = 1 / 60
inv_score = False


def extract_single(eval_pair):
    
    pred_path, label_path = eval_pair

    # Load pred npy
    assert os.path.exists(pred_path)
    pred = np.load(pred_path)
    if inv_score:
        pred = -pred
    
    # Load gt anno
    assert os.path.exists(label_path)
    gt = np.asarray(Image.open(label_path))
    
    height, width = gt.shape
    
    if pred.ndim == 3:
        assert pred.shape[0] == 1
        pred = pred[0]
    assert pred.shape == gt.shape
    
    converted_gt = np.zeros_like(gt)
    converted_gt += train_id_in
    converted_gt[gt == anomaly_id] = train_id_out
    converted_gt[gt == ignore_id] = ignore_id  # ignored label
    
    out_scores = pred[converted_gt == train_id_out]
    in_scores = pred[converted_gt == train_id_in]
    
    if (len(out_scores) != 0) and (len(in_scores) != 0):
        auroc, aupr, fpr = get_measures(out_scores, in_scores)
    else:
        return None
    return auroc, aupr, fpr


def extract_batch(eval_pairs, num_cpus=16):
    aurocs, auprs, fprs = [], [], []
    output_list = p_map(extract_single, eval_pairs, num_cpus=num_cpus)
    for output in output_list:
        if output is not None:
            aurocs.append(output[0])
            auprs.append(output[1])
            fprs.append(output[2])
    auroc = np.mean(aurocs);
    aupr = np.mean(auprs);
    fpr = np.mean(fprs)
    return auroc, aupr, fpr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_helper", type=str, required=True, help="Path to eval_helper.json")
    parser.add_argument("--pred_list", type=str, required=True, help="A path to file. Each line is a path to a prediction file, with order corresponding to that of val.csv of dataset.")
    parser.add_argument("--inv_anomaly_score", action='store_true', help="Normally, predicted logits for anomaly objects should be larger than those in-distribution pixels. If this flag is set, we will take the negative value of the logits for evaluation.")
    parser.add_argument("--eval_time", type=float, required=True, help="How many seconds it takes to evaluate one image for your method.")
    parser.add_argument("--num_cpus", type=int, default=16)
    opt = parser.parse_args()
    
    if opt.inv_anomaly_score:
        inv_score = True
    
    # construct eval pairs using eval_helper
    eval_pairs = []  # List of (pred, label)
    
    with open(opt.eval_helper, 'r') as f:
        eval_helper = json.load(f)
    
    with open(opt.pred_list, 'r') as f:
        pred_list = f.read().strip().split('\n')
    
    for seq_name, seq_info_list in eval_helper.items():
        for i in range(len(seq_info_list)):
            seq_info = seq_info_list[i]
    
            # Example
            # {
            #     "image_path": "/data21/tb5zhh/datasets/anomaly_dataset/v5_release/val/seq09-5/rgb_v/2.png",
            #     "label_path": "/data21/tb5zhh/datasets/anomaly_dataset/v5_release/val/seq09-5/mask_id_v/2.png",
            #     "global_idx": 396
            # }
            
            pred_path = pred_list[seq_info['global_idx']]
            if not os.path.exists(pred_path):
                pred_path = os.path.join(os.path.dirname(opt.pred_list), pred_path)
                assert os.path.exists(pred_path)
            
            # You should correspond to label X frames later
            # where X = ceil(eval_time / frame_time)
            ii = int(np.ceil(opt.eval_time / frame_time))
            if i + ii >= len(seq_info_list):
                continue
            
            label_path = seq_info_list[i + ii]['label_path']
            eval_pairs.append((pred_path, label_path))
    
    print("Found {} pairs.".format(len(eval_pairs)))
    
    auroc, aupr, fpr = extract_batch(eval_pairs, num_cpus=opt.num_cpus)
    print(f"auroc: {auroc}, aupr: {aupr}, fpr: {fpr}")
