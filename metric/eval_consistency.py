# Since we are delivering a video anomaly dataset
# It is necessary to evaluate the consistency of the anomaly scores across frames

import os
import json
import numpy as np
import argparse
from PIL import Image
from utils.measures import stable_cumsum
from utils.consistency import project

from p_tqdm import p_map

train_id_in = 0
train_id_out = 1
anomaly_id = 254
ignore_id = 255
inv_score = False
WIDTH, HEIGHT = 1920, 1080
TPR_THRESHOLD = 0.95  # Consistency@TPR=95%


def extract_pred(pred_path, label_path):
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
    
    pos = pred[converted_gt == train_id_out].reshape(-1, 1)
    neg = pred[converted_gt == train_id_in].reshape(-1, 1)
    y_score = np.squeeze(np.vstack((pos, neg)))
    
    y_true = np.zeros(len(y_score), dtype=np.int32)
    y_true[:len(pos)] += 1
    y_true = y_true.view(bool)
    
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    
    tps = stable_cumsum(y_true)
    that_score = y_score[tps.searchsorted(int(tps[-1] * TPR_THRESHOLD))]
    
    prediction = np.zeros_like(pred)
    prediction[pred > that_score] = 1  # anomaly
    prediction[pred <= that_score] = 0  # normal
    
    return prediction[None, ...]


def extract_single(eval_pair):
    
    pred1_path, label1_path, depth1_path, extrinsics1, pred2_path, label2_path, depth2_path, extrinsics2 = eval_pair
    
    # Get Projected attribute (namely AnomalyPrediction@TPR95)
    pred1 = extract_pred(pred1_path, label1_path)
    pred2 = extract_pred(pred2_path, label2_path)
    
    proj_1_to_2 = project(extrinsics1, depth1_path, pred1, extrinsics2, depth2_path)
    reconstructed_depth, reconstructed_attribute, mask = proj_1_to_2
    
    # Calculate prediction consistency between reconstructed_attribute and pred2, using mask
    
    match_cnt = np.sum((reconstructed_attribute == pred2)[0] & mask)
    all_cnt = np.sum(mask)
    return match_cnt / all_cnt
    

def extract_batch(eval_pairs, num_cpus=16):
    output_list = p_map(extract_single, eval_pairs, num_cpus=num_cpus)
    return np.mean(output_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_helper", type=str, required=True, help="Path to eval_helper.json")
    parser.add_argument("--pred_list", type=str, required=True, help="A path to file. Each line is a path to a prediction file, with order corresponding to that of val.csv of dataset.")
    parser.add_argument("--frame_interval", type=int, default=60, help="The interval between frames to be compared. Default: 60")
    parser.add_argument("--inv_anomaly_score", action='store_true', help="Normally, predicted logits for anomaly objects should be larger than those in-distribution pixels. If this flag is set, we will take the negative value of the logits for evaluation.")
    parser.add_argument("--num_cpus", type=int, default=16)
    opt = parser.parse_args()
    
    if opt.inv_anomaly_score:
        inv_score = True
    
    # construct eval pairs using eval_helper
    eval_pairs = []  # List of (pred1, label1, pred2, label2)
    
    with open(opt.eval_helper, 'r') as f:
        eval_helper = json.load(f)
    
    with open(opt.pred_list, 'r') as f:
        pred_list = f.read().strip().split('\n')
    
    for seq_name, seq_info_list in eval_helper.items():
        for i in range(len(seq_info_list)):
            seq1 = seq_info_list[i]
            if i + opt.frame_interval >= len(seq_info_list):
                continue
            seq2 = seq_info_list[i + opt.frame_interval]
    
            pred1_path = pred_list[seq1['global_idx']]
            pred2_path = pred_list[seq2['global_idx']]
            label1_path = seq1['label_path']
            depth1_path = seq1['depth_path']
            extrinsics1 = seq1['extrinsics']
            label2_path = seq2['label_path']
            depth2_path = seq2['depth_path']
            extrinsics2 = seq2['extrinsics']
            
            assert os.path.exists(pred1_path)
            assert os.path.exists(pred2_path)
            assert os.path.exists(label1_path)
            assert os.path.exists(label2_path)
            assert os.path.exists(depth1_path)
            assert os.path.exists(depth2_path)

            eval_pairs.append((pred1_path, label1_path, depth1_path, extrinsics1, pred2_path, label2_path, depth2_path, extrinsics2))
    
    consistency_metric = extract_batch(eval_pairs, num_cpus=opt.num_cpus)
    print("Consistency@TPR=95%: {:.4f}".format(consistency_metric))
