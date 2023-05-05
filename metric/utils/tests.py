import os
import argparse
import numpy as np
from PIL import Image

from measures import get_measures

train_id_in = 0
train_id_out = 1
ignore_id = 255

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_npy", type=str)
    parser.add_argument("--gt_anno", type=str)
    opt = parser.parse_args()
    
    # Load pred npy
    assert os.path.exists(opt.pred_npy)
    pred = np.load(opt.pred_npy)
    
    # Load gt anno
    assert os.path.exists(opt.gt_anno)
    gt = np.asarray(Image.open(opt.gt_anno))
    
    height, width = gt.shape
    
    if pred.ndim == 3:
        assert pred.shape[0] == 1
        pred = pred[0]
    assert pred.shape == gt.shape
    
    converted_gt = np.zeros_like(gt)
    converted_gt += train_id_in
    converted_gt[gt == 254] = train_id_out
    converted_gt[gt == ignore_id] = ignore_id  # ignored label
    
    auroc, aupr, fpr = get_measures(pred[gt == train_id_out], pred[gt == train_id_in])

    print(auroc, aupr, fpr)
