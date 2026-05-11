import os
import torch
from ultralytics.utils.tal import TaskAlignedAssigner

# deterministic small tensors
torch.manual_seed(0)
bs = 1
na = 8
nmax = 2
num_classes = 3

# Create synthetic scores and boxes where overlaps are non-zero
pd_scores = torch.zeros(bs, na, num_classes)
pd_scores[:, :, 0] = 0.9  # high score for class 0 on all anchors

# define two gt boxes (xyxy) and replicate anchors near them
gt_labels = torch.tensor([[[0], [1]]], dtype=torch.long)
gt_bboxes = torch.tensor([[[0.1, 0.1, 0.4, 0.4], [0.5, 0.5, 0.9, 0.9]]], dtype=torch.float32)

# make predicted boxes equal to first gt box for all anchors (so IoU > 0)
pd_bboxes = gt_bboxes[:, 0:1, :].expand(bs, na, 4).clone()

# anchor centers are unused for this direct metrics call but set to zeros
anc_points = torch.zeros(na, 2)

# mask: all anchors considered
mask_gt = torch.ones(bs, nmax, na, dtype=torch.bool)

assigner = TaskAlignedAssigner(topk=3, num_classes=num_classes, alpha=1.0, beta=2.0)
# set attributes expected by get_box_metrics
assigner.bs = pd_scores.shape[0]
assigner.n_max_boxes = gt_bboxes.shape[1]

# baseline
os.environ.pop('CRAZING_BOOST', None)
os.environ.pop('CRAZING_CLASS_INDEX', None)
align_base, overlaps = assigner.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt)

# boosted
os.environ['CRAZING_BOOST'] = '2.5'
os.environ['CRAZING_CLASS_INDEX'] = '0'
align_boosted, overlaps2 = assigner.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt)

changed = (align_boosted != align_base).any().item()
print('focused-TAL boost changed align metric:', changed)
print('align_base sample:', align_base[0, :, :min(8, align_base.shape[-1])])
print('align_boosted sample:', align_boosted[0, :, :min(8, align_boosted.shape[-1])])
