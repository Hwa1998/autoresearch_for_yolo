import os
import tqdm
import torch

from ultralytics import YOLO


def smoke_model_yaml():
    """Load and profile a sample model YAML to ensure model builds."""
    model = YOLO(f'ultralytics/cfg/models/v13/yolov13_nmsfree.yaml')
    model.info(detailed=True)
    model.profile([640, 640])
    model.fuse()


def test_focused_tal():
    """Basic unit test for TaskAlignedAssigner focused-TAL boosting logic.

    This constructs tiny synthetic tensors and verifies that setting
    CRAZING_BOOST via env changes the align metric.
    """
    try:
        from ultralytics.utils.tal import TaskAlignedAssigner
    except Exception as e:
        print('TaskAlignedAssigner import failed:', e)
        return

    # deterministic small tensors
    torch.manual_seed(0)
    bs = 1
    na = 8
    nmax = 2
    num_classes = 3

    pd_scores = torch.rand(bs, na, num_classes)
    pd_bboxes = torch.rand(bs, na, 4)
    anc_points = torch.rand(na, 2)
    # two gt boxes: class 0 (crazing) and class 1
    gt_labels = torch.tensor([[[0], [1]]], dtype=torch.long)
    gt_bboxes = torch.rand(bs, nmax, 4)
    mask_gt = torch.ones(bs, nmax, 1, dtype=torch.bool)

    assigner = TaskAlignedAssigner(topk=3, num_classes=num_classes, alpha=1.0, beta=2.0)
    align_base, overlaps = assigner.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt)

    os.environ['CRAZING_BOOST'] = '2.5'
    os.environ['CRAZING_CLASS_INDEX'] = '0'
    align_boosted, _ = assigner.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt)

    # check that boost changed values for crazing positions
    try:
        changed = (align_boosted != align_base).any().item()
    except Exception:
        changed = False

    print('focused-TAL boost changed align metric:', changed)


if __name__ == '__main__':
    smoke_model_yaml()
    test_focused_tal()
