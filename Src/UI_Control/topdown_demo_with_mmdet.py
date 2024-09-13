# Copyright (c) OpenMMLab. All rights reserved.
import logging
import mimetypes
import os
import time
from argparse import ArgumentParser
import sys
import json_tricks as json
import numpy as np
from mmengine.logging import print_log
from mmcv.transforms import Compose
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "..", "tracker"))
from utils.timer import FPS_Timer

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def process_one_image(model,
                      img,
                      select_id = None):
    fps_timer = FPS_Timer()
    detect_args = model["Detector"]["args"]
    detector = model["Detector"]["detector"]
    test_pipeline = model["Detector"]["test_pipeline"]
    pose_args = model["Pose Estimator"]["args"]
    pose_estimator = model["Pose Estimator"]["pose estimator"]
    tracker = model["Tracker"]["tracker"]
    result = inference_detector(detector, img,test_pipeline=test_pipeline)
    det_result = result.pred_instances[
        result.pred_instances.scores > detect_args.score_thr].cpu().numpy()
    pred_instance = det_result
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[pred_instance.labels == detect_args.det_cat_id]
    
    bboxes = bboxes[nms(bboxes, detect_args.nms_thr), :4]
 # Update tracker with new detections
    new_bboxes = np.hstack((bboxes, np.zeros((bboxes.shape[0], 2))))
    new_bboxes[:, -2] = 0.9
    new_bboxes[:, -1] = 0
    online_targets = tracker.update(new_bboxes, img.copy())
    
    online_bbox = [t.tlwh for t in online_targets if (t.tlwh[2] * t.tlwh[3] > 10) and (select_id is None or t.track_id == select_id)]
    online_ids = [t.track_id for t in online_targets if (t.tlwh[2] * t.tlwh[3] > 10) and (select_id is None or t.track_id == select_id)]
    
    new_online_box = [[x1, y1, x1 + w, y1 + h] for x1, y1, w, h in online_bbox]
    bboxes = np.array(new_online_box)
    
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)
    
    return data_samples.get('pred_instances', None), online_ids