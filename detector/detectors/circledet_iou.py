from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
import os

from external.nms import soft_nms
from models.decode import cdiou_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import cdiou_post_process
from utils.debugger import Debugger

from .base_detector import BaseDetector

class CircledetIOUDetector(BaseDetector):
  def __init__(self, opt):
    super(CircledetIOUDetector, self).__init__(opt)

  def process(self, images, return_time=False):
    with torch.no_grad():
      output = self.model(images)[-1]
      hm = output['hm'].sigmoid_()
      cl = output['cl']
      reg = output['reg'] if self.opt.reg_offset else None
      occ = output['occ']
      if self.opt.flip_test:
        hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
        cl = (cl[0:1] + flip_tensor(cl[1:2])) / 2
        reg = reg[0:1] if reg is not None else None
      torch.cuda.synchronize()
      forward_time = time.time()
      dets = cdiou_decode(hm, cl, occ, self.opt,reg=reg, K=self.opt.K )
      #x, y, r, occ, scores, clses],
    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  def post_process(self, dets, meta, scale=1):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = cdiou_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], self.opt.num_classes)
    for j in range(1, self.num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 6)
      dets[0][j][:, :3] /= scale
      dets[0][j][:, 4:5] /= scale
    return dets[0]

  def merge_outputs(self, detections):
    results = {}
    for j in range(1, self.num_classes + 1):
      results[j] = np.concatenate(
        [detection[j] for detection in detections], axis=0).astype(np.float32)
      if len(self.scales) > 1 or self.opt.nms:
         soft_nms(results[j], Nt=0.5, method=2)
    scores = np.hstack(
      [results[j][:, 5] for j in range(1, self.num_classes + 1)])
    if len(scores) > self.max_per_image:
      kth = len(scores) - self.max_per_image
      thresh = np.partition(scores, kth)[kth]
      for j in range(1, self.num_classes + 1):
        keep_inds = (results[j][:, 5] >= thresh)
        results[j] = results[j][keep_inds]
    return results

  def debug(self, debugger, images, dets, output, scale=1):
    detection = dets.detach().cpu().numpy().copy()
    detection[:, :, :4] *= self.opt.down_ratio
    for i in range(1):
      img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
      img = ((img * self.std + self.mean) * 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
      debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
      for k in range(len(dets[i])):
        if detection[i, k, 4] > self.opt.center_thresh:
          debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
                                 detection[i, k, 4],
                                 img_id='out_pred_{:.1f}'.format(scale))


  def show_results(self, debugger, image, image_or_path_or_tensor, results, name="demo"):

    if self.opt.filter_boarder:
      output_h = self.opt.input_h  # hard coded
      output_w = self.opt.input_w  # hard coded
      for j in range(1, self.num_classes + 1):
        for i in range(len(results[j])):
          cp = [0, 0]
          cp[0] = results[j][i][0]
          cp[1] = results[j][i][1]
          cr = results[j][i][2]
          occ = results[j][i][3]
          if cp[0] - cr < 0 or cp[0] + cr > output_w:
            results[j][i][4] = 0
            continue
          if cp[1] - cr < 0 or cp[1] + cr > output_h:
            results[j][i][3] = 0
            continue

    debugger.add_img(image, img_id=os.path.basename(name.split('.')[0]))
    for j in range(1, self.num_classes + 1):
      for circle in results[j]:
        if circle[4] > self.opt.vis_thresh:
          debugger.add_circle_and_occlusion(circle[:3], circle[3], circle[-1],
                                   circle[4], img_id=os.path.basename(name.split('.')[0]))
    debugger.show_all_imgs(image_or_path_or_tensor, pause=self.pause)
