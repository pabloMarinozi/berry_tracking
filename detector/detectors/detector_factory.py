from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .exdet import ExdetDetector
from .ddd import DddDetector
from .ctdet import CtdetDetector
from .multi_pose import MultiPoseDetector
from .circledet import GrapesCircledetDetector
from .polygondet import PolygondetDetector
from .circledet_iou import CircledetIOUDetector

detector_factory = {
  'exdet': ExdetDetector, 
  'ddd': DddDetector,
  'ctdet': CtdetDetector,
  'multi_pose': MultiPoseDetector,
  'circledet': GrapesCircledetDetector,
  'polygondet': PolygondetDetector,
  'cdiou': CircledetIOUDetector
}
