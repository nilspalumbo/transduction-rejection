"""
Various models. All models extend Classifier allowing to be easily saved an loaded using common.state.
"""

import models.torch_utils as torch_utils
from .classifier import Classifier, MLP, DenseNet
from .resnet import ResNet
from .wideresnet import WideResNet
from .fixed_lenet import FixedLeNet
from .detectors import *
