from turtle import forward
import numpy as np
import torch

from ..builder import RECOGNIZERS
from .recognizergcn import RecognizerGCN

@RECOGNIZERS.register_module()
class RecognizerGCNKinematic(RecognizerGCN):
    def forward_train(self, keypoint, angle, label, **kwargs):
        return super().forward_train(angle, label, **kwargs)

    def forward_test(self, keypoint, angle, **kwargs):
        return super().forward_test(angle **kwargs)
    
    def forward(self, keypoint, angle, label=None, return_loss=True, **kwargs):
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            return self.forward_train(keypoint, angle, label, **kwargs)

        return self.forward_test(keypoint, angle, **kwargs)

    # def extract_feat(self, keypoint, angle):
    #     return super().extract_feat(angle)