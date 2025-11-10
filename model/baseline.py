# coding=UTF-8
from __future__ import division
from __future__ import absolute_import
import torch


class Baseline(object):
    """Base class for baseline estimators"""
    def get_baseline_value(self):
        """Get current baseline value"""
        pass

    def update(self, target):
        """Update baseline with new target value"""
        pass


class ReactiveBaseline(Baseline):
    """
    Reactive baseline that maintains a moving average
    
    Args:
        l: Learning rate for exponential moving average
    """
    def __init__(self, l):
        self.l = l
        self.b = 0.0  # PyTorch中用Python float即可,不需要Variable

    def get_baseline_value(self):
        """Return current baseline value"""
        return self.b

    def update(self, target):
        """
        Update baseline using exponential moving average
        
        Args:
            target: New target value (can be float or torch.Tensor)
        """
        if isinstance(target, torch.Tensor):
            target = target.item()  # Convert tensor to Python float
        
        self.b = (1 - self.l) * self.b + self.l * target
