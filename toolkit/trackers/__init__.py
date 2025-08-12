from __future__ import absolute_import

import numpy as np
import time
from PIL import Image

from ..utils.viz import show_frame


class Tracker(object):

    def __init__(self, name, is_deterministic=False):
        self.name = name
        self.is_deterministic = is_deterministic
    
    def init(self, image, box):
        raise NotImplementedError()

    def update(self, image):
        raise NotImplementedError()

    def track(self, img_files, anno, visualize=False):
        frame_num = len(img_files)
        if len(anno.shape) == 1:
            preds = np.zeros((frame_num, anno.shape[0]))
        else:
            preds = np.zeros((frame_num, anno.shape[0], anno.shape[1]))
        preds[0] = anno
        times = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            image = Image.open(img_file)
            if not image.mode == 'RGB':
                image = image.convert('RGB')

            start_time = time.time()
            if f == 0:
                self.init(image, anno)
            else:
                preds[f, :] = self.update(image)
            times[f] = time.time() - start_time

            if visualize:
                show_frame(image, preds[f, :])

        return preds, times


from .identity_tracker import IdentityTracker