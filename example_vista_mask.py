from toolkit.trackers import Tracker
from toolkit.experiments import ExperimentVISTA
import numpy as np
import sys
sys.path.append('./DAM4SAM')
from DAM4SAM.dam4sam_tracker import DAM4SAMTracker

# wrapping the original DAM4SAm implementation
# into the got10k Tracker class
class TrackerDAM4SAM(Tracker):

    def __init__(self, **kwargs):
        super(TrackerDAM4SAM, self).__init__('DAM4SAM', True)
        self.dam4sam = DAM4SAMTracker()

    def init(self, img, mask):
        self.dam4sam.initialize(img, mask)

    def update(self, img):
        out_dict = self.dam4sam.track(img)
        return out_dict['pred_mask']


tracker = TrackerDAM4SAM()

root_dir = './VISTA'
exp = ExperimentVISTA(root_dir, split='test', mode='lt', anno_type='mask', result_dir='./', report_dir='./')

# Run an experiment with synchronized one pass evaluation and save results
exp.run_sope(tracker, visualize=False)

# Generate a report for the tracker
exp.report([tracker.name])