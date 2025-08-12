from siamfc_pytorch.siamfc.siamfc import TrackerSiamFC
from toolkit.experiments import ExperimentVISTA

tracker = TrackerSiamFC()

root_dir = './VISTA'
exp = ExperimentVISTA(root_dir, split='test', mode='lt', anno_type='mask', result_dir='./', report_dir='./')

# Run an experiment with synchronized one pass evaluation and save results
exp.run_sope(tracker, visualize=False)

# Generate a report for the tracker
exp.report([tracker.name])