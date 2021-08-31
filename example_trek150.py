from siamfc_pytorch.siamfc.siamfc import TrackerSiamFC
from toolkit.experiments import ExperimentTREK150

tracker = TrackerSiamFC()

root_dir = './'
exp = ExperimentTREK150(root_dir, result_dir='./', report_dir='./')
prot = 'ope'

# Run an experiment with the protocol of interest and save results
exp.run(tracker, protocol=prot, visualize=False)

# Generate a report for the protocol of interest
exp.report([tracker.name], protocol=prot)