from siamfc_pytorch.siamfc.siamfc import TrackerSiamFC
from toolkit.experiments import ExperimentOTB

tracker = TrackerSiamFC()

root_dir = './'
exp = ExperimentOTB(root_dir, version=2015, result_dir='./', report_dir='./')

# Run an experiment with the protocol of interest and save results
exp.run(tracker)

# Generate a report for the protocol of interest
exp.report([tracker])