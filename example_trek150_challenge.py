from siamfc_pytorch.siamfc.siamfc import TrackerSiamFC
from toolkit.experiments import ExperimentTREK150

tracker = TrackerSiamFC()

root_dir = './'
exp = ExperimentTREK150(root_dir, result_dir='./', report_dir='./')

# Run an experiment with the protocols of interest and save results
exp.run(tracker, protocol='ope', visualize=False)
exp.run(tracker, protocol='mse', visualize=False)
exp.run(tracker, protocol='hoi', visualize=False)

# Export results for the CodaLab challenge
exp.export_results_for_challenge(tracker.name)