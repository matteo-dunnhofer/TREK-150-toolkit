from toolkit.datasets import VISTA

dset = VISTA('./VISTA', split='test', mode='lt', download=True)

# [optional] create a DAVIS-like dataset separating FPV and TPV sequences
# dset.export_as_davis()