'''
TODO Experiment description
'''

import matplotlib

from experiments.MaximizeActivity.args import ARGS
from  xdream.experiment.utils.misc import run_single
from  xdream.experiment.maximize_activity import MaximizeActivityExperiment

#matplotlib.use('TKAgg')

if __name__ == '__main__': run_single(args_conf=ARGS, exp_type=MaximizeActivityExperiment)
