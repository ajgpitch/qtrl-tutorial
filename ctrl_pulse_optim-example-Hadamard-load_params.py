# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 12:58:53 2014
@author: Alexander Pitchford
@email1: agp1@aber.ac.uk
@email2: alex.pitchford@gmail.com
@organization: Aberystwyth University
@supervisor: Daniel Burgarth

The code in this file was is intended for use in not-for-profit research,
teaching, and learning. Any other applications may require additional
licensing

Example to demonstrate configuration through manual creation of the 
the optimiser and its child objects. Note that this is not necessary for
using the CRAB algorithm, it's just one way of configuring.
See the main Hadamard example for how to call the CRAB alg using the 
pulseoptim functions. 

The system in this example is a single qubit in a constant field in z
with a variable control field in x
The target evolution is the Hadamard gate irrespective of global phase

The user can experiment with the timeslicing, by means of changing the
number of timeslots and/or total time for the evolution.
Different initial (starting) pulse types can be tried.
The initial and final pulses are displayed in a plot
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime

from qutip import Qobj, identity, sigmax, sigmaz
from qutip.qip import hadamard_transform
import qutip.logging as logging
logger = logging.get_logger()
#QuTiP control modules
import qutip.control.optimconfig as optimconfig
import qutip.control.dynamics as dynamics
import qutip.control.termcond as termcond
import qutip.control.optimizer as optimizer
import qutip.control.stats as stats
import qutip.control.pulsegen as pulsegen
import qutip.control.errors as errors
import qutip.control.loadparams as loadparams

example_name = 'Hadamard-load_params'
log_level = logging.INFO

# ****************************************************************
# Define the physics of the problem

nSpins = 1

# Note that for now the dynamics must be specified as ndarrays
# when using manual config
# This is until GitHub issue #370 is resolved
H_d = sigmaz()
H_c = sigmax()

U_0 = identity(2**nSpins)
U_targ = hadamard_transform(nSpins)

# Evolution parameters
# time-slicing
n_ts = 100
# Total drive time
evo_time = 6.0

print("\n***********************************")
print("Creating and configuring control optimisation objects")

example_name = 'Hadamard-load_params'
log_level = logging.DEBUG

# Create the OptimConfig object
cfg = optimconfig.OptimConfig()
cfg.param_fname = "Hadamard_params.ini"
cfg.param_fpath = os.path.join(os.getcwd(), cfg.param_fname)
cfg.log_level = log_level
cfg.pulse_type = "ZERO"

# load the config parameters
# note these will overide those above if present in the file
print("Loading config parameters from {}".format(cfg.param_fpath))
loadparams.load_parameters(cfg.param_fpath, config=cfg)
# Update the log level, as this may have been changed in the config
logger.setLevel(cfg.log_level)

# Create the dynamics object
dyn = dynamics.DynamicsUnitary(cfg)
# Physical parameters
dyn.target = U_targ.full()
dyn.initial = U_0.full()
dyn.drift_dyn_gen = H_d.full()
dyn.ctrl_dyn_gen = list([H_c.full()])
# load the dynamics parameters
# note these will overide those above if present in the file
print("Loading dynamics parameters from {}".format(cfg.param_fpath))
loadparams.load_parameters(cfg.param_fpath, dynamics=dyn)
dyn.init_timeslots()      
n_ts = dyn.num_tslots
n_ctrls = dyn.get_num_ctrls()

# Create a pulse generator of the type specified
pgen = pulsegen.create_pulse_gen(pulse_type=cfg.pulse_type, dyn=dyn)
# load the pulse generator parameters
# note these will overide those above if present in the file
print("Loading pulsegen parameters from {}".format(cfg.param_fpath))
loadparams.load_parameters(cfg.param_fpath, pulsegen=pgen)

# Create the TerminationConditions instance
tc = termcond.TerminationConditions()
# load the termination condition parameters
# note these will overide those above if present in the file
print("Loading termination condition parameters from {}".format(
        cfg.param_fpath))
loadparams.load_parameters(cfg.param_fpath, term_conds=tc)

# Create the optimiser object
if cfg.optim_method == 'BFGS':
    optim = optimizer.OptimizerBFGS(cfg, dyn)
elif cfg.optim_method == 'FMIN_L_BFGS_B':
    optim = optimizer.OptimizerLBFGSB(cfg, dyn)
elif cfg.optim_method is None:
    raise errors.UsageError("Optimisation algorithm must be specified "
                            "via 'optim_method' parameter")
else:
    optim = optimizer.Optimizer(cfg, dyn)
    optim.method = cfg.optim_method
# load the optimiser parameters
# note these will overide those above if present in the file
print("Loading optimiser parameters from {}".format(cfg.param_fpath))
loadparams.load_parameters(cfg.param_fpath, optim=optim)

sts = stats.Stats()
dyn.stats = sts
optim.stats = sts
optim.config = cfg
optim.dynamics = dyn
optim.pulse_generator = pgen
optim.termination_conditions = tc

init_amps = np.zeros([n_ts, n_ctrls])
# Initialise the dynamics
init_amps = np.zeros([n_ts, n_ctrls])
for j in range(n_ctrls):
    init_amps[:, j] = pgen.gen_pulse()
dyn.initialize_controls(init_amps)

# *************************************************************
# File extension for output files
f_ext = "{}_n_ts{}.txt".format(example_name, n_ts)

# Run the optimisation
print("\n***********************************")
print("Starting pulse optimisation")
result = optim.run_optimization()

print("\n***********************************")
print("Optimising complete. Stats follow:")
result.stats.report()
print("\nFinal evolution\n{}\n".format(result.evo_full_final))

print("********* Summary *****************")
print("Initial fidelity error {}".format(result.initial_fid_err))
print("Final fidelity error {}".format(result.fid_err))
print("Final gradient normal {}".format(result.grad_norm_final))
print("Terminated due to {}".format(result.termination_reason))
print("Number of iterations {}".format(result.num_iter))
#print("wall time: ", result.wall_time
print("Completed in {} HH:MM:SS.US".\
        format(datetime.timedelta(seconds=result.wall_time)))
print("***********************************")
# Plot the initial and final amplitudes
fig1 = plt.figure()
ax1 = fig1.add_subplot(2, 1, 1)
ax1.set_title("Initial control amps")
ax1.set_xlabel("Time")
ax1.set_ylabel("Control amplitude")
t = result.time[:n_ts]
for j in range(n_ctrls):
    amps = result.initial_amps[:, j]
    ax1.plot(t, amps)
ax2 = fig1.add_subplot(2, 1, 2)
ax2.set_title("Optimised Control Sequences")
ax2.set_xlabel("Time")
ax2.set_ylabel("Control amplitude")
for j in range(n_ctrls):
    amps = result.final_amps[:, j]
    ax2.plot(t, amps)

plt.show()
