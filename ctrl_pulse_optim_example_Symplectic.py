# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 15:14:58 2014
Updated 2015-06-27

@author: Alexander Pitchford
@email1: agp1@aber.ac.uk
@email2: alex.pitchford@gmail.com
@organization: Aberystwyth University
@supervisor: Daniel Burgarth

Example to demonstrate using the control library to determine control
pulses using the ctrlpulseoptim.optimize_pulse function.
The (default) L-BFGS-B algorithm is used to optimise the pulse to
minimise the fidelity error, which in this case is given by the
'Trace difference' norm.

This in an Symplectic quantum system example, with two coupled oscillators

The user can experiment with the timeslicing, by means of changing the
number of timeslots and/or total time for the evolution.
Different initial (starting) pulse types can be tried.
The initial and final pulses are displayed in a plot
"""

import os
import numpy as np
import numpy.matlib as mat
#import scipy.linalg as la
import matplotlib.pyplot as plt
import datetime

#QuTiP
from qutip import Qobj, identity
import qutip.logging as logging
logger = logging.get_logger()
#QuTiP control modules
import qutip.control.symplectic as sympl
import qutip.control.optimconfig as optimconfig
import qutip.control.dynamics as dynamics
import qutip.control.termcond as termcond
import qutip.control.optimizer as optimizer
import qutip.control.stats as stats
import qutip.control.errors as errors
import qutip.control.pulsegen as pulsegen
import qutip.control.loadparams as loadparams

example_name = 'Coupled_osc'
log_level = logging.DEBUG

# Create the OptimConfig object
cfg = optimconfig.OptimConfig()
cfg.param_fname = "coup_osc_param.ini"
cfg.param_fpath = os.path.join(os.getcwd(), cfg.param_fname)
cfg.log_level = log_level
# Initial pulse type
# pulse type alternatives: 
# RNDWAVES|RNDFOURIER|RNDWALK1|RNDWALK2|RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE|
cfg.pulse_type = 'RNDWAVES'
cfg.amp_lbound = -5.0
cfg.amp_ubound = 5.0
    
# load the config parameters
# note these will overide those above if present in the file
print("Loading config parameters from {}".format(cfg.param_fpath))
loadparams.load_parameters(cfg.param_fpath, config=cfg)
# Update the log level, as this may have been changed in the config
logger.setLevel(cfg.log_level)

# Create the dynamics object
dyn = dynamics.DynamicsSymplectic(cfg)
dyn.num_tslots = 200
dyn.evo_time = 10.0

# Physical parameters
dyn.coupling1 = 0.3
dyn.coupling2 = 0.2
dyn.sqz = 0.5
dyn.rot = 1.0

# load the dynamics parameters
# note these will overide those above if present in the file
print("Loading dynamics parameters from {}".format(cfg.param_fpath))
loadparams.load_parameters(cfg.param_fpath, dynamics=dyn)

dyn.init_timeslots()      
n_ts = dyn.num_tslots

# Create a pulse generator of the type specified
p_gen = pulsegen.create_pulse_gen(pulse_type=cfg.pulse_type, dyn=dyn)
p_gen.lbound = cfg.amp_lbound
p_gen.ubound = cfg.amp_ubound

# load the pulse generator parameters
# note these will overide those above if present in the file
print("Loading pulsegen parameters from {}".format(cfg.param_fpath))
loadparams.load_parameters(cfg.param_fpath, pulsegen=p_gen)

# Create the TerminationConditions instance
tc = termcond.TerminationConditions()
tc.fid_err_targ = 1e-3
tc.min_gradient_norm = 1e-10
tc.max_iter_total = 200
tc.max_wall_time_total = 30
tc.break_on_targ = True

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

sts = stats.Stats()
dyn.stats = sts
optim.stats = sts
optim.config = cfg
optim.dynamics = dyn
optim.termination_conditions = tc
optim.pulse_generator = p_gen
    
# load the optimiser parameters
# note these will overide those above if present in the file
print("Loading optimiser parameters from {}".format(cfg.param_fpath))
loadparams.load_parameters(cfg.param_fpath, optim=optim)
    
# ****************************************************************
# Define the physics of the problem

#Drift
g1 = 2*(dyn.coupling1 + dyn.coupling2)
g2 = 2*(dyn.coupling1 - dyn.coupling2)
#g1 = 1.0
#g2 = 0.2
A0 = np.array([[1, 0, g1, 0], 
                   [0, 1, 0, g2], 
                   [g1, 0, 1, 0], 
                   [0, g2, 0, 1]])
dyn.drift_dyn_gen = A0

#Rotate control
A_rot = dyn.rot*np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]
                    ])


#Squeeze Control
A_sqz = dyn.sqz*np.array([
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]
                    ])
dyn.ctrl_dyn_gen = [A_rot, A_sqz]  
n_ctrls = dyn.get_num_ctrls()

dyn.initial = identity(4).full()

# Target
A_targ = Qobj(np.array([
                [0, 0, 1, 0], 
                [0, 0, 0, 1], 
                [1, 0, 0, 0], 
                [0, 1, 0, 0]
                ]))
          
Omg = Qobj(sympl.calc_omega(2))
print("Omega:\n{}\n".format(Omg))

S_targ = (-A_targ*Omg*np.pi/2.0).expm()
dyn.target = S_targ.full()
#S_targ = (Omg*A_targ*np.pi/2.0).expm()
print("Target S:\n{}\n".format(S_targ))


# Initialise the dynamics
init_amps = np.zeros([n_ts, n_ctrls])
for j in range(n_ctrls):
    init_amps[:, j] = p_gen.gen_pulse()
dyn.initialize_controls(init_amps)


f_ext = "{}_n_ts{}_ptype{}.txt".format(example_name, n_ts, cfg.pulse_type)

print("***********************************")
print("Starting pulse optimisation")
result = optim.run_optimization()

# Save final amplitudes to a text file
pulsefile = "ctrl_amps_final_" + f_ext
dyn.save_amps(pulsefile)
if (log_level <= logging.INFO):
    print("Final amplitudes output to file: " + pulsefile)
        
print("\n***********************************")
print("Optimising complete. Stats follow:")
result.stats.report()
print("Final evolution\n{}\n".format(result.evo_full_final))

print("********* Summary *****************")
print("Initial fidelity error {}".format(result.initial_fid_err))
print("Final fidelity error {}".format(result.fid_err))
print("Terminated due to {}".format(result.termination_reason))
print("Number of iterations {}".format(result.num_iter))
#print "wall time: ", result.wall_time
print("Completed in {} HH:MM:SS.US".format(
        datetime.timedelta(seconds=result.wall_time)))
# print "Final gradient normal {}".format(result.grad_norm_final)
print("***********************************")

# Plot the initial and final amplitudes
fig1 = plt.figure()
ax1 = fig1.add_subplot(2, 1, 1)
ax1.set_title("Initial ctrl amps")
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

