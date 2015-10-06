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
import sys
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
import qutip.control.errors as errors
import qutip.control.pulsegen as pulsegen

example_name = 'Hadamard-CRAB-man_cfg'
log_level = logging.INFO

# ****************************************************************
# Define the physics of the problem

nSpins = 1

# Note that for now the dynamics must be specified as ndarrays
# when using manual config
# This is until GitHub issue #370 is resolved
H_d = sigmaz().full()
H_c = [sigmax().full()]
# Number of ctrls
n_ctrls = len(H_c)

U_0 = identity(2**nSpins).full()
# Hadamard gate
#U_targ = Qobj(np.array([[1,1],[1,-1]], dtype=complex)/np.sqrt(2))
U_targ = hadamard_transform(nSpins).full()

# Evolution parameters
# time-slicing
n_ts = 100
# Total drive time
evo_time = 6.0

print("\n***********************************")
print("Creating and configuring control optimisation objects")

log_level = logging.DEBUG

# Create the OptimConfig object
cfg = optimconfig.OptimConfig()
cfg.log_level = log_level

# Create the dynamics object
dyn = dynamics.DynamicsUnitary(cfg)
dyn.num_tslots = n_ts
dyn.evo_time = evo_time

# Physical parameters
dyn.target = U_targ
dyn.initial = U_0
dyn.drift_dyn_gen = H_d
dyn.ctrl_dyn_gen = H_c

# Create the TerminationConditions instance
tc = termcond.TerminationConditions()
tc.fid_err_targ = 1e-3
tc.min_gradient_norm = 1e-10
tc.max_iter_total = 200
tc.max_wall_time_total = 30
tc.break_on_targ = True

optim = optimizer.OptimizerCrabFmin(cfg, dyn)

sts = stats.Stats()
dyn.stats = sts
optim.stats = sts
optim.config = cfg
optim.dynamics = dyn
optim.termination_conditions = tc

guess_pgen = pulsegen.create_pulse_gen('LIN', dyn)
init_amps = np.zeros([n_ts, n_ctrls])
optim.pulse_generator = []
for j in range(n_ctrls):
    # Note that for CRAB each control must have its own pulse generator
    # as the frequencies and coeffs values are stored in the pulsegen
    pgen = pulsegen.PulseGenCrabFourier(dyn)
    pgen.scaling = 0.1
    # comment out the next line for no guess pulse 
    pgen.guess_pulse = guess_pgen.gen_pulse()
    optim.pulse_generator.append(pgen)
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
print "Initial fidelity error {}".format(result.initial_fid_err)
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
