# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 17:18:29 2014

@author: Alexander Pitchford
@email1: agp1@aber.ac.uk
@email2: alex.pitchford@gmail.com
@organization: Aberystwyth University
@supervisor: Daniel Burgarth

The code in this file was is intended for use in not-for-profit research,
teaching, and learning. Any other applications may require additional
licensing

Example to demonstrate using the control library to determine control
pulses using the ctrlpulseoptim.create_pulse_optimizer function to
generate an Optimizer object, through which the configuration can be
manipulated before running the optmisation algorithm. In this case it is
demonstrated by setting the initial ctrl pulses.from a file

The (default) L-BFGS-B algorithm is used to optimise the pulse to
minimise the fidelity error, which is equivalent maximising the fidelity
to optimal value of 1.

The system in this example is two qubits in constant fields in x, y and z
with a variable independant controls fields in x and y acting on each qubit
The target evolution is the QFT gate. The user can experiment with the
different:
    phase options - phase_option = SU or PSU
    propagtor computer type prop_type = DIAG or FRECHET
    fidelity measures - fid_type = UNIT or TRACEDIFF

The user can experiment with the timeslicing, by means of changing the
number of timeslots and/or total time for the evolution.
Different initial (starting) pulse types can be tried.
The initial and final pulses are displayed in a plot

Note the physics of this example was taken from a demo in:
DYNAMO - Dynamic Framework for Quantum Optimal Control
See Machnes et.al., arXiv.1011.4874
"""
import numpy as np
import numpy.matlib as mat
from numpy.matlib import kron
import matplotlib.pyplot as plt
import datetime
from scipy.optimize import check_grad
from numpy.testing import (
    assert_, assert_almost_equal, run_module_suite, assert_equal)

#QuTiP
from qutip import Qobj, identity, sigmax, sigmay, sigmaz, tensor
import qutip.logging_utils as logging
logger = logging.get_logger()
#QuTiP control modules
import qutip.control.pulseoptim as cpo
import qutip.control.pulsegen as pulsegen
from qutip.qip.algorithms import qft

example_name = 'QFT'
log_level=logging.INFO
# ****************************************************************
# Define the physics of the problem
Sx = sigmax()
Sy = sigmay()
Sz = sigmaz()
Si = 0.5*identity(2)

# Drift Hamiltonian
H_d = 0.5*(tensor(Sx, Sx) + tensor(Sy, Sy) + tensor(Sz, Sz))
print("drift {}".format(H_d))
# The (four) control Hamiltonians
H_c = [tensor(Sx, Si), tensor(Sy, Si), tensor(Si, Sx), tensor(Si, Sy)]
j = 0
for c in H_c:
    j += 1
    print("ctrl {} \n{}".format(j, c))

n_ctrls = len(H_c)
# start point for the gate evolution
U_0 = tensor(identity(2), identity(2))
print("U_0 {}".format(U_0))
# Target for the gate evolution - Quantum Fourier Transform gate
U_targ = (qft.qft(2)).tidyup()
#U_targ.dims = U_0.dims
print("target {}".format(U_targ))

# ***** Define time evolution parameters *****
# Number of time slots
n_ts = 100
# Time allowed for the evolution
evo_time = 10

# ***** Define the termination conditions *****
# Fidelity error target
fid_err_targ = 1e-3
# Maximum iterations for the optisation algorithm
max_iter = 200
# Maximum (elapsed) time allowed in seconds
max_wall_time = 120
# Minimum gradient (sum of gradients squared)
# as this tends to 0 -> local minima has been found
min_grad = 1e-20

check_gradient = False

# Initial pulse type
# pulse type alternatives: RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE|
p_type = 'LIN'
# *************************************************************
# File extension for output files

f_ext = "{}_n_ts{}_ptype{}.txt".format(example_name, n_ts, p_type)

print("\n***********************************")
print("Creating optimiser objects")
optim = cpo.create_pulse_optimizer(H_d, list(H_c), U_0, U_targ, n_ts, evo_time,
#                amp_lbound=-10.0, amp_ubound=10.0,
                fid_err_targ=fid_err_targ, min_grad=min_grad,
                max_iter=max_iter, max_wall_time=max_wall_time,
#                optim_method='LBFGSB',
#                method_params={'max_metric_corr':40, 'accuracy_factor':1e7,
#                                'ftol':1e-7},
#                optim_method='fmin_l_bfgs_b',
#                optim_method='l-bfgs-b',
                dyn_type='UNIT',
#                prop_type='DIAG',
                fid_type='UNIT',
#                fid_params={'phase_option':'SU'},
                init_pulse_type=p_type, pulse_scaling=1.0,
                log_level=log_level, gen_stats=True)

print("\n***********************************")
print("Configuring optimiser objects")
# **** Set some optimiser config parameters ****
optim.test_out_files = 0
dyn = optim.dynamics

print("Phase option: {}".format(dyn.fid_computer.phase_option))

# check method params
#print("max_metric_corr: {}".format(optim.max_metric_corr))
#print("accuracy_factor: {}".format(optim.accuracy_factor))
#print("phase_option: {}".format(dyn.fid_computer.phase_option))


# Generate different pulses for each control
p_gen = optim.pulse_generator
init_amps = np.zeros([n_ts, n_ctrls])
if (p_gen.periodic):
    phase_diff = np.pi / n_ctrls
    for j in range(n_ctrls):
        init_amps[:, j] = p_gen.gen_pulse(start_phase=phase_diff*j)
elif (isinstance(p_gen, pulsegen.PulseGenLinear)):
    for j in range(n_ctrls):
        p_gen.scaling = float(j) - float(n_ctrls - 1)/2
        init_amps[:, j] = p_gen.gen_pulse()
elif (isinstance(p_gen, pulsegen.PulseGenZero)):
    for j in range(n_ctrls):
        #p_gen.offset = -0.5
        p_gen.offset = sf = float(j) - float(n_ctrls - 1)/2
        init_amps[:, j] = p_gen.gen_pulse()
else:
    # Should be random pulse
    for j in range(n_ctrls):
        p_gen.init_pulse()
        init_amps[:, j] = p_gen.gen_pulse()

dyn.initialize_controls(init_amps)

print("Initial fid_err: ".format(dyn.fid_computer.get_fid_err()))
print("dim norm {}".format(dyn.fid_computer.dimensional_norm))

full_evo = dyn.full_evo
print("Check evo unitary:\n{}".format(full_evo.dag()*full_evo))

overlap = U_targ.dag()*full_evo
print("overlap:\n{}".format(overlap))

print("man fid check:\n{}".format(overlap.tr()))

print("***********************************")
print("Starting first pulse optimisation")
result1 = optim.run_optimization()

# Save final amplitudes to a text file
pulsefile = "ctrl_amps_inter_" + f_ext
dyn.save_amps(pulsefile)
if (log_level <= logging.INFO):
    print("Final amplitudes output to file: " + pulsefile)

print("\n***********************************")
print("Optimising complete. Stats follow:")
result1.stats.report()
print("Final evolution\n{}\n".format(result1.evo_full_final))

print("********* Summary *****************")
print("Initial fidelity error {}".format(result1.initial_fid_err))
print("Intermediate fidelity error {}".format(result1.fid_err))
print("Terminated due to {}".format(result1.termination_reason))
print("Number of iterations {}".format(result1.num_iter))
#print "wall time: ", result1.wall_time
print("Completed in {} HH:MM:SS.US".format(
        datetime.timedelta(seconds=result1.wall_time)))
# print "Final gradient normal {}".format(result1.grad_norm_final)
print("***********************************")

#optim.termination_conditions.fid_err_targ = 1e-10
optim.amp_lbound = -1
optim.amp_ubound = 1
print("***********************************")
print("Starting second optimisation")
result2 = optim.run_optimization()

# Save final amplitudes to a text file
pulsefile = "ctrl_amps_final_" + f_ext
dyn.save_amps(pulsefile)
if (log_level <= logging.INFO):
    print("Final amplitudes output to file: " + pulsefile)

print("\n***********************************")
print("Optimising complete. Stats follow:")
result2.stats.report()
print("Final evolution\n{}\n".format(result2.evo_full_final))

print("********* Summary *****************")
print("Starting fidelity error {}".format(result2.initial_fid_err))
print("Final fidelity error {}".format(result2.fid_err))
print("Terminated due to {}".format(result2.termination_reason))
print("Number of iterations {}".format(result2.num_iter))
#print "wall time: ", result2.wall_time
print("Completed in {} HH:MM:SS.US".format(
        datetime.timedelta(seconds=result2.wall_time)))
# print "Final gradient normal {}".format(result2.grad_norm_final)
print("***********************************")

# Plot the initial and final amplitudes
fig1 = plt.figure()
ax1 = fig1.add_subplot(4, 1, 1)
ax1.set_title("Initial control amps")
ax1.set_xlabel("Time")
ax1.set_ylabel("Control amplitude")
for j in range(n_ctrls):
    ax1.step(result1.time,
             np.hstack((result1.initial_amps[:, j], result1.initial_amps[-1, j])),
             where='post')

ax2 = fig1.add_subplot(4, 1, 2)
ax2.set_title("Intermediate Control Sequences")
ax2.set_xlabel("Time")
ax2.set_ylabel("Control amplitude")
for j in range(n_ctrls):
    ax2.step(result1.time,
             np.hstack((result1.final_amps[:, j], result1.final_amps[-1, j])),
             where='post')

ax3 = fig1.add_subplot(4, 1, 3)
ax3.set_title("control amps (second optim)")
ax3.set_xlabel("Time")
ax3.set_ylabel("Control amplitude")
for j in range(n_ctrls):
    ax3.step(result2.time,
             np.hstack((result2.initial_amps[:, j], result2.initial_amps[-1, j])),
             where='post')

ax4 = fig1.add_subplot(4, 1, 4)
ax4.set_title("Final Control Sequences")
ax4.set_xlabel("Time")
ax4.set_ylabel("Control amplitude")
for j in range(n_ctrls):
    ax4.step(result2.time,
             np.hstack((result2.final_amps[:, j], result2.final_amps[-1, j])),
             where='post')

plt.show()


