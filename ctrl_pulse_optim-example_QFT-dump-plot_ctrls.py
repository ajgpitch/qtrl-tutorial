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
demonstrated by modifying the initial ctrl pulses.

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

Example to test data dumping.
NOTE: This will put a LOT of files into $HOME/qtrl_dump
"""
import os
import numpy as np
import numpy.matlib as mat
from numpy.matlib import kron
from scipy.sparse import csr_matrix
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

example_name = 'QFT-dump'
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

print("Check unitary (should be I) {}".format(U_targ.dag()*U_targ))

# ***** Define time evolution parameters *****
# Number of time slots
n_ts = 10
# Time allowed for the evolution
evo_time = 10

# ***** Define the termination conditions *****
# Fidelity error target
fid_err_targ = 1e-14
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
                amp_lbound=-10.0, amp_ubound=10.0,
                fid_err_targ=fid_err_targ, min_grad=min_grad,
                max_iter=max_iter, max_wall_time=max_wall_time,
#                optim_method='LBFGSB',
                method_params={'max_metric_corr':10, 'accuracy_factor':1e-3,
                                'ftol':1e-15},
                optim_method='fmin_l_bfgs_b',
                optim_params={'dumping':'FULL', 'dump_to_file':False},
#                optim_method='l-bfgs-b',
                dyn_type='UNIT',
                dyn_params={'dumping':'FULL', 'dump_to_file':True,
                            'dump_dir':"~/QFT-dyndump"},
#                dyn_params={'oper_dtype':Qobj},
#                prop_type='APPROX',
#                fid_type='TDAPPROX',
                fid_params={'phase_option':'PSU'},
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


dyn.test_out_files = 0
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

if dyn.dump:
    # Dump configuration
    # specify folder
    #dyn.dump.dump_dir = "~/QFT-dump"

    # specify base name
    dyn.dump.fname_base = "QFT-example"

    # specify summary file
    #dyn.dump.summary_file = os.path.join(dyn.dump.dump_dir, "QFT-summary-specific.dat")

    # use tab for separator
    dyn.dump.summary_sep = '\t'
    dyn.dump.data_sep = '\t'

    # write interactive (or not)
    # customise calc obj dumps

if optim.dump:
    # Dump configuration
    # specify folder
    optim.dump.dump_dir = "~/QFT-optim-dump"

    # specify base name
    optim.dump.fname_base = "QFT-optim"

    # specify summary file
    optim.dump.summary_file = os.path.join(optim.dump.dump_dir, "QFT-optim_sum-specific.dat")

    # write interactive (or not)
    # customise calc obj dumps

    # use tab for separator
    optim.dump.summary_sep = '\t'
    optim.dump.data_sep = '\t'

dyn.initialize_controls(init_amps)

print("dimensional norm: {}".format(dyn.fid_computer.dimensional_norm))
print("Initial infidelity: {}".format(dyn.fid_computer.get_fid_err()))
#print("onto_evo_target: {}".format(dyn.onto_evo_target))

# Save initial amplitudes to a text file
pulsefile = "ctrl_amps_initial_" + f_ext
dyn.save_amps(pulsefile, times="exclude")
if (log_level <= logging.INFO):
    print("Initial amplitudes output to file: " + pulsefile)

if check_gradient:
    print("***********************************")
    print("Checking gradient")
    func = optim.fid_err_func_wrapper
    grad = optim.fid_err_grad_wrapper
    x0 = dyn.ctrl_amps.flatten()
    grad_diff = check_grad(func, grad, x0)
    print("Normalised grad diff: {}".format(grad_diff))

print("***********************************")
print("Starting pulse optimisation")
result = optim.run_optimization()

# Save final amplitudes to a text file
pulsefile = "ctrl_amps_final_" + f_ext
dyn.save_amps(pulsefile)
if (log_level <= logging.INFO):
    print("Final amplitudes output to file: " + pulsefile)

print("\n***********************************")
print("Optimising complete.")
if result.stats:
    print(" Stats follow:")
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

if dyn.dump and not dyn.dump.write_to_file:
    # standard location
    dyn.dump.writeout()
    # use a specific file stream
    dump_dest = os.path.expanduser("~/qtrl-QFT-dump1.txt")
    f = open(dump_dest, 'wb')
    dyn.dump.writeout(f)
    f.close()

    # use a specific file name
    dump_dest = "qtrl-QFT-dump1.txt"
    dyn.dump.writeout(dump_dest)

if optim.dump and not optim.dump.write_to_file:
    # standard location
    optim.dump.writeout()
    # use a specific file stream
    dump_dest = os.path.expanduser("~/qtrl-QFT-optimdump1.txt")
    f = open(dump_dest, 'wb')
    optim.dump.writeout(f)
    f.close()

    # use a specific file name
    dump_dest = "qtrl-QFT-optimdump2.txt"
    optim.dump.writeout(dump_dest)

n_iter = len(optim.dump.iter_summary)
ctrls_in_iter = np.empty([n_iter, n_ts, n_ctrls])
for i, iter_item in enumerate(optim.dump.iter_summary):
    ctrls_in_iter[i, :, :] = \
                dyn.dump.evo_dumps[iter_item.fid_func_call_num].ctrl_amps



# Plot the initial and final amplitudes
fig1 = plt.figure()
ax1 = fig1.add_subplot(2, 1, 1)
ax1.set_title("Initial control amps")
ax1.set_xlabel("Time")
ax1.set_ylabel("Control amplitude")
for j in range(n_ctrls):
    ax1.step(result.time,
             np.hstack((result.initial_amps[:, j], result.initial_amps[-1, j])),
             where='post')

ax2 = fig1.add_subplot(2, 1, 2)
ax2.set_title("Optimised Control Sequences")
ax2.set_xlabel("Time")
ax2.set_ylabel("Control amplitude")
for j in range(n_ctrls):
    ax2.step(result.time,
             np.hstack((result.final_amps[:, j], result.final_amps[-1, j])),
             where='post')
plt.tight_layout()

# Plot the progress of amplitudes towards final values
fig2 = plt.figure()
ax1 = fig2.add_subplot(2, 1, 1)
ax1.set_title("First timeslot amplitude")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Control amplitude")
for j in range(n_ctrls):
    ax1.step(range(1, n_iter+1), ctrls_in_iter[:, 0, j], where='post')

ax2 = fig2.add_subplot(2, 1, 2)
ax2.set_title("Last timeslot amplitude")
ax2.set_xlabel("Time")
ax2.set_ylabel("Control amplitude")
for j in range(n_ctrls):
    ax2.step(range(1, n_iter+1), ctrls_in_iter[:, -1, j], where='post')

plt.tight_layout()
plt.show()
