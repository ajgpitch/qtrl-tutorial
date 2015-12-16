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

Example to demonstrate using the control library to determine control
pulses using the ctrlpulseoptim.optimize_pulse_unitary function.
The (default) L-BFGS-B algorithm is used to optimise the pulse to
minimise the fidelity error, which is equivalent maximising the fidelity
to optimal value of 1.

The system in this example is a single qubit in a constant field in z
with a variable control field in x
The target evolution is the Hadamard gate irrespective of global phase

The user can experiment with the timeslicing, by means of changing the
number of timeslots and/or total time for the evolution.
Different initial (starting) pulse types can be tried.
The initial and final pulses are displayed in a plot
"""
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime

from qutip import Qobj, identity, sigmax, sigmaz, tensor
import qutip.logging_utils as logging
logger = logging.get_logger()
#QuTiP control modules
import qutip.control.pulseoptim as cpo
#local import
import plot_util

example_name = '2qubit_interact'
log_level = logging.INFO

# ****************************************************************
# Define the physics of the problem

random.seed(20)
alpha = [random.random(),random.random()]
beta  = [random.random(),random.random()]

Sx = sigmax()
Sz = sigmaz()

H_d = (alpha[0]*tensor(Sx,identity(2)) + 
      alpha[1]*tensor(identity(2),Sx) +
      beta[0]*tensor(Sz,identity(2)) +
      beta[1]*tensor(identity(2),Sz))
H_c = [tensor(Sz,Sz)]
# Number of ctrls
n_ctrls = len(H_c)

q1_0 = q2_0 = Qobj([[1], [0]])

q1_T = q2_T = Qobj([[0], [1]])

psi_0 = tensor(q1_0, q2_0)

psi_T = tensor(q1_T, q2_T)

# ***** Define time evolution parameters *****
# Number of time slots
n_ts = 100
# Time allowed for the evolution
evo_time = 18

# Fidelity error target
fid_err_targ = 1e-15

# Maximum iterations for the optisation algorithm
max_iter = 100
# Maximum (elapsed) time allowed in seconds
max_wall_time = 120


# Initial pulse type
# pulse type alternatives: RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE|
p_type = 'LIN'
# *************************************************************
# File extension for output files

f_ext = "{}_n_ts{}_ptype{}.txt".format(example_name, n_ts, p_type)

# Run the optimisation
print("\n***********************************")
print("Starting pulse optimisation")
result = cpo.optimize_pulse_unitary(H_d, H_c, psi_0, psi_T, n_ts, evo_time, 
                fid_err_targ=fid_err_targ,
                max_iter=max_iter, max_wall_time=max_wall_time, 
#                dyn_params={'oper_dtype':Qobj},
                #phase_option='SU',
                fid_params={'phase_option':'PSU'},
                out_file_ext=f_ext, init_pulse_type=p_type, 
                log_level=log_level, gen_stats=True)

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
for j in range(n_ctrls):
    plot_util.plot_pulse(result.time, result.initial_amps[:, j], ax=ax1)

ax2 = fig1.add_subplot(2, 1, 2)
ax2.set_title("Optimised Control Sequences")
ax2.set_xlabel("Time")
ax2.set_ylabel("Control amplitude")
for j in range(n_ctrls):
    plot_util.plot_pulse(result.time, result.final_amps[:, j], ax=ax2)

plt.show()
