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
import numpy as np
import matplotlib.pyplot as plt
import datetime

from qutip import Qobj, identity, sigmax, sigmaz
from qutip.qip import hadamard_transform
import qutip.logging_utils as logging
logger = logging.get_logger()
#QuTiP control modules
import qutip.control.pulseoptim as cpo

example_name = 'Hadamard'
log_level = logging.INFO

# ****************************************************************
# Define the physics of the problem

nSpins = 1

H_0 = sigmaz()
H_c = [sigmax()]
# Number of ctrls
n_ctrls = len(H_c)

U_0 = identity(2**nSpins)
# Hadamard gate
#U_targ = Qobj(np.array([[1,1],[1,-1]], dtype=complex)/np.sqrt(2))
U_targ = hadamard_transform(nSpins)
# ***** Define time evolution parameters *****
# Number of time slots
n_ts = 100
# Time allowed for the evolution
evo_time = 6

# ***** Define the termination conditions *****
# Fidelity error target
fid_err_targ = 1e-10
# Maximum iterations for the optisation algorithm
max_iter = 200
# Maximum (elapsed) time allowed in seconds
max_wall_time = 120
# Minimum gradient (sum of gradients squared)
# as this tends to 0 -> local minima has been found
min_grad = 1e-20


# Initial pulse type
# pulse type alternatives: RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE|
p_type = 'LIN'
# *************************************************************
# File extension for output files

# ***************************
# Set up time dependent drift
# comment in/out desired drift amps
# *** sin wave modulated ***
# cycles = 10
# drift_amps = [np.sin(2*cycles*np.pi*float(k)/n_ts) for k in range(n_ts)]

# *** flat ***
# - this should produce same as a fixed drift
# drift_amps = np.ones([n_ts], dtype=float)

# *** step ***
drift_amps = [np.round(float(k)/n_ts) for k in range(n_ts)]

# Generate list of drifts for each time slot
H_d = [drift_amps[k]*H_0 for k in range(n_ts)]
    
# ***************************
f_ext = "{}_n_ts{}_ptype{}.txt".format(example_name, n_ts, p_type)

# Run the optimisation
print("\n***********************************")
print("Starting pulse optimisation")
result = cpo.optimize_pulse_unitary(H_d, H_c, U_0, U_targ, n_ts, evo_time, 
                fid_err_targ=fid_err_targ, min_grad=min_grad, 
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

# Plot the drift, initial and final amplitudes
fig1 = plt.figure()

ax1 = fig1.add_subplot(3, 1, 1)
ax1.set_title("Drift amps")
#ax1.set_xlabel("Time")
ax1.set_ylabel("Drift amplitude")
ax1.step(result.time, 
         np.hstack((drift_amps[:], drift_amps[-1])), 
         where='post')
             
ax1 = fig1.add_subplot(3, 1, 2)
ax1.set_title("Initial control amps")
#ax1.set_xlabel("Time")
ax1.set_ylabel("Control amplitude")
for j in range(n_ctrls):
    ax1.step(result.time, 
             np.hstack((result.initial_amps[:, j], result.initial_amps[-1, j])), 
             where='post')
             
ax2 = fig1.add_subplot(3, 1, 3)
ax2.set_title("Optimised Control Sequences")
ax2.set_xlabel("Time")
ax2.set_ylabel("Control amplitude")
for j in range(n_ctrls):
    ax2.step(result.time, 
             np.hstack((result.final_amps[:, j], result.final_amps[-1, j])), 
             where='post')
plt.show()
