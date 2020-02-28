# -*- coding: utf-8 -*-
"""
Created on Fri 28 Feb 2020
@author: Alexander Pitchford
@email1: agp1@aber.ac.uk
@email2: alex.pitchford@gmail.com
@organization: Aberystwyth University

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
*** In this case the control is modulated by a time-dependent function ***

The user can experiment with the timeslicing, by means of changing the
number of timeslots and/or total time for the evolution.
Different initial (starting) pulse types can be tried.
The initial and final pulses are displayed in a plot
"""
import numpy as np
import matplotlib.pyplot as plt
import datetime

from qutip import identity, sigmax, sigmaz
from qutip.qip import hadamard_transform
import qutip.logging_utils as logging
logger = logging.get_logger()
#QuTiP control modules
import qutip.control.pulseoptim as cpo

example_name = 'modulated_ctrl'
log_level = logging.INFO

# ****************************************************************
# Define the physics of the problem

nSpins = 1

H_0 = sigmaz()
H_c = sigmax()

U_0 = identity(2**nSpins)
# Hadamard gate
#U_targ = Qobj(np.array([[1,1],[1,-1]], dtype=complex)/np.sqrt(2))
U_targ = hadamard_transform(nSpins)
# ***** Define time evolution parameters *****
# Number of time slots
n_ts = 100
# Time allowed for the evolution
evo_time = 6.0

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
p_type = 'ZERO'
# *************************************************************
# File extension for output files

# Generate list of controls for each timeslot
H_d = H_0
ctrls = []

# Full Hamiltonian
# H = H0 + v(t)*H_c
# v(t) = u(t)*cost(w*t)

times = np.linspace(0, evo_time, n_ts, endpoint=False)
# frequency
w = 3.0

def modulate(t):
    return np.cos(w*t)

# Make list of controls for each timeslot
# In this case, one control, so list len=1 for each tslot.
for k in range(n_ts):
    ctrls.append([modulate(times[k])*H_c])

ctrls = np.array(ctrls, dtype=object)

# ***************************
f_ext = "{}_n_ts{}_ptype{}.txt".format(example_name, n_ts, p_type)

# Run the optimisation
print("\n***********************************")
print("Starting pulse optimisation")
result = cpo.optimize_pulse_unitary(H_d, ctrls, U_0, U_targ, n_ts, evo_time,
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

u0 = result.initial_amps[:, 0]
v0 = u0*modulate(result.time[:-1])
u_t = result.final_amps[:, 0]
v_t = u_t*modulate(result.time[:-1])

# Plot the drift, initial and final amplitudes
fig = plt.figure()

ax = fig.add_subplot(2, 1, 1)
ax.set_title("Initial control amps")
ax.set_ylabel("Control amplitude")
ax.step(result.time, np.hstack([u0, u0[-1]]), where='post', label='ctrl')
ax.step(result.time, np.hstack([v0, v0[-1]]), where='post',
                               label='modulated ctrl')
ax.legend()

ax = fig.add_subplot(2, 1, 2)
ax.set_title("Optimised Control Sequences")
ax.set_xlabel("Time")
ax.set_ylabel("Amplitude")
ax.step(result.time, np.hstack([u_t, u_t[-1]]), where='post', label='ctrl')
ax.step(result.time, np.hstack([v_t, v_t[-1]]), where='post',
                               label='modulated ctrl')
ax.legend()
fig.tight_layout()
plt.show()
