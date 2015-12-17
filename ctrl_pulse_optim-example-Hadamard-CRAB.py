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

#for p in sys.path:
#    print p
#    
#sys.exit()

#QuTiP
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
#Sx = np.array([[0, 1], [1, 0]], dtype=complex)
#Sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
#Sz = np.array([[1, 0], [0, -1]], dtype=complex)
#Si = mat.eye(2)/2

H_d = sigmaz()
H_c = [sigmax()]
# Number of ctrls
n_ctrls = len(H_c)

U_0 = identity(2**nSpins)
# Hadamard gate
#U_targ = Qobj(np.array([[1,1],[1,-1]], dtype=complex)/np.sqrt(2))
U_targ = hadamard_transform(nSpins)
# ***** Define time evolution parameters *****
# Number of time slots
n_ts = 12
# Time allowed for the evolution
evo_time = 10

# ***** Define the termination conditions *****
# Fidelity error target
fid_err_targ = 1e-5
# Maximum iterations for the optisation algorithm
max_iter = 500
# Maximum (elapsed) time allowed in seconds
max_wall_time = 60
p_type = 'DEF'

# Initial pulse type
# pulse type alternatives: RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE|
# File extension for output files

save_plot = True
PLOT_FEXT = 'png'
DATA_FEXT = 'txt'
f_end = "{}_n_ts{}".format(example_name, n_ts)
f_ext = "{}.{}".format(f_end, DATA_FEXT)

# Run the optimisation
print("\n***********************************")
print("Starting pulse optimisation")
result = cpo.opt_pulse_crab_unitary(H_d, H_c, U_0, U_targ, n_ts, evo_time, 
                fid_err_targ=fid_err_targ, 
                max_iter=max_iter, max_wall_time=max_wall_time, 
                init_coeff_scaling=0.5,
                alg_params={'crab_pulse_params':{'randomize_coeffs':False, 
                                                 'randomize_freqs':False}},
#                optim_method='fmin_l_bfgs_b',
#                optim_method='l-bfgs-b',
#                method_params={'xtol':1e-8},
                guess_pulse_type='GAUSSIAN', 
                guess_pulse_params={'variance':0.1*evo_time},
                guess_pulse_scaling=1.0, guess_pulse_offset=1.0,
#                guess_pulse_params={'variance':0.1*evo_time},
                amp_lbound=None, amp_ubound=None,
                ramping_pulse_type='GAUSSIAN_EDGE', 
                ramping_pulse_params={'decay_time':evo_time/100.0},
                out_file_ext=f_ext,
                log_level=log_level, gen_stats=True)
#,
                
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
plt.show()
