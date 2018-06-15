# -*- coding: utf-8 -*-

"""

Created on Tue Feb 10 15:12:53 2014
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

This in an open quantum system example, with a single qubit subject to
an amplitude damping channel. The target evolution is the Hadamard gate.
For a d dimensional quantum system in general we represent the Lindbladian
as a d^2 x d^2 dimensional matrix by vectorizing the denisty operator (row vectorization).
Here done for the Lindbladian that describes the amplitude damping channel
and the coherent drift- and control generators.
The user can experiment with the strength of the amplitude damping by
changing the gamma variable value

The user can experiment with the timeslicing, by means of changing the
number of timeslots and/or total time for the evolution.
Different initial (starting) pulse types can be tried.
The initial and final pulses are displayed in a plot
"""

import sys
import numpy as np
import numpy.matlib as mat
from numpy.matlib import kron
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import datetime

#QuTiP
from qutip import Qobj, identity, sigmax, sigmay, sigmaz, sigmam, tensor
from qutip.superoperator import (spre, sprepost, liouvillian,
                                vector_to_operator, operator_to_vector)
from qutip.qip import hadamard_transform
import qutip.logging_utils as logging
logger = logging.get_logger()

#QuTiP control modules
import qutip.control.pulseoptim as cpo

example_name = 'Lindblad'
log_level = logging.INFO

vectorization = 'column'   # row|column

# ****************************************************************
# Define the physics of the problem
Sx = sigmax()
Sy = sigmay()
Sz = sigmaz()
Si = identity(2)
Sm = sigmam()
#Hadamard gate
had_gate = hadamard_transform(1)

# Hamiltonian
Del = 0.1    # Tunnelling term
wq = 1.0   # Energy of the 2-level system.
H0 = 0.5*wq*sigmaz() + 0.5*Del*sigmax()

#Amplitude damping#
#Damping rate:
gamma = 0.1

if vectorization == 'row':
    # Row vectorisation version

    D0_Ad = gamma*(tensor(Sm, Sm.conj()) -
                   0.5*(tensor(Sm.dag()*Sm, Si) + tensor(Si, Sm.trans()*Sm)))
    L0 = -1j*(tensor(H0, Si) - tensor(Si, H0)) + D0_Ad

    #sigma X control
    LC_x = -1j*(tensor(Sx, Si) - tensor(Si, Sx))
    #sigma Y control
    LC_y = -1j*(tensor(Sy, Si) - tensor(Si, Sy.trans()))
    #sigma Z control
    LC_z = -1j*(tensor(Sz, Si) - tensor(Si, Sz))

    E0 = tensor(Si, Si)
    E_targ = tensor(had_gate, had_gate)

elif vectorization == 'column':

    # qutip column vectorisation
    L0 = liouvillian(H0, [np.sqrt(gamma)*Sm])

    #sigma X control
    LC_x = liouvillian(Sx)
    #sigma Y control
    LC_y = liouvillian(Sy)
    #sigma Z control
    LC_z = liouvillian(Sz)

    E0 = sprepost(Si, Si)
    E_targ = sprepost(had_gate, had_gate)

else:
    raise RuntimeError("No option for vectorization={}".format(vectorization))

print("L0:\n{}\n".format(L0))
print("LC_x:\n{}\n".format(LC_x))
print("LC_y:\n{}\n".format(LC_y))
print("LC_z:\n{}\n".format(LC_z))


#Drift
drift = L0

#Controls
ctrls = [LC_x, LC_z]
#ctrls = [LC_y]
ctrls = [LC_x]

# Number of ctrls
n_ctrls = len(ctrls)

initial = E0

#Target

target_DP = E_targ

# ***** Define time evolution parameters *****
# Number of time slots
n_ts = 10

# Time allowed for the evolution
evo_time = 10

# ***** Define the termination conditions *****
# Fidelity error target
fid_err_targ = 1e-3

# Maximum iterations for the optisation algorithm
max_iter = 500

# Maximum (elapsed) time allowed in seconds
max_wall_time = 30

# Minimum gradient (sum of gradients squared)
# as this tends to 0 -> local minima has been found
min_grad = 1e-20

# Initial pulse type
# pulse type alternatives: RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE|
p_type = 'LIN'

# *************************************************************

# File extension for output files
f_ext = "{}_n_ts{}_ptype{}.txt".format(example_name, n_ts, p_type)

# Run the optimisation

print("\n***********************************")
print("Starting pulse optimisation")

# Note that this call will take the defaults
#    dyn_type='GEN_MAT'
# This means that matrices that describe the dynamics are assumed to be
# general, i.e. the propagator can be calculated using:
# expm(combined_dynamics*dt)
#    prop_type='FRECHET'
# and the propagators and their gradients will be calculated using the
# Frechet method, i.e. an exact gradent
#    fid_type='TRACEDIFF'
# and that the fidelity error, i.e. distance from the target, is give
# by the trace of the difference between the target and evolved operators

result = cpo.optimize_pulse(drift, ctrls, initial, target_DP, n_ts, evo_time,
                fid_err_targ=fid_err_targ, min_grad=min_grad,
                max_iter=max_iter, max_wall_time=max_wall_time,
                amp_lbound=-10.0, amp_ubound=10.0,
#                dyn_params={'oper_dtype':Qobj},
#                prop_type='AUG_MAT',
#                fid_type='UNIT',
                out_file_ext=f_ext, init_pulse_type=p_type,
                log_level=log_level, gen_stats=True)

print("***********************************")
print("\nOptimising complete. Stats follow:")
result.stats.report()
print("Final evolution\n{}\n".format(result.evo_full_final))
print("********* Summary *****************")
print("Initial fidelity error {}".format(result.initial_fid_err))
print("Final fidelity error {}".format(result.fid_err))
print("Terminated due to {}".format(result.termination_reason))
print("Number of iterations {}".format(result.num_iter))
#print("wall time: ", result.wall_time
print("Completed in {} HH:MM:SS.US".format(
                datetime.timedelta(seconds=result.wall_time)))

# print("Final gradient normal {}".format(result.grad_norm_final)
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
fig1.tight_layout()
plt.show()
