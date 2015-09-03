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

The main purpose of this file is to demonstrate how to implement and use
a custom fidelity class. It is otherwise the same as the QFT example.
For convenience the custom fidelity class is implemented in this file,
however, it is probably better practice to implement it in its own file

The (default) L-BFGS-B algorithm is used to optimise the pulse to
minimise the fidelity error, which is equivalent maximising the fidelity
to optimal value of 1.

The system in this example is two qubits in constant fields in x, y and z
with a variable independant controls fields in x and y acting on each qubit
The target evolution is the QFT gate. The user can experiment with the
different:
    phase options - phase_option = SU or PSU
    propagtor computer type prop_type = DIAG or FRECHET

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
import timeit

#QuTiP
from qutip import Qobj, identity, sigmax, sigmay, sigmaz, tensor
import qutip.logging as logging
logger = logging.get_logger()
#QuTiP control modules
import qutip.control.pulseoptim as cpo
import qutip.control.pulsegen as pulsegen
import qutip.control.fidcomp as fidcomp
from qutip.qip.algorithms import qft

example_name = 'QFT'
log_level=logging.DEBUG
logger.setLevel(log_level)

class FidCompCustom(fidcomp.FidCompUnitary):
    """
    Customised fidelity computer based on the Unitary fidelity computer
    At this stage it does nothing different other than print a DEBUG message
    to say that it is 'custom'
    Note: It is recommended to put this class in a separate file in a real
        project
    """

    def get_fid_err(self):
        """
        Note: This a copy from FidCompUnitary, if you don't need to change it
                then remove it
        Gets the absolute error in the fidelity
        """
        return np.abs(1 - self.get_fidelity())

    def get_fidelity(self):
        """
        Note: This a copy from FidCompUnitary, if you don't need to change it
                then remove it
        Gets the appropriately normalised fidelity value
        The normalisation is determined by the fid_norm_func pointer
        which should be set in the config
        """
        if not self.fidelity_current:
            self.fidelity = \
                self.fid_norm_func(self.get_fidelity_prenorm())
            self.fidelity_current = True
            if self.log_level <= logging.DEBUG:
                logger.debug("Fidelity (normalised): {}".format(self.fidelity))

        return self.fidelity

    def get_fidelity_prenorm(self):
        """
        Gets the current fidelity value prior to normalisation
        Note the gradient function uses this value
        The value is cached, because it is used in the gradient calculation
        """
        if not self.fidelity_prenorm_current:
            dyn = self.parent
            if self.log_level <= logging.DEBUG:
                logger.debug("**** Computing custom fidelity ****")
            k = dyn.tslot_computer.get_timeslot_for_fidelity_calc()
            dyn.compute_evolution()
            # **** CUSTOMISE this line below *****
            f = np.trace(dyn.evo_init2t[k].dot(dyn.evo_t2targ[k]))
            self.fidelity_prenorm = f
            self.fidelity_prenorm_current = True
            if dyn.stats is not None:
                    dyn.stats.num_fidelity_computes += 1
            if self.log_level <= logging.DEBUG:
                logger.debug("Fidelity (pre normalisation): {}".format(
                    self.fidelity_prenorm))
        return self.fidelity_prenorm

    def get_fid_err_gradient(self):
        """
        Note: This a copy from FidCompUnitary, if you don't need to change it
                then remove it
        Returns the normalised gradient of the fidelity error
        in a (nTimeslots x n_ctrls) array
        The gradients are cached in case they are requested
        mutliple times between control updates
        (although this is not typically found to happen)
        """
        if not self.fid_err_grad_current:
            dyn = self.parent
            grad_prenorm = self.compute_fid_grad()
            if self.log_level <= logging.DEBUG_INTENSE:
                logger.log(logging.DEBUG_INTENSE, "pre-normalised fidelity "
                           "gradients:\n{}".format(grad_prenorm))
            # AJGP: Note this check should not be necessary if dynamics are
            #       unitary. However, if they are not then this gradient
            #       can still be used, however the interpretation is dubious
            if self.get_fidelity() >= 1:
                self.fid_err_grad = self.grad_norm_func(grad_prenorm)
            else:
                self.fid_err_grad = -self.grad_norm_func(grad_prenorm)

            self.fid_err_grad_current = True
            if dyn.stats is not None:
                dyn.stats.num_grad_computes += 1

            self.grad_norm = np.sqrt(np.sum(self.fid_err_grad**2))
            if self.log_level <= logging.DEBUG_INTENSE:
                logger.log(logging.DEBUG_INTENSE, "Normalised fidelity error "
                           "gradients:\n{}".format(self.fid_err_grad))

            if self.log_level <= logging.DEBUG:
                logger.debug("Gradient (sum sq norm): "
                             "{} ".format(self.grad_norm))

        return self.fid_err_grad

    def compute_fid_grad(self):
        """
        Calculates exact gradient of function wrt to each timeslot
        control amplitudes. Note these gradients are not normalised
        These are returned as a (nTimeslots x n_ctrls) array
        """
        dyn = self.parent
        n_ctrls = dyn.get_num_ctrls()
        n_ts = dyn.num_tslots
        
        if self.log_level <= logging.DEBUG:
            logger.debug("**** Computing custom fidelity gradient ****")

        # create n_ts x n_ctrls zero array for grad start point
        grad = np.zeros([n_ts, n_ctrls], dtype=complex)

        dyn.tslot_computer.flag_all_calc_now()
        dyn.compute_evolution()

        # loop through all ctrl timeslots calculating gradients
        time_st = timeit.default_timer()
        for j in range(n_ctrls):
            for k in range(n_ts):
                owd_evo = dyn.evo_t2targ[k+1]
                fwd_evo = dyn.evo_init2t[k]
                # **** CUSTOMISE this line below *****
                g = np.trace(owd_evo.dot(dyn.prop_grad[k, j]).dot(fwd_evo))
                grad[k, j] = g
        if dyn.stats is not None:
            dyn.stats.wall_time_gradient_compute += \
                timeit.default_timer() - time_st
        return grad
        
# ****************************************************************
# Define the physics of the problem
Sx = sigmax()
Sy = sigmay()
Sz = sigmaz()
Si = 0.5*identity(2)

# Drift Hamiltonian
H_d = 0.5*(tensor(Sx, Sx) + tensor(Sy, Sy) + tensor(Sz, Sz))
# The (four) control Hamiltonians
H_c = [tensor(Sx, Si), tensor(Sy, Si), tensor(Si, Sx), tensor(Si, Sy)]
n_ctrls = len(H_c)
# start point for the gate evolution
U_0 = identity(4)
# Target for the gate evolution - Quantum Fourier Transform gate
U_targ = qft.qft(2)

# ***** Define time evolution parameters *****
# Number of time slots
n_ts = 100
# Time allowed for the evolution
evo_time = 10

# ***** Define the termination conditions *****
# Fidelity error target
fid_err_targ = 1e-5
# Maximum iterations for the optisation algorithm
max_iter = 200
# Maximum (elapsed) time allowed in seconds
max_wall_time = 120
# Minimum gradient (sum of gradients squared)
# as this tends to 0 -> local minima has been found
min_grad = 1e-20

# Initial pulse type
# pulse type alternatives: RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE|
p_type = 'RNDWAVES'
# *************************************************************
# File extension for output files

f_ext = "{}_n_ts{}_ptype{}.txt".format(example_name, n_ts, p_type)

print("\n***********************************")
print("Creating optimiser objects")
optim = cpo.create_pulse_optimizer(H_d, H_c, U_0, U_targ, n_ts, evo_time, 
                amp_lbound=-10.0, amp_ubound=10.0, 
                fid_err_targ=fid_err_targ, min_grad=min_grad, 
                max_iter=max_iter, max_wall_time=max_wall_time, 
#                optim_method='LBFGSB', 
#                method_params={'max_metric_corr':20, 'accuracy_factor':1e8},
#                optim_method='fmin_l_bfgs_b',
                optim_method='l-bfgs-b',
                dyn_type='UNIT', 
                prop_type='DIAG', 
                fid_type='UNIT', fid_params={'phase_option':'PSU'}, 
                init_pulse_type=p_type, pulse_scaling=1.0,
                log_level=log_level, gen_stats=True)
                
print("\n***********************************")
print("Configuring optimiser objects")
# **** Set some optimiser config parameters ****
optim.test_out_files = 0
dyn = optim.dynamics
# **** CUSTOMISE this is where the custom fidelity is specified *****
dyn.fid_computer = FidCompCustom(dyn)

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
        p_gen.offset = sf = float(j) - float(n_ctrls - 1)/2
        init_amps[:, j] = p_gen.gen_pulse()
else:
    # Should be random pulse
    for j in range(n_ctrls):
        init_amps[:, j] = p_gen.gen_pulse()

dyn.initialize_controls(init_amps)

# Save initial amplitudes to a text file
pulsefile = "ctrl_amps_initial_" + f_ext
dyn.save_amps(pulsefile)
if (log_level <= logging.INFO):
    print("Initial amplitudes output to file: " + pulsefile)

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


