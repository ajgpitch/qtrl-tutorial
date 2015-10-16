# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 15:12:53 2014
@author: Alexander Pitchford
@email1: agp1@aber.ac.uk
@email2: alex.pitchford@gmail.com
@organization: Aberystwyth University
@supervisor: Daniel Burgarth

The main purpose of this file is to demonstrate how to implement and use
a custom fidelity class. It is otherwise the same as the Lindbladian example.
For convenience the custom fidelity class is implemented in this file,
however, it is probably better practice to implement it in its own file

This in an open quantum system example, with a single qubit subject to
an amplitude damping channel. The target evolution is the Hadamard gate.
The user can experiment with the strength of the amplitude damping by
changing the gamma variable value

The user can experiment with the timeslicing, by means of changing the
number of timeslots and/or total time for the evolution.
Different initial (starting) pulse types can be tried.
The initial and final pulses are displayed in a plot

"""
import numpy as np
import matplotlib.pyplot as plt
import datetime
import timeit

#QuTiP
from qutip import Qobj, identity, sigmax, sigmay, sigmaz, tensor
from qutip.qip import hadamard_transform
import qutip.logging_utils as logging
logger = logging.get_logger()
#QuTiP control modules
import qutip.control.pulseoptim as cpo
import qutip.control.fidcomp as fidcomp
import qutip.control.errors as errors

example_name = 'Lindblad-cust_fid'
log_level = logging.INFO

class FidCompCustom(fidcomp.FidelityComputer):

    """
    Customised fidelity computer copied the TraceDiff fidelity computer
    At this stage it does nothing different other than print a DEBUG message
    to say that it is 'custom'
    Note: It is recommended to put this class in a separate file in a real
        project
        
    Computes fidelity error and gradient for general system dynamics
    by calculating the the fidelity error as the trace of the overlap
    of the difference between the target and evolution resulting from
    the pulses with the transpose of the same.
    This should provide a distance measure for dynamics described by matrices
    Note the gradient calculation is taken from:
    'Robust quantum gates for open systems via optimal control:
    Markovian versus non-Markovian dynamics'
    Frederik F Floether, Pierre de Fouquieres, and Sophie G Schirmer

    Attributes
    ----------
    scale_factor : float
        The fidelity error calculated is of some arbitary scale. This
        factor can be used to scale the fidelity error such that it may
        represent some physical measure
        If None is given then it is caculated as 1/2N, where N
        is the dimension of the drift, when the Dynamics are initialised.
    """

    def reset(self):
        fidcomp.FidelityComputer.reset(self)
        self.id_text = 'TRACEDIFF'
        self.scale_factor = None
        self.uses_evo_t2end = True
        if not self.parent.prop_computer.grad_exact:
            raise errors.UsageError(
                "This FidelityComputer can only be"
                " used with an exact gradient PropagatorComputer.")
        self.apply_params()
        
    def init_comp(self):
        """
        initialises the computer based on the configuration of the Dynamics
        Calculates the scale_factor is not already set
        """
        if self.scale_factor is None:
            self.scale_factor = 1.0 / (2.0*self.parent.get_drift_dim())
            if self.log_level <= logging.DEBUG:
                logger.debug("Scale factor calculated as {}".format(
                    self.scale_factor))

    def get_fid_err(self):
        """
        Gets the absolute error in the fidelity
        """
        if not self.fidelity_current:
            dyn = self.parent
            dyn.compute_evolution()
            n_ts = dyn.num_tslots
            if self.log_level <= logging.DEBUG:
                logger.debug("**** Computing custom fidelity ****")
            evo_final = dyn.evo_init2t[n_ts]
            evo_f_diff = dyn.target - evo_final
            if self.log_level <= logging.DEBUG_VERBOSE:
                logger.log(logging.DEBUG_VERBOSE, "Calculating TraceDiff "
                           "fidelity...\n Target:\n{}\n Evo final:\n{}\n"
                           "Evo final diff:\n{}".format(dyn.target, evo_final,
                                                        evo_f_diff))

            # **** CUSTOMISE this line below *****
            # Calculate the fidelity error using the trace difference norm
            # Note that the value should have not imagnary part, so using
            # np.real, just avoids the complex casting warning
            self.fid_err = self.scale_factor*np.real(
                np.trace(evo_f_diff.conj().T.dot(evo_f_diff)))

            if np.isnan(self.fid_err):
                self.fid_err = np.Inf

            if dyn.stats is not None:
                    dyn.stats.num_fidelity_computes += 1

            self.fidelity_current = True
            if self.log_level <= logging.DEBUG:
                logger.debug("Fidelity error: {}".format(self.fid_err))

        return self.fid_err

    def get_fid_err_gradient(self):
        """
        Returns the normalised gradient of the fidelity error
        in a (nTimeslots x n_ctrls) array
        The gradients are cached in case they are requested
        mutliple times between control updates
        (although this is not typically found to happen)
        """
        if not self.fid_err_grad_current:
            dyn = self.parent
            self.fid_err_grad = self.compute_fid_err_grad()
            self.fid_err_grad_current = True
            if dyn.stats is not None:
                dyn.stats.num_grad_computes += 1

            self.grad_norm = np.sqrt(np.sum(self.fid_err_grad**2))
            if self.log_level <= logging.DEBUG_INTENSE:
                logger.log(logging.DEBUG_INTENSE, "fidelity error gradients:\n"
                           "{}".format(self.fid_err_grad))

            if self.log_level <= logging.DEBUG:
                logger.debug("Gradient norm: "
                             "{} ".format(self.grad_norm))

        return self.fid_err_grad

    def compute_fid_err_grad(self):
        """
        Calculate exact gradient of the fidelity error function
        wrt to each timeslot control amplitudes.
        Uses the trace difference norm fidelity
        These are returned as a (nTimeslots x n_ctrls) array
        """
        dyn = self.parent
        n_ctrls = dyn.get_num_ctrls()
        n_ts = dyn.num_tslots
        if self.log_level <= logging.DEBUG:
            logger.debug("**** Computing custom fidelity gradient ****")
            
        # create n_ts x n_ctrls zero array for grad start point
        grad = np.zeros([n_ts, n_ctrls])

        dyn.tslot_computer.flag_all_calc_now()
        dyn.compute_evolution()

        # loop through all ctrl timeslots calculating gradients
        time_st = timeit.default_timer()
        evo_final = dyn.evo_init2t[n_ts]
        evo_f_diff = dyn.target - evo_final

        for j in range(n_ctrls):
            for k in range(n_ts):
                fwd_evo = dyn.evo_init2t[k]
                evo_grad = dyn.prop_grad[k, j].dot(fwd_evo)

                if k+1 < n_ts:
                    owd_evo = dyn.evo_t2end[k+1]
                    evo_grad = owd_evo.dot(evo_grad)
                # **** CUSTOMISE this line below *****
                g = -2*self.scale_factor*np.real(
                    np.trace(evo_f_diff.conj().T.dot(evo_grad)))
                if np.isnan(g):
                    g = np.Inf

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
Si = identity(2)

Sd = Qobj(np.array([[0, 1],
             [0, 0]]))
Sm = Qobj(np.array([[0, 0],
             [1, 0]]))
Sd_m = Qobj(np.array([[1, 0],
              [0, 0]]))
Sm_d = Qobj(np.array([[0, 0],
              [0, 1]]))

#Amplitude damping#
#Damping rate:
gamma = 0.1
L0_Ad = gamma*(2*tensor(Sm, Sd.trans()) - 
            (tensor(Sd_m, Si) + tensor(Si, Sd_m.trans())))
#sigma X control
LC_x = -1j*(tensor(Sx, Si) - tensor(Si, Sx))
#sigma Y control
LC_y = -1j*(tensor(Sy, Si) - tensor(Si, Sy.trans()))
#sigma Z control
LC_z = -1j*(tensor(Sz, Si) - tensor(Si, Sz))

#Drift
drift = L0_Ad
#Controls
ctrls = [LC_z, LC_x]
# Number of ctrls
n_ctrls = len(ctrls)

initial = identity(4)
#Target
#Hadamard gate
had_gate = hadamard_transform(1)
target_DP = tensor(had_gate, had_gate)

# ***** Define time evolution parameters *****
# Number of time slots
n_ts = 200
# Time allowed for the evolution
evo_time = 2

# ***** Define the termination conditions *****
# Fidelity error target
fid_err_targ = 1e-3
# Maximum iterations for the optisation algorithm
max_iter = 200
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
print("***********************************")
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
optim = cpo.create_pulse_optimizer(drift, ctrls, initial, target_DP, n_ts, evo_time, 
                fid_err_targ=fid_err_targ, min_grad=min_grad, 
                max_iter=max_iter, max_wall_time=max_wall_time, 
                init_pulse_type=p_type, 
                log_level=log_level, gen_stats=True)
                

dyn = optim.dynamics
pgen = optim.pulse_generator
# **** CUSTOMISE this is where the custom fidelity is specified *****
dyn.fid_computer = FidCompCustom(dyn)

p_gen = optim.pulse_generator
init_amps = np.zeros([n_ts, n_ctrls])
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

print("\n***********************************")
print("Optimising complete. Stats follow:")
result.stats.report()
print("Final evolution\n{}\n".format(result.evo_full_final))

print("********* Summary *****************")
print("Final fidelity error {}".format(result.fid_err))
print("Terminated due to {}".format(result.termination_reason))
print("Number of iterations {}".format(result.num_iter))
#print("wall time: ", result.wall_time
print("Completed in {} HH:MM:SS.US".\
        format(datetime.timedelta(seconds=result.wall_time)))
# print("Final gradient normal {}".format(result.grad_norm_final)
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
