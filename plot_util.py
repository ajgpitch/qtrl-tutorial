# -*- coding: utf-8 -*-

# @date: 14 Dec 2015
# @author: Alexander Pitchford
# @email1: agp1@aber.ac.uk
# @email2: alex.pitchford@gmail.com
# @organization: Aberystwyth University

import numpy as np
import matplotlib.pyplot as plt

"""
Plotting utilities for qtrl
To be moved to the main library at some point
"""

def plot_pulse(time, amps, ax=None, **plot_kwargs):
    """
    Plot the pulse on Matplotlib axes. 
    If ax is passed, then these axes will be used, and it will be left
    to the user to 'show' the plot.
    Otherwise a fresh set of axes will be created and the plot shown - note
    this will block the code.
    
    A simple line plot is produced, with two points per timeslot to
    make clear that the pulse is constant within the timeslot
    
    Parameters
    ----------
    amps : ndarray of float
        These are the pulse amplitudes to be plotted
        If the array is 2d, for instance of OptimResult.final_amps is passed
        when there are muliple controls, then it is assumed that each column
        is a pulse and each will be plotted. If specific parameters are to
        be used for each (label for example), then this function should
        be separately for each pulse (column)
        If the array is 1d, then it is assumed to be a single pulse
        
    time : array[num_tslots+1] of float
        Time of the start of each timeslot
        with the final value being the total evolution time
        OptimResult.time can be used for this
        
    ax : matplotlib.AxesSubplot
        Axes upon which the plot will be made
        
    plot_kwargs : kwargs (dict)
        These will be past to the ax.plot function call
    """
    

    show = False
    if ax is None:
        fig1 = plt.figure()
        ax = fig1.add_subplot(1, 1, 1)
        ax.set_title("Control Pulse")
        ax.set_xlabel("Time")
        ax.set_ylabel("Control amplitude")
        show = True
        
    try:
        if len(time.shape) != 1:
            raise ValueError("time is expected to be a 1d array")
    except:
        raise ValueError("time is expected to be a 1d array")
    
    n_ts = len(time) - 1
    n_pts = 2*n_ts
    
    x = np.zeros([n_pts])
    x[::2] = time[:-1]
    x[1::2] = time[1:]
    y = np.zeros([n_pts])
    try:
        if len(amps.shape) == 1:
            n_pulse = 1
        else:
            n_pulse = amps.shape[1]
            if n_pulse == 1:
                amps = amps[:, 0]
    except:
        raise ValueError(
                "Unable to plot the amps, suspect incorrect array shape")
    
    if n_pulse == 1:
        y[::2] = amps
        y[1::2] = amps
        ax.plot(x, y, **plot_kwargs)
    else:
        for j in range(n_pulse):
            y[::2] = amps[:, j]
            y[1::2] = amps[:, j]
            ax.plot(x, y, **plot_kwargs)
        
    if show:
        plt.show()
    

    