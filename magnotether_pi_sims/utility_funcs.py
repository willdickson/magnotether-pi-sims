import numpy as np

def delayed_pulse(t, t_start=0.0, period=1.0, duty=0.5, minval=0.0, maxval=1.0):
    if t < t_start:
        rval = 1.0
    else:
        rval = pulse(t-t_start, period=period, duty=duty, minval=minval, maxval=maxval)
    return rval

def pulse(t, period=1.0, duty=0.5, minval=0.0, maxval=1.0):
    n = int(np.floor(t/period))
    frac = (t - n*period)/period
    if frac <= duty:
        rval = 0.0
    else:
        rval = 1.0
    return rval
