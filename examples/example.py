import pickle
import functools
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from magnotether_pi_sims import PIController
from magnotether_pi_sims import utility_funcs

dt = 0.1
duration = 1000.0
speed = 0.1

igain_list = [0.047, 0.0]
color_list = ['g', 'b']

fbscale = functools.partial(
        utility_funcs.delayed_pulse, 
        t_start=200.0, 
        period=50.0, 
        duty=0.1, 
        )

param = {
        'dcoef'    : 0.069, 
        'pgain'    : 3.08,
        'ileak'    : 0.0, 
        'setpt'    : 0.0, 
        'fbscale'  : fbscale, 
        }

eq_bais = 30.0
param['bias']  = eq_bais*param['dcoef']


num_pts = int(np.ceil(duration/dt))
t = dt*np.arange(num_pts)
fbscale_vals = [fbscale(item) for item in t]


sim_data = {}
for igain, color in zip(igain_list, color_list):
    #if igain > 0:
    #    y_init = [0.0, -param['bias']/igain]
    #else:
    #    y_init = [0.0, 0.0]
    y_init = [0.0, 0.0]

    param['igain'] = igain
    ctlr = PIController(param)
    y = ctlr.solve(t, y_init)
    omega = y[0,:]
    ierror = y[1,:]
    heading = integrate.cumulative_trapezoid(omega, dx=dt, initial=0.0)
    heading_deg = np.deg2rad(heading)
    xcoord = speed*integrate.cumulative_trapezoid(np.cos(heading_deg), dx=dt, initial=0.0)
    ycoord = speed*integrate.cumulative_trapezoid(np.sin(heading_deg), dx=dt, initial=0.0)

    sim_data[igain] = {
            'color'   : color,
            't'       : t, 
            'omega'   : y[0,:],
            'ierror'  : y[1,:],
            'heading' : heading, 
            'fbscale' : fbscale_vals, 
            'xcoord'  : xcoord, 
            'ycoord'  : ycoord,
            }

fig, ax = plt.subplots(3, 1,num=1)
for igain, data in sim_data.items():
    color = data['color']
    t = data['t']
    omega = data['omega']
    heading = data['heading']
    fbscale_vals = data['fbscale']
    ax[0].plot(t, omega, color)
    ax[1].plot(t, heading, color)
ax[0].grid(True)
ax[0].set_ylabel(r'$\omega$ (deg/s)')
ax[1].grid(True)
ax[1].set_ylabel('heading (deg)')
ax[2].plot(t, fbscale_vals)
ax[2].grid(True)
ax[2].set_ylabel('fbscale)')
ax[2].set_xlabel('t (s)')


fig, ax = plt.subplots(1, 1, num=2)
for igain, data in sim_data.items():
    color = data['color']
    t = data['t']
    xcoord = data['xcoord']
    ycoord = data['ycoord']
    ax.plot(xcoord, ycoord, color)
ax.grid(True)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title(f'Flight Trajectories, speed={speed:1.2f}(m/s)')
ax.axis('equal')
plt.show()
