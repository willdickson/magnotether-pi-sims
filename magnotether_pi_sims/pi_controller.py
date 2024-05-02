import numpy as np
import scipy.integrate as integrate

class PIController:
    
    """
    Implements a simple proportional-integral controller, i.e., a proportional
    controller with a leaky integrator. 

    Note, moment of inertia I is assumed to be = 1.0. 

    err = setpt - y[0]
    dy[0]/dt = -dcoef*y[0] + pgain*fbscale*err + gain*y[1] + bias
    dy[1]/dt = -ikeak*y[1] + fbscale*err

    """

    def __init__(self, param):
        """ Class contructor

        Parameters
        ----------
        param : dict
            dictionary of controller parameters
            param = {
                    'dcoef'   : (float or callable) plant damping coefficient, 
                    'bias'    : (float or callable) plant bias,
                    'pgain'   : (float or callable) controller proportional gain,  
                    'igain'   : (float or callable) controller integral gain, 
                    'ileak'   : (float or callable) controller integrator leak coefficient, 
                    'setpt'   : (float or callable) controller set point,
                    'fbscale'  : (float or callable) feed back scaling in range (0,1), 
                    }

        """
        self.dcoef = param.get('dcoef', 0.0)  # plant damping coefficient
        self.bias = param.get('bias', 0.0)    # plant bias
        self.pgain = param.get('pgain', 0.0)  # controller proportional gain
        self.igain = param.get('igain', 0.0)  # controller integral gain
        self.ileak = param.get('ileak', 0.0)  # controller integrator leak coeff
        self.setpt = param.get('setpt', 0.0)  # controller set-point
        self.fbscale = param.get('fbscale', 0.0) # feedback scaling function


    def state_func(self, t, y):
        """
        Dynamical system state function

        Parameters: 
        -----------
        t : float 
            time in seconds

        y : ndarray 
            1D array with shape=(2,) and float type containing the angular
            velocity y[0] and angular acceleration y[1]. 

        Returns:
        --------
        dy : ndarry
             1D array with shape=(2,) and float type containing the derivatives
             of the angular velocity dy[0] and angular acceleration dy[1]. 

        """
        dcoef = func_or_scalar(self.dcoef, t)
        bias  = func_or_scalar(self.bias,  t)
        pgain = func_or_scalar(self.pgain, t)
        igain = func_or_scalar(self.igain, t)
        ileak = func_or_scalar(self.ileak, t)
        setpt = func_or_scalar(self.setpt, t)
        fbscale = func_or_scalar(self.fbscale, t)

        dy = np.zeros(y.shape)
        err = setpt - y[0]
        dy[0] = -dcoef*y[0] + pgain*fbscale*err + igain*y[1] + bias
        dy[1] = -ileak*y[1] + fbscale*err
        return dy


    def solve(self, t_vals, y_init=None, method='RK45'):
        """
        Solves the LPI controller differential equations for the specified time values. 

        Parameters:
        -----------
        t_vals : ndarray
                 1D array with float type containing the time values at which return
                 the numerical solution of the differential equation. 

        y_init : ndarray, optional
                 1D array with initial condition.  Default all zeros

        method : string, optional
                 a string specifiying ODE solver method, e.g. 'RK45', 'LSODA', etc

        Returns:
        --------
        y_vals : ndarray
                2D array, shape = (2,N), LPI controller solution at specified time points. 

        """
        y_init = np.array([0.0,0.0]) if y_init is None else y_init
        t_min, t_max = t_vals[0], t_vals[-1]
        max_step = t_vals[1] - t_vals[0]
        result = integrate.solve_ivp(
                self.state_func, 
                (t_min, t_max), 
                y_init, 
                t_eval=t_vals, 
                method=method, 
                max_step=max_step,
                )
        y_vals = result.y
        return y_vals 


# -----------------------------------------------------------------------------------------

def func_or_scalar(obj, t):
    return obj(t) if callable(obj) else obj




