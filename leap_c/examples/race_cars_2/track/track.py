from os.path import join
from pathlib import Path
from numpy import array, loadtxt, argmin, sin, cos, arctan2, load, hstack, newaxis
from numpy.linalg import norm
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.interpolate import PPoly

class Track:

    def __init__(self, filename="LMS_Track.txt"):
        track_file = join(str(Path(__file__).parent), "data/"+filename)
        self.__data = loadtxt(track_file)
        self.__spline = CubicSpline(self.thetaref, self.array[:, 1:3], bc_type="periodic")

        if filename == 'inje_kappa_splined.txt':
            dirname = join(str(Path(__file__).parent),"data/smoothing_low")
            c = load(dirname+"/inje.spl.c.npy")
            x = load(dirname+"/inje.spl.x.npy")
            self.__spline = PPoly(c, x)#, extrapolate="periodic")

        cpsi = cos(self.array[:, 3])
        cpsi[-1] = cpsi[0]
        spsi = sin(self.array[:, 3])
        spsi[-1] = spsi[0]
        self.__psi_spline = CubicSpline(
            self.thetaref,
            hstack([cpsi[:, newaxis], spsi[:, newaxis]]),
            bc_type="periodic"
        )

    def __call__(self, theta, *args, **kwargs):
        return self.__spline(theta, *args, **kwargs)

    def __iter__(self):
        return iter(self.__data)

    def __getitem__(self, key):
        return self.__data[key]

    def __setitem__(self, key, val):
        self.__data[key] = val

    def get_theta(self, X, Y, initial_guess=None, eval_ey=False):
        if initial_guess is None:
            initial_guess = self.thetaref[argmin((self.Xref - X)**2 + (self.Yref - Y)**2)]
        if isinstance(initial_guess, int):
            initial_guess = self.thetaref[initial_guess]
        theta = minimize(
            lambda x: self.__dist_sq(x[0], X, Y),
            [initial_guess]
        ).x[0]
        if eval_ey:
            p = self.__spline(theta)
            dp = self.__spline(theta, 1)
            psi = arctan2(dp[1], dp[0])
            ey = -sin(psi) * (X - p[0]) + cos(psi) * (Y - p[1])
            return theta, ey
        return theta

    def __dist_sq(self, theta, X, Y, eval_gradient=True):
        # p = self.__spline(theta)
        # d = (p[0] - X)**2 - (p[1] - Y)**2
        # if eval_gradient:
        #     dp = self.__spline(theta, 1)
        #     return d, 2 * (dp[0] * (p[0] - X) + dp[1] * (p[1] - Y))
        
        p = self.__spline(theta)
        dp = self.__spline(theta, 1)
        psi = arctan2(dp[1], dp[0])
        el = cos(psi) * (X - p[0]) + sin(psi) * (Y - p[1])

        return el**2

    def get_XY(self, theta, *args, **kwargs):
        return self.__spline(theta, *args, **kwargs)
    
    def get_psi(self, theta, *args, radian=False, **kwargs):
        if radian:
            cspsi = self.__psi_spline(theta, *args, **kwargs)
            return arctan2(cspsi[:, 1], cspsi[:, 0])
        else:
            return self.__psi_spline(theta, *args, **kwargs)

    @property
    def array(self):
        return self.__data

    @property
    def thetaref(self):
        return self.__data[:, 0]

    @property
    def Xref(self):
        return self.__data[:, 1]

    @property
    def Yref(self):
        return self.__data[:, 2]

    @property
    def psiref(self):
        return self.__data[:, 3]
    
    @property
    def cpsiref(self):
        return cos(self.__data[:, 3])

    @property
    def spsiref(self):
        return sin(self.__data[:, 3])

    @property
    def kapparef(self):
        return self.__data[:, 4]

    @property
    def border_left(self):
        return self.__data[:, 5]

    @property
    def border_right(self):
        return self.__data[:, 6]
    
    @property
    def cpsiref(self):
        return self.__data[:, 7]
    
    @property
    def spsiref(self):
        return self.__data[:, 8]