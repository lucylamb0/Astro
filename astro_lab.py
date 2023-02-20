print('Importing libraries ... please wait ...')
import matplotlib as mpl
mpl.use('Qt5Agg')
import time

from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.modeling import models, fitting
from astropy.utils.exceptions import AstropyUserWarning
from astropy.visualization import ZScaleInterval
from photutils import aperture_photometry, Background2D, MedianBackground, CircularAnnulus, CircularAperture
from photutils.utils import calc_total_error
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp2d, RegularGridInterpolator
from scipy.optimize import curve_fit
from lmfit import Model, Parameter, minimize

import matplotlib.pyplot as plt
import numpy as np
import os, sys, warnings

# Objects
data     = None
data_sub = None
h0       = None
h1       = None
zs       = ZScaleInterval()
eff_gain = 0.0
error    = None
bg_sub   = False

def open_file(filename):
    global data, h0, h1, eff_gain, bg_sub
    
    if not filename.endswith('.fits'):
        print('WARNING : The file {} does not have a .fits extension, are you sure you want to open this file ?'.format(filename))

    try:
        f = fits.open(filename)
    except FileNotFoundError as e:
        print('File not found {}'.format(filename))
        return None
    except:
        print('Something wrong happened while opening {}. Aborting.'.format(filename))
        return None
        
    if len(f) != 4:
        print('WARNING : The file {} does not have 4 HDUlists, are you sure you are opening the correct file ?'.format(filename))
    if len(f) < 2:
        print('ERROR : No curated image found in the file. Aborting')
        return None

    bg_sub = False

    data = f[1].data # Using the curated image
    h0   = f[0].header
    h1   = f[1].header

    eff_gain = h0['ATODGAIN']
    print('File {} opened'.format(filename))

def get_parameter(param_name):
    if param_name in h0:
        return h0[param_name]
    elif param_name in h1:
        return h1[param_name]
    else:
        print('WARNING : parameter {} is not in the files headers !'.format(param_name))
        return None

def plot_data(zscale=True):
    if bg_sub:
        pdata = data_sub
    else:
        pdata = data
        
    if zscale:
        lims = zs.get_limits(pdata)
    else:
        lims = [pdata.min(), pdata.max()]
        
    plt.imshow(pdata, cmap='Greys_r', vmin=lims[0], vmax=lims[1], origin='lower')
    plt.show()

def subtract_background(bg_wsize=50, sclip=3.0, plot=False):
    global data, error, data_sub, bg_sub
    sigma_clip = SigmaClip(sigma=sclip)
    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (bg_wsize, bg_wsize), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    if plot:
        plt.imshow(bkg.background, origin='lower', cmap='Greys_r')
        plt.show()
    error = calc_total_error(data, bkg.background_rms, eff_gain)
    data_sub = data - bkg.background
    bg_sub = True
    print('Background successfully subtracted')
    
nx, ny = 0, 0 # Forced to include this in onclick
def find_center(x, y, mod_fit_size=10, plot=True, contour=True):
    if not bg_sub:
        print('ERROR : You have not subtracted background ! Aborting !')
        return
    
    global nx, ny
    x_min  = x - mod_fit_size
    x_max  = x + mod_fit_size
    y_min  = y - mod_fit_size
    y_max  = y + mod_fit_size

    window = data_sub[y_min:y_max+1, x_min:x_max+1]

    # Initial guess 
    z0     = data_sub[y, x]
    m_init = models.Moffat2D(z0, x, y)

    manual_pick = False

    # Fitting, we catch warnings as exceptions in case the fit fails
    with warnings.catch_warnings(record=True) as w:
        fit_m  = fitting.LevMarLSQFitter()
        xv, yv = np.meshgrid(range(x_min, x_max+1), range(y_min, y_max+1))
        p      = fit_m(m_init, xv, yv, window)

        if w and issubclass(w[-1].category, AstropyUserWarning):
            print('Warning : The fit might not have converged ! Check fitting parameters !')

            manual_pick=True
            plot=True

    nx, ny = p.x_0.value, p.y_0.value

    if plot:
        # For manual picking
        def onclick(event):
            global nx, ny
            if event.button != 1:
                return
            
            nx = event.xdata
            ny = event.ydata

            pt.set_offsets((nx, ny))
            fig.canvas.draw_idle()
            print('Centre set to {} {}'.format(nx, ny))
            
        lims = zs.get_limits(window)
        fig, ax = plt.subplots()
        
        ax.imshow(window, origin='lower', vmin=lims[0], vmax=lims[1], extent=(x_min, x_max, y_min, y_max))

        if manual_pick:
            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            ax.set_title('Please click on the centre of the star')

        if not (x_min <= nx <= x_max and y_min <= ny <= y_max):
            nx, ny = (x_min+x_max)*0.5, (y_min+y_max)*0.5
        pt = ax.scatter(nx, ny, s=5, marker='+', color='red')
            
        plt.show()
    print('Fitted centre : ', nx, ny)

def compute_photometry(x, y, aperture_r=3.0, sky_in=6.0, sky_out=8.0):
    print('Computing photometry at : ', x, y)
    # Aperture photometry : https://photutils.readthedocs.io/en/stable/aperture.html
    pos = [(x, y)]
    apertures = CircularAperture(pos, r=aperture_r)
    annulus_apertures = CircularAnnulus(pos, r_in=sky_in, r_out=sky_out)
    apers = [apertures, annulus_apertures]

    if not bg_sub:
        print('WARNING : Background has not been subtracted !')
        print('          Performing photometry measures without error model')
        phot_table = aperture_photometry(data, apers)
    else:
        phot_table = aperture_photometry(data_sub, apers, error)

    # Mean sky subtraction
    bkg_mean  = phot_table['aperture_sum_1'] / annulus_apertures.area
    bkg_sum   = bkg_mean * apertures.area
    final_sum = phot_table['aperture_sum_0'] - bkg_sum

    # Calculating zero-point : http://www.stsci.edu/hst/wfpc2/analysis/wfpc2_cookbook.html
    phot_zpt  = h1['PHOTZPT']
    phot_flam = h1['PHOTFLAM']
    zero_pt = -2.5 * np.log10(phot_flam) + phot_zpt

    # TODO : Correct from STMAG to Johnson,
    # although according to WFPC2-cookbook, the zero point almost match Johnson's V band on f555w
    flux = final_sum[0]
    if flux <= 0.0:
        print('ERROR : the background subtracted flux is negative !')
        print('        no star has been detected here !')
        return 0.0, 0.0, 0.0, False

    if np.isnan(flux):
        print('ERROR : flux impossible to measure ! Skipping star')
        return 0.0, 0.0, 0.0, False

    ferr = phot_table['aperture_sum_err_0'][0] - phot_table['aperture_sum_err_1'][0] # Is that correct ???
    if flux + ferr <= 0.0 or flux - ferr <= 0.0:
        print('ERROR : Cannot determine errors, star is skipped !')
        return 0.0, 0.0, 0.0, False
    
    m    = -2.5 * np.log10(flux) + zero_pt
    minf = -2.5 * np.log10(flux + ferr) + zero_pt
    msup = -2.5 * np.log10(flux - ferr) + zero_pt
    
    print('Flux, magnitude, m_inf, m_sup : ', flux, m, minf, msup)
    return m, minf, msup, True

def fit_period(epochs, magnitudes, errors, plot=True):
    epochs     = np.asarray(epochs)
    magnitudes = np.asarray(magnitudes)
    errors     = np.asarray(errors)

    ids = list(range(epochs.shape[0]))
    ids.sort(key=lambda x:epochs[x])
    epochs = epochs[ids]
    magnitudes = magnitudes[ids]
    errors = errors[ids]
    
    def simple_fit(x, mu, amplitude, period, phase):
        return mu + amplitude*np.cos(x * 2.0 * np.pi / period + phase)

    def objective(params):
        mu  = params['mu'].value
        amp = params['amplitude'].value
        P   = params['period'].value
        phi = params['phase'].value

        fy = simple_fit(t, mu, amp, P, phi)
        return (magnitudes - fy) / errors

    model = Model(simple_fit)
    params = model.make_params()


    t = epochs - epochs.min()

    params['mu']        = Parameter(name='mu',        value=25.75, min=23,     max=28)
    params['amplitude'] = Parameter(name='amplitude', value=0.1,   min=0.01,   max=10.0)
    params['period']    = Parameter(name='period',    value=30.0,  min=1.0,    max=100.0)
    params['phase']     = Parameter(name='phase',     value=0.0,   min=-np.pi, max=np.pi)

    result = minimize(objective, params)
    p = result.params
    mu  = p['mu'].value
    A   = p['amplitude'].value
    P   = p['period'].value
    phi = p['phase'].value
    best_fit = simple_fit(t, mu, A, P, phi)
    smooth_t = np.linspace(t.min(), t.max(), 1000)
    smooth_y = simple_fit(smooth_t, mu, A, P, phi)

    print('Estimated period : {} +/- {}'.format(P, p['period'].stderr))
    print('Estimated mean luminosity : {} +/- {}'.format(mu, p['mu'].stderr))
    
    if plot:
        plt.errorbar(t, magnitudes, fmt='ob', yerr=errors, capsize=3, elinewidth=0.5, label='observations')
        plt.plot(t, best_fit, 'r-', label='fit')
        plt.plot(smooth_t, smooth_y, 'r--', label='smoothed fit')
        plt.xlabel(r'Epoch $t - t_0$ [MJD]')
        plt.ylabel(r'Apparent magnitude')
        plt.gca().invert_yaxis()
        plt.legend()
        plt.show()

def fit_PL(filename, n_samples=100):
    try:
        data = np.loadtxt(filename)
    except:
        print('ERROR : Cannot open this file. Check if it exists and check the format')
        return


    if data.shape[1] != 4:
        print('ERROR : The file should have 4 columns !')
        return

    Nl = data.shape[0]
    if Nl < 3:
        print('WARNING : You are trying to fit the PL relation with only {} data points !'.format(Nl))
        print('          Results might be very bad. Try adding more points !')
        
    nd = np.stack((data[:,0], data[:,2])).T

    # Random sampling to help with the fitting, assuming normal distribution of the points
    for i, line in enumerate(data):
        rand = np.random.randn(n_samples, 2)
        rand[:,0] = rand[:,0] * line[1] + line[0]
        rand[:,1] = rand[:,1] * line[3] + line[2]
        
        mask = rand[:,0] > 0.0
        R = rand[mask]

        nd = np.concatenate((nd, R))
        
    plt.errorbar(data[:,0], data[:,2], xerr=data[:,1], yerr=data[:,3], fmt='ob', capsize=3, elinewidth=0.5, label='Observations')
    plt.scatter(nd[:,0], nd[:,1], s=1, color='grey', label='Sampling')

    a = -2.76
    PL = lambda P, b : a * (np.log10(P) - 1.0) + b
    p0 = (-4.16) # LMC value

    popt, pcov = curve_fit(PL, nd[:,0], nd[:,1], p0=p0)
    b = popt[0]
    print('Fitted intersect = ', b)
    print('Error : ', np.sqrt(pcov[0]))

    x = (nd[:,0].min(), nd[:,0].max())
    y = (PL(x[0], b), PL(x[1], b))

    plt.plot(x, y, '--k', label='Model')
    plt.legend()
    plt.xlabel('Period [d]')
    plt.ylabel('Apparent magnitude')
    plt.gca().invert_yaxis()
    plt.title('Fitted PL relation, with a={:.5f} and b={:.5f}'.format(a, b))
    plt.show()
    

if __name__ == '__main__':
    print('''
================================================
=     Introduction to astronomy (PHY2071)      =
=             University of Surrey             =
= Faculty of Engineering and Physical Sciences =
=              Physics Department              =
=                                              =
=      For any enquiry about the scripts :     =
=           m.delorme@surrey.ac.uk             =
================================================''')

for i in range(1, 12):
    string = ("/user/HS401/lc01390/Astro/fits{}.fits" .format(i))
    open_file(string)
    print(get_parameter("EXPEND"))
    time.sleep(1)
