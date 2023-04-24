import csv

print('Importing libraries ... please wait ...')
import matplotlib as mpl
# mpl.use('Qt5Agg')
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
data = None
data_sub = None
h0 = None
h1 = None
zs = ZScaleInterval()
eff_gain = 0.0
error = None
bg_sub = False


def open_file(filename):
    global data, h0, h1, eff_gain, bg_sub

    if not filename.endswith('.fits'):
        print('WARNING : The file {} does not have a .fits extension, are you sure you want to open this file ?'.format(
            filename))

    try:
        f = fits.open(filename)
    except FileNotFoundError as e:
        print('File not found {}'.format(filename))
        return None
    except:
        print('Something wrong happened while opening {}. Aborting.'.format(filename))
        return None

    if len(f) != 4:
        print('WARNING : The file {} does not have 4 HDUlists, are you sure you are opening the correct file ?'.format(
            filename))
    if len(f) < 2:
        print('ERROR : No curated image found in the file. Aborting')
        return None

    bg_sub = False

    data = f[1].data  # Using the curated image
    h0 = f[0].header
    h1 = f[1].header

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


nx, ny = 0, 0  # Forced to include this in onclick


def find_center(x, y, cepheid_no=1, mod_fit_size=10, plot=True, contour=True):
    if not bg_sub:
        print('ERROR : You have not subtracted background ! Aborting !')
        return

    global nx, ny
    x_min = x - mod_fit_size
    x_max = x + mod_fit_size
    y_min = y - mod_fit_size
    y_max = y + mod_fit_size

    window = data_sub[y_min:y_max + 1, x_min:x_max + 1]

    # Initial guess 
    z0 = data_sub[y, x]
    m_init = models.Moffat2D(z0, x, y)

    manual_pick = False

    # Fitting, we catch warnings as exceptions in case the fit fails
    with warnings.catch_warnings(record=True) as w:
        fit_m = fitting.LevMarLSQFitter()
        xv, yv = np.meshgrid(range(x_min, x_max + 1), range(y_min, y_max + 1))
        p = fit_m(m_init, xv, yv, window)

        if w and issubclass(w[-1].category, AstropyUserWarning):
            print('Warning : The fit might not have converged ! Check fitting parameters !')

            manual_pick = True
            plot = True

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
            ax.set_title('Please click on the centre of the star, cepheid {}'.format(cepheid_no))

        if not (x_min <= nx <= x_max and y_min <= ny <= y_max):
            nx, ny = (x_min + x_max) * 0.5, (y_min + y_max) * 0.5
        pt = ax.scatter(nx, ny, s=5, marker='+', color='red')

        plt.show()
    print('Fitted centre : ', nx, ny)
    return nx, ny


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
    bkg_mean = phot_table['aperture_sum_1'] / annulus_apertures.area
    bkg_sum = bkg_mean * apertures.area
    final_sum = phot_table['aperture_sum_0'] - bkg_sum

    # Calculating zero-point : http://www.stsci.edu/hst/wfpc2/analysis/wfpc2_cookbook.html
    phot_zpt = h1['PHOTZPT']
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

    ferr = phot_table['aperture_sum_err_0'][0] - phot_table['aperture_sum_err_1'][0]  # Is that correct ???
    if flux + ferr <= 0.0 or flux - ferr <= 0.0:
        print('ERROR : Cannot determine errors, star is skipped !')
        return 0.0, 0.0, 0.0, False

    m = -2.5 * np.log10(flux) + zero_pt
    minf = -2.5 * np.log10(flux + ferr) + zero_pt
    msup = -2.5 * np.log10(flux - ferr) + zero_pt

    print('Flux, magnitude, m_inf, m_sup : ', flux, m, minf, msup)
    return flux, m, minf, msup  # why was this returning m, minf, msup, TRUE?


def fit_period(epochs, magnitudes, errors, cepheid, save_figure=False, plot=True, user_check=False):
    epochs = np.asarray(epochs)
    magnitudes = np.asarray(magnitudes)
    errors = np.asarray(errors)

    ids = list(range(epochs.shape[0]))
    ids.sort(key=lambda x: epochs[x])
    epochs = epochs[ids]
    magnitudes = magnitudes[ids]
    errors = errors[ids]

    def simple_fit(x, mu, amplitude, period, phase):
        return mu + amplitude * np.cos(x * 2.0 * np.pi / period + phase)

    def objective(params):
        mu = params['mu'].value
        amp = params['amplitude'].value
        P = params['period'].value
        phi = params['phase'].value

        fy = simple_fit(t, mu, amp, P, phi)
        return (magnitudes - fy) / errors

    model = Model(simple_fit)
    params = model.make_params()

    t = epochs - epochs.min()

    params['mu'] = Parameter(name='mu', value=25.75, min=23, max=28)
    params['amplitude'] = Parameter(name='amplitude', value=0.1, min=0.01, max=10.0)
    params['period'] = Parameter(name='period', value=30.0, min=1.0, max=100.0)
    params['phase'] = Parameter(name='phase', value=0.0, min=-np.pi, max=np.pi)

    result = minimize(objective, params)
    p = result.params
    mu = p['mu'].value
    A = p['amplitude'].value
    P = p['period'].value
    phi = p['phase'].value
    best_fit = simple_fit(t, mu, A, P, phi)
    smooth_t = np.linspace(t.min(), t.max(), 1000)
    smooth_y = simple_fit(smooth_t, mu, A, P, phi)

    print('Estimated period : {} +/- {}'.format(P, p['period'].stderr))
    print('Estimated mean luminosity : {} +/- {}'.format(mu, p['mu'].stderr))

    if plot:
        plt.title('Graph showing the period of Cepheid {}'.format(cepheid))
        plt.errorbar(t, magnitudes, fmt='ob', yerr=errors, capsize=3, elinewidth=0.5, label='observations')
        plt.plot(t, best_fit, 'r-', label='fit')
        plt.plot(smooth_t, smooth_y, 'r--', label='smoothed fit')
        plt.xlabel(r'Epoch $t - t_0$ [MJD]')
        plt.ylabel(r'Apparent magnitude')
        plt.gca().invert_yaxis()
        plt.legend()
        if save_figure:
            plt.savefig('{}_period_fit.png'.format(cepheid))
        plt.show()
        if user_check:
            response = input("Is this a cepheid ? (y/n)")
            if not response == 'y':
                return P, p['period'].stderr, mu, p['mu'].stderr, False
            return P, p['period'].stderr, mu, p['mu'].stderr, True

    return P, p['period'].stderr, mu, p['mu'].stderr, True


def fit_PL(filename, n_samples=100):
    try:
        data = np.loadtxt(filename, delimiter=',')
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

    nd = np.stack((data[:, 0], data[:, 2])).T

    # Random sampling to help with the fitting, assuming normal distribution of the points
    for i, line in enumerate(data):
        rand = np.random.randn(n_samples, 2)
        rand[:, 0] = rand[:, 0] * line[1] + line[0]
        rand[:, 1] = rand[:, 1] * line[3] + line[2]

        mask = rand[:, 0] > 0.0
        R = rand[mask]

        nd = np.concatenate((nd, R))

    plt.errorbar(data[:, 0], data[:, 2], xerr=data[:, 1], yerr=data[:, 3], fmt='ob', capsize=3, elinewidth=0.5,
                 label='Observations')
    plt.scatter(nd[:, 0], nd[:, 1], s=1, color='grey', label='Sampling')

    a = -2.76
    PL = lambda P, b: a * (np.log10(P) - 1.0) + b
    p0 = (-4.16)  # LMC value

    popt, pcov = curve_fit(PL, nd[:, 0], nd[:, 1], p0=p0)
    b = popt[0]
    print('Fitted intersect = ', b)
    print('Error : ', np.sqrt(pcov[0]))

    x = (nd[:, 0].min(), nd[:, 0].max())
    y = (PL(x[0], b), PL(x[1], b))

    plt.plot(x, y, '--k', label='Model')
    plt.legend()
    plt.xlabel('Period [d]')
    plt.ylabel('Apparent magnitude')
    plt.gca().invert_yaxis()
    plt.title('Fitted PL relation, with a={:.5f} and b={:.5f}'.format(a, b))
    plt.show()

    return a, b


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
import simplejson as json

user_check = True  # if True, the user will be asked to check if the fit is correct
get_data = True  # if True, the data will be obtained from json file
plot = True  # if True, the fit will be plotted
save_figure = True  # if True, the figure will be saved

background_subtraction_wsize = 50  # bg_wsize to be used for background subtraction
mod_fits_size = 50  # mod_fits_size to be used for the modified fits

# list of cepheids positions to pass
list_of_cephids_x = [423.712, 595.427, 562.789, 418.865, 390.749,
                     372.335, 339.137, 323.487, 260.109, 164.974, 125.047]

list_of_cephids_y = [624.758, 251.897, 258.963, 242.069, 346.961,
                     349.782, 323.986, 193.577, 236.518, 260.783, 259.101]

no_of_fits = 8  # number of fits (epochs) being used


def epochs_to_float_list(string_list):
    float_list = []
    for x in string_list:
        # the first 6 characters are 'epoch' so we remove them
        float_list.append(float(x[6:]))
    return float_list


cephid_dict = {}

# cephid_dict[cephid_number] = {epoch_number: [flux, mag, minf, msup]}

if get_data == True:
    with open("cephid_dict.json", "r") as f:
        try:
            cephid_dict = json.load(f)
            print("loaded")
        except:
            print("failed to load")
else:
    for i in range(1, no_of_fits + 1):
        string = ("/user/HS401/lc01390/Astro/fits{}.fits".format(i))
        open_file(string)
        epoch = get_parameter("EXPEND")
        subtract_background(plot=False, bg_wsize=background_subtraction_wsize)
        for j in range(0, len(list_of_cephids_x)):
            print("Cephid {} Epoch {}".format(j + 1, epoch))
            center_x, center_y = find_center(int(list_of_cephids_x[j]), int(list_of_cephids_y[j]), cepheid_no=j + 1,
                                             mod_fit_size=mod_fits_size, plot=False)
            flux, mag, minf, msup = compute_photometry(center_x, center_y)
            time.sleep(0.1)
            if i == 1:
                cephid_dict['Cephid {}'.format(j + 1)] = {
                    "Epoch {}".format(epoch): {"X": center_x, "Y": center_y, "FLUX": flux, "MAG": mag, "MINF": minf,
                                               "MSUP": msup}}
            else:
                cephid_dict['Cephid {}'.format(j + 1)]["Epoch {}".format(epoch)] = {"X": center_x, "Y": center_y,
                                                                                    "FLUX": flux, "MAG": mag,
                                                                                    "MINF": minf, "MSUP": msup}
        print(cephid_dict)
        print(get_parameter("EXPEND"))
        time.sleep(1)

with open('cephid_dict.json', 'w') as fp:
    try:
        json.dump(cephid_dict, fp)
        print('Saved')
    except:
        print('Failed to save')

print(cephid_dict)

# fit period
cephid_period_dict = {}
list_cephid_NaN = []
discovered_cephid_dict = {}

# for every cephid we get the epochs and then the mags and mag_errs of that epoch and put each in a list
# this is so we can fit the period as it uses lists
# leads to a list of epochs, a list of mags and a list of mag_errs
for i in range(1, len(cephid_dict) + 1):
    list_of_epochs = epochs_to_float_list(list(cephid_dict['Cephid {}'.format(i)].keys()))
    print(list_of_epochs)
    list_of_mags = []
    list_of_mag_errs = []
    for j in cephid_dict['Cephid {}'.format(
            i)].keys():  # TODO: try using list_of_epochs instead of cephid_dict['Cephid {}'.format(i)].keys()
        count = 0
        list_of_mags.append(cephid_dict['Cephid {}'.format(i)][j]['MAG'])
        list_of_mag_errs.append(
            cephid_dict['Cephid {}'.format(i)][j]['MINF'] - cephid_dict['Cephid {}'.format(i)][j]['MSUP'])

        # if the last value is 0, remove it (it's a NaN and can't be used in the fit)
        if (list_of_mags[-1] == 0) or (list_of_mag_errs[-1] == 0):
            list_of_mags.pop()
            list_of_mag_errs.pop()
            list_of_epochs.pop(count)
        count += 1

    # we now try to fit the period for each cephid
    try:
        # print(
        #     "The average mag is {} \nand the average error in mag is {}".format(np.mean(list_of_mags),
        #                                                                                       np.mean(
        #                                                                                           list_of_mag_errs)))
        period, period_err, mu, mu_error, response = fit_period(list_of_epochs, list_of_mags, list_of_mag_errs, i,
                                                                save_figure=save_figure, plot=plot,
                                                                user_check=user_check)

        # from the response we can see if the fit was good or not based on the user input
        if response == True:
            discovered_cephid_dict['Cephid {}'.format(i)] = {"Period": period, "Period error": period_err, "mu": mu,
                                                             "mu error": mu_error}
        # cephid_period_dict['Cephid {}'.format(i)] = {"Period": period, "Period error": period_err, "mu": mu, "mu error": mu_error}
    except ValueError:

        print(''' 
        Warning!
        NaN values in data for cepheid {}
        
        '''.format(i))
        print("Cephid {}\n Epochs: {}\n mags: {}\n mag_errors: {}".format(i, list_of_epochs, list_of_mags,
                                                                          list_of_mag_errs))
        list_cephid_NaN.append(i)

# print(list_cephid_NaN)
# check for large errors in the period and mu
# for i in discovered_cephid_dict:
#     if (discovered_cephid_dict[i]['Period error'] > 0.5) or (cephid_period_dict[i]['mu error'] > 0.5):
#         print("{} has a large error in period or mu" .format(i))
#         print("Removing from discovered_cephid_dict")
#         discovered_cephid_dict.pop(i)

# store data in a csv file for handing to fit_PL
with open('cephid_period_dict2.txt', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for value in discovered_cephid_dict.values():
        writer.writerow(value.values())

a,b = fit_PL("cephid_period_dict2.txt")
print("a = {} and b = {}".format(a,b))
lmc_a = -2.76
lmc_b = -4.16

def get_magnitude_using_PL(a,b,period):
    m = a*np.log10(period-1) + b
    return m

dis_mod_list = []
for i in discovered_cephid_dict:
    print("The period of {} is {} and the magnitude from the lmc relation is {}".format(i, discovered_cephid_dict[i]['Period'], get_magnitude_using_PL(lmc_a,lmc_b,discovered_cephid_dict[i]['Period'])))
    print("The period of {} is {} and the magnitude from the fit is {}".format(i, discovered_cephid_dict[i]['Period'], get_magnitude_using_PL(a,b,discovered_cephid_dict[i]['Period'])))
    dis_mod_list.append(get_magnitude_using_PL(a,b,discovered_cephid_dict[i]['Period']) - get_magnitude_using_PL(lmc_a,lmc_b,discovered_cephid_dict[i]['Period']))

print("The average distance modulus is {}".format(np.mean(dis_mod_list)))

extinction_term = 0.25
print("The distance in megaparsecs is {}".format((10**((np.mean(dis_mod_list) - extinction_term)/5 + 1))*10**-6))

#save data
with open("Final_data.txt", 'w') as f:
    f.write("The average distance modulus is {}\n".format(np.mean(dis_mod_list)))
    f.write("The distance in megaparsecs is {}".format((10**((np.mean(dis_mod_list) - extinction_term)/5 + 1))*10**-6))
    string = ("\nUsing data:\n a = {} and b = {}".format(a,b) + "\nlmc_a = {} and lmc_b = {}".format(lmc_a,lmc_b) +
              "\nextinction_term = {}".format(extinction_term) + "\ndiscovered_cephid_dict = {}".format(discovered_cephid_dict))
    f.write(string)

