# Anthony Catanese

"""
This program reads a text file containing neutrino energies and reconstructed energies and infers
the probability density function of these parameters.
"""

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import emcee
import corner
import time
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.special
from scipy.stats import distributions
from scipy.stats import pareto
import gc
import sys

import psutil
from kde.pykde import gaussian_kde, bootstrap_kde
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pylab as plt
from pyqt_fit import kde
import time
from scipy.stats import pareto
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import pareto

startTime = time.time()

lnormal = scipy.stats.norm.logpdf

# construct a save figure object
pp = PdfPages('skytrino4.3.pdf')
time_log = []

# read in data from file
data = np.loadtxt("Energy_data.txt", unpack=True)
Enu, Erec, weights = data

'''
# Histogram the complete data set
x_bin_val = np.linspace(0, 100, 100)
xlim = [(0, 40), (0, 40), (0, 5)]
# Histogram the data
for i in range(len(data)):
    plt.hist(data[i, :], bins=x_bin_val, normed=True, color="k", histtype="step")
    plt.xlim(xlim[i])
    plt.title("Dimension {0:d}".format(i))
    plt.show()
    plt.savefig(pp, format='pdf')
    plt.clf()

plt.clf()
'''

# create a log data set
idx = np.where(Erec > 0.0)[0]
Enu_log = np.log10(np.take(Enu, idx))
Erec_log = np.log10(np.take(Erec, idx))

# trim the non-log data down to size
weights = np.take(weights, idx)
Enu = np.take(Enu, idx)

'''
Construct the function
f(Erec|Enu) = f(Erec, Enu|alpha) / f(Enu|alpha)
'''
# weights don't actually matter for this calculation
a = 1.0
wt = pareto.pdf(Enu, a) / weights

# 1D KDE to compute the denominator f(Enu|alpha)
kernel1D_log = kde.KDE1D(Enu_log, weights=wt, bw_method=0.04, adaptive=True, weight_adaptive_bw=True, alpha=0.3)

# 2D KDE to compute the numerator f(Erec, Enu|alpha)
points = np.vstack([Erec_log, Enu_log])

kernel2D_log = gaussian_kde(points, weights=wt, bw_method=0.06,
                            adaptive=True, weight_adaptive_bw=True, alpha=0.3)


# funtion to return the entire prefactor term f(Erec|Enu) evaluated at E_reconstructed and Enu_proposed
def prefactor(E_reconstruted, Enu_proposed):
    v = np.vstack([E_reconstruted, Enu_proposed])
    return kernel2D_log.evaluate(np.log10(v), adaptive=True) / kernel1D_log(np.log10(Enu_proposed))

# test plot the prefactor conditioned on Enu_proposed
def plot_prefactor(Enu_proposed):
    Erec = np.linspace(0.01, 100.0, 500)
    Enu_proposed_array = np.empty(len(Erec))
    Enu_proposed_array.fill(Enu_proposed)
    plt.plot(np.log10(Erec), prefactor(Erec, Enu_proposed_array))
    plt.xlim(-1.0, 2.5)
    plt.ylim(0.0, 6.0)
    plt.show()


'''
Specify the reconstructed neutrino energy observed
'''
E_reconstruted = 30.0

# prior on alpha
def lnprior(alpha):
    return lnormal(alpha, 1.0, 0.2)


# construct the model from which to draw samples
def model(param):
    # unpack the param vector
    Enu_proposed, alpha_proposed = param

    # plot the prefactor conditioned on Enu_proposed
    #print(Enu_proposed)
    #plot_prefactor(Enu_proposed)

    # regenerate f(Enu|alpha) at the proposed alpha
    wt = pareto.pdf(Enu, alpha_proposed) / weights

    kernel1D_log = kde.KDE1D(Enu_log, weights=wt, bw_method=0.04, adaptive=True, weight_adaptive_bw=True,
                                   alpha=0.3)
    return prefactor(E_reconstruted, Enu_proposed) * kernel1D_log.evaluate(np.log10(Enu_proposed))


# return the log probability
def lnprob(param):
    time_log.append(time.time())
    # print(time_log)
    Enu_proposed, alpha_proposed = param

    if alpha_proposed < 0.0:
        return -np.inf
    if Enu_proposed < 1.0:
        return -np.inf

    prob = model(param)

    if not np.isfinite(prob):
        return -np.inf
    if prob <= 0.0:
        return -np.inf

    return np.log(prob) + lnprior(alpha_proposed)


##Main##

# Set up the sampler.
ndim = 2  # dimension
nwalkers = 250  # walkers
burn = 100  # Number of samples to discard
run = 500  # Number of samples to keep

# create a vector of starting positions for the walkers
v0 = np.array([30.0, 1.0])

# Random initial set of positions for the walkers.
p0 = [(v0 + np.random.rand(ndim)) for i in range(nwalkers)]

# Create a sampler object
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

# Run 100 steps as a burn-in.
# Record the starting time
print("Running burn in...")
t1 = time.time()
pos, prob, state = sampler.run_mcmc(p0, burn)
print("Burn in complete...")

# Reset the chain to remove the burn-in samples.
sampler.reset()

# Clear and run the production chain.
print("Running MCMC...")
sampler.run_mcmc(pos, run, rstate0=np.random.get_state())
m, s = divmod((time.time() - t1), 60)
print("Done.")
print("Total time: ", m, " min, ", s, " sec")

# Print the mean acceptance fraction and the autocorrelation time
print("Mean acceptance fraction: {0:.3f}"
      .format(np.mean(sampler.acceptance_fraction)))
print("Autocorrelation time:", sampler.get_autocorr_time())

# the chain attribute is an array witn shape (num_walkers, simulation steps, num_param)
fig, axes = plt.subplots(ndim, 1, sharex=True, figsize=(8, 9))
for i in range(ndim):
    axes[i].plot(sampler.chain[:, :, i].T, color="k", alpha=0.4)
    axes[i].set_title("Dimension {0:d}".format(i))

plt.savefig(pp, format='pdf')
plt.clf()
# plt.show()

# plot a histogram of the marginal distributions
# The EnsembleSampler.flatchain is an array with shape(num_walkers * num_steps, num_parameters)
# ******************
# Set the x-axis limits for each dimension
xlim = [(0, 100), (0, 5)]
# select the x-value for each bin
x_bin_val = [np.linspace(0, 500, 500), np.linspace(0, 5, 75)]
# Integration limits for cubic spline
# integral_bounds = [(0, 40), (-5, 40), (0, 5)]
# set the label for the x-axis
x_axis_label = ["Neutrino Energy", "alpha"]

# ******************

# plot a histogram of the trace
for i in range(ndim):
    array = sampler.flatchain[:, i]
    plt.xlim(xlim[i])
    plt.xlabel(x_axis_label[i])
    values, bins, _ = plt.hist(array, bins=x_bin_val[i], normed=True, color="k", histtype="step")
    # if i != 2:
    # plt.hist(data[i, :], bins=x_bin_val[i], normed=True, color="r", histtype="step")
    '''
    # fit a spline to the histogram and integrate between (a, b)
    x = bins[:-1] + np.diff(bins / 2.0)
    spl = InterpolatedUnivariateSpline(x, values)
    plt.plot(x, spl(x), 'g', lw=1, alpha=0.7)
    a, b = integral_bounds[i]
    print("Dimension:" + str(i) + " " + str(spl.integral(a, b)))
    '''

    # diff = np.diff(bins)
    # area = sum(diff * values)
    # print(area)
    plt.title("Dimension {0:d}".format(i))
    plt.savefig(pp, format='pdf')
    # plt.show()
    plt.clf()

# Plot a histogram of the 1st half of the trace to compare to the 2nd half
for i in range(ndim):
    array = [sampler.flatchain[:int(nwalkers * run / 2), i], sampler.flatchain[int(nwalkers * run / 2):, i]]
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 9))
    for j in range(0, 2):
        axes[j].hist(array[j], bins=x_bin_val[i], normed=True, color="k", histtype="step")
        axes[j].set_title("Dimension {0:d}".format(i))
        axes[j].set_xlabel(x_axis_label[i])
        axes[j].set_xlim(xlim[i])
    plt.savefig(pp, format='pdf')
    # plt.show()
    plt.clf()

# Plot a histogram of the 2nd half of the trace overlayed on the 1st half
for i in range(ndim):
    array = [sampler.flatchain[:int(nwalkers * run / 2), i], sampler.flatchain[int(nwalkers * run / 2):, i]]
    plt.xlim(xlim[i])
    plt.xlabel(x_axis_label[i])
    values, bins, _ = plt.hist(array[0], bins=x_bin_val[i], normed=True, color="k", histtype="step")
    values, bins, _ = plt.hist(array[1], bins=x_bin_val[i], normed=True, color="g", histtype="step")
    plt.savefig(pp, format='pdf')
    # plt.show()
    plt.clf()

# flatten the chain so that we have a flat list of samples
samples = sampler.chain[:, :, :].reshape((-1, ndim))

# append the sample chain to a text file

if len(sys.argv) == 1:
    with open("skytrino4.1.txt", 'ab') as file:
        np.savetxt(file, samples)
else:
    with open(sys.argv[1], 'ab') as file:
        np.savetxt(file, samples)

# np.savetxt("MultiMode2Samples3.txt", samples)

# Generate a corner plot of the posterior
fig = corner.corner(samples, labels=x_axis_label)
plt.savefig(pp, format='pdf')
plt.clf()
# plt.show()

# print the total number of iterations and calculate the average
print("the number of iterations is: ")
print(len(time_log))
time_diff = [time_log[i + 1] - time_log[i] for i in range(len(time_log) - 1)]
ave = sum(time_diff) / len(time_diff)
print("The average iteration time is: ")
print(ave)

# Plot the autocorrelation function
autocorr_func = emcee.autocorr.function(samples)
x = np.linspace(0, 500, 500, endpoint=False)
for i in range(ndim):
    plt.plot(x, autocorr_func[:500, i])
    plt.savefig(pp, format='pdf')
    # plt.show()
    plt.clf()

pp.close()

endTime = time.time()
totalTime = endTime - startTime
TT = np.array([totalTime])

if len(sys.argv) > 1:
    with open(sys.argv[1] + ".log", 'ab') as file:
        np.savetxt(file, TT)
