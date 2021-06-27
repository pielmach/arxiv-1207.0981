#!/usr/bin/env python

"""
author:       Michael Prim
contact:      michael.prim@kit.edu
date:         2012-05-31
version:      1.1
description:  CAT - A correlation analysis tool

---
Copyright 2012 Michael Prim

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Dieses Programm ist Freie Software: Sie koennen es unter den Bedingungen
der GNU General Public License, wie von der Free Software Foundation,
Version 3 der Lizenz oder (nach Ihrer Option) jeder spaeteren
veroeffentlichten Version, weiterverbreiten und/oder modifizieren.

Dieses Programm wird in der Hoffnung, dass es nuetzlich sein wird, aber
OHNE JEDE GEWAEHRLEISTUNG, bereitgestellt; sogar ohne die implizite
Gewaehrleistung der MARKTFAEHIGKEIT oder EIGNUNG FUER EINEN BESTIMMTEN ZWECK.
Siehe die GNU General Public License fuer weitere Details.

Sie sollten eine Kopie der GNU General Public License zusammen mit diesem
Programm erhalten haben. Wenn nicht, siehe <http://www.gnu.org/licenses/>. 
"""

import sys, os, csv
import math
import datetime
from optparse import OptionParser
import numpy as np
from matplotlib import rc
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.stats as stats
import scipy.version as spversion
import matplotlib as mpl
import uuid


class histogram:
    """
    A simple 1-dim histogram class that takes bin center values, contents and errors.
    If no errors are given it will calculate binomial errors for each bin.
    
    The error list is twice as long as the values and bins list, as it stores first upper, then lower errors.
    
    Dividing a histogram by another histogram returns a new histogram with proper errors.    
    """

    def __init__(self, xmin, xmax, bins, values, errors = None):
        if(len(bins) != len(values)):
            raise Exception("# bins doesn't equal # values")
        else:
            self.__nbins = len(bins)
        self.__xmin = xmin
        self.__xmax = xmax
        self.__bin_centers = bins
        self.__bin_contents = values
        self.__bin_errors = errors
        if(errors == None):
            #self.__calc_poisson_errors()
            self.__calc_poisson_conf_interval()
        elif(len(errors) * len(errors[-1]) != 2 * len(bins)):
            raise Exception("2x # bins doesn't equal # errors")

    def __calc_poisson_errors(self):
        self.__bin_errors = np.zeros((2, self.__nbins), dtype = 'd')
        for c in xrange(0, self.__nbins):
            if(self.__bin_contents[c] > 0):
                self.__bin_errors[0][c] = math.sqrt(self.__bin_contents[c])
                self.__bin_errors[1][c] = math.sqrt(self.__bin_contents[c])
            else:
                self.__bin_errors[0][c] = 0
                self.__bin_errors[1][c] = 0

    def __calc_poisson_conf_interval(self):
        self.__bin_errors = np.zeros((2, self.__nbins), dtype = 'd')
        for c in xrange(0, self.__nbins):
            if(self.__bin_contents[c] >= 0):
                self.__bin_errors[0][c], self.__bin_errors[1][c] = get_poisson_conf_interval(self.__bin_contents[c])

    def __repr__(self):
        return "Histogram with " + str(self.__nbins) + " bins from " + str(self.__xmin) + " to " + str(self.__xmax) + "\n" + str(self.__bin_centers) + "\n" + str(self.__bin_contents) + "\n" + str(self.__bin_errors)

    def get_bin_centers(self):
        return self.__bin_centers

    def get_bin_contents(self):
        return self.__bin_contents

    def get_bin_errors(self):
        return self.__bin_errors

    def get_x_min_max(self):
        return self.__xmin, self.__xmax

    def get_pull_distribution(self, hist):
        if self.__nbins != hist.__nbins:
            raise Exception("Histograms do not have same number of bins, can't calc pull distribution")
        if self.get_x_min_max() != hist.get_x_min_max():
            raise Exception("Histograms do not have same borders, can't calc pull distribution")

        data_new = np.zeros(self.__nbins, dtype = 'd')
        for i in xrange(0, self.__nbins):
            # assume that average of both hists is good estimator of truth
            avg = (self.__bin_contents[i] + hist.get_bin_contents()[i]) / 2
            if(avg != 0 and self.__bin_errors[0][i] > 0):
                data_new[i] = self.__bin_contents[i] - avg
                # estimate error from upper (larger) error
                data_new[i] /= self.__bin_errors[0][i]
            else:
                data_new[i] = 0

        error_new = np.zeros((2, self.__nbins), dtype = 'd')

        return histogram(self.__xmin, self.__xmax, self.get_bin_centers(), data_new, error_new)


cat_poisson_conf_interval_dict = {}
cat_poisson_conf_interval_dict[0] = (0, 1.84)
def get_poisson_conf_interval(n):
    """
    Return tuple with (lower,upper) 1-sigma poisson confidence interval
    """
    if(n in cat_poisson_conf_interval_dict):
        return cat_poisson_conf_interval_dict[n]
    else:
        if n < 0:
            raise Exception("Calculation of Poisson convidence interval for bin content < 0 not possible")

        cat_poisson_conf_interval_dict[n] = (n - stats.chi2.ppf(0.1587, 2 * n) / 2), (stats.chi2.ppf(0.8413, 2 * (n + 1)) / 2 - n)
        return cat_poisson_conf_interval_dict[n]


def get_nbins_for_plot_matrix(data, verbose):
    """
    Returns number ob rows/coloums in plot matrix based on the size of the given data sample.
    
    A minimum of 3 bins is used and a maximum of 10 for high stat samples.
    """
    data_length = len(data[0])
    nbins = 3
    while(data_length / nbins > 5000):
        nbins += 1
        if(nbins == 10):
            break

    if(verbose):
        print "Data points: " + str(data_length)
        print "Bins used for analysis: " + str(nbins)
    return nbins


def get_bins_and_content(vmin, vmax, nbins, datapoints):
    """
    For a given list of datapoints it creates the necessary inputs to create
    a histogram from with nbins in [vmin,vmax] interval.
    """
    step = float(vmax - vmin) / nbins

    bins = np.zeros(nbins, dtype = 'd')
    values = np.zeros(nbins, dtype = 'd')

    for i in xrange(0, len(datapoints)):
        binnr = int((datapoints[i] - vmin) / step)
        binnr = min(binnr, nbins - 1) #overflow values in last bin
        binnr = max(binnr, 0) # underflow values in first bin
        values[binnr] += 1

    for i in xrange(0, len(bins)):
        bins[i] = vmin + step / 2 + i * step

    return vmin, vmax, bins, values, None


def get_patches(x, y, (xmin, xmax), patchcolor):
    """
    Creates a list of patches with given patchcolor.
    
    Requires xmin, xmax of histogram and x,y values. Assumes equidistant binning.
    """
    bins = len(x)
    binwidth_half = (xmax - xmin) / float(bins) / 2
    patch_list = []
    for i in xrange(0, len(x)):
        ll = (x[i] - binwidth_half, 0)
        ul = (x[i] - binwidth_half, y[i])
        ur = (x[i] + binwidth_half, y[i])
        lr = (x[i] + binwidth_half, 0)

        verts = [ll] + [ul] + [ur] + [lr]

        poly = patches.Polygon(verts, facecolor = patchcolor, edgecolor = 'none', alpha = 1.0)
        patch_list.append(poly)
    return patch_list


def plot_histogram_to_axes(hist, ax, patchcolor = None):
    """
    Plots histogram to ax and returns the histogram, adds patches if a patchcolor is given.
    """
    ax.set_xlim(hist.get_x_min_max())
    if(patchcolor != None):
        for i in get_patches(hist.get_bin_centers(), hist.get_bin_contents(), ax.get_xlim(), patchcolor):
            ax.add_patch(i)
    ax.errorbar(x = hist.get_bin_centers(), y = hist.get_bin_contents(),
                yerr = hist.get_bin_errors(), xerr = [(hist.get_bin_centers()[0] - hist.get_bin_centers()[1]) / 2 for i in xrange(0, len(hist.get_bin_centers()))],
                linestyle = 'None', marker = 'None', capsize = 0, elinewidth = 0.5, ecolor = 'black')
    ax.set_xlim(hist.get_x_min_max())
    return hist


def plot_profile_plot_to_axes(x, y, (xmin, xmax), (ymin, ymax), nbins, ax):
    """
    Plots profile plot of y in nbins of x with limits (xmin, xmax) and (ymin, ymax) to ax
    """

    def calc_mean_and_error(values):
        """
        Helper method to calc mean and error of given list of values
        """
        if(len(values) == 0):
            return 0, 0
        elif(len(values) == 1):
            return values[0], 2e20
        else:
            summe = sum(values)
            mean = summe / len(values)
            error = 0
            for i in values:
                error += math.pow((i - mean), 2)
            error /= len(values) - 1
            error = math.sqrt(error / len(values))
            return mean, error

    data_new = (x, y)
    values = []
    for i in xrange(0, nbins):
        values.append([])
    mean = np.zeros(nbins, dtype = 'd')
    error = np.zeros(nbins, dtype = 'd')

    step = float(xmax - xmin) / nbins
    for i in xrange(0, len(data_new[0])):
        binnr = int((data_new[0][i] - xmin) / step)
        binnr = min(binnr, nbins - 1)
        values[binnr].append(data_new[1][i])

    for i in xrange(0, len(values)):
        mean[i], error[i] = calc_mean_and_error(np.array(values[i], dtype = 'd'))

    bins = np.zeros(nbins, dtype = 'd')
    binerrors = np.zeros(nbins, dtype = 'd')
    for i in xrange(0, nbins):
        bins[i] = xmin + step / 2 + i * step
        binerrors[i] = step / 2

    ax.errorbar(x = bins, y = mean, xerr = binerrors, yerr = error,
                linestyle = 'None', marker = 'None', capsize = 0, elinewidth = 0.5, ecolor = 'black')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def create_title_page(header, inputfile, outfile, verbose):
    """
    Creates a title page for the document
    """

    if(verbose):
        print "Create title page"

    fig = plt.figure(num = 0, figsize = (11.69, 2 * 8.27), dpi = None, facecolor = 'white', edgecolor = None, frameon = True)
    fig.clear()

    fig.text(0.5, 0.75, "A correlation analysis created by CAT", fontsize = 'xx-large', va = 'center', ha = 'center', transform = fig.transFigure)
    fig.text(0.5, 0.70, "Creation time: " + str(datetime.datetime.today().strftime("%Y-%m-%d  %H:%M")), fontsize = 'large', va = 'center', ha = 'center', transform = fig.transFigure)
    fig.text(0.5, 0.68, "   Input file: " + str(inputfile), fontsize = 'large', va = 'center', ha = 'center', transform = fig.transFigure)
    fig.text(0.5, 0.1, "Please report bugs to michael.prim@kit.edu", fontsize = 'large', va = 'center', ha = 'center', transform = fig.transFigure)

    # list variables
    xpos = 0.25
    if(len(header) <= 10):
        xpos = 0.45
    ypos = 0
    for i in xrange(0, len(header)):
        if(i == 10):
            xpos += 0.2
            ypos -= 9
        else:
            ypos += 1
        fig.text(xpos, 0.6 - ypos * 0.02, str(i + 1) + ": " + header[i], fontsize = 'large', va = 'center', ha = 'left', transform = fig.transFigure)

    if(type(outfile) == str):
        fig.savefig(outfile)
    else:
        outfile.savefig(fig)


def create_correlation_matrix_page(data, header, cor_type, outfile, verbose):
    """
    Creates a page with a correlation matrix of the given cor_type
    (Pearson, Spearman or Kendall's tau) for the data
    """

    if(verbose):
        print "Create page with " + cor_type + " correlation matrix"

    nvars = len(data)
    cor_matrix = np.zeros((nvars, nvars), dtype = 'd')

    for i in xrange(0, nvars):
        for j in xrange(0, nvars):
            if(cor_type == "Pearson"):
                cor_matrix[i][j] = stats.pearsonr(data[i], data[j])[0]
            elif(cor_type == "Spearman"):
                cor_matrix[i][j] = stats.spearmanr(data[i], data[j])[0]
            elif(cor_type == "Kendall's tau"):
                cor_matrix[i][j] = stats.kendalltau(data[i], data[j])[0]
            else:
                raise Exception("Unknown correlation coefficient cor_type to calculate...")

    if(verbose):
        print cor_type
        print cor_matrix

    # set diagonal elements to value out of range
    for i in xrange(0, nvars):
        cor_matrix[i][i] = -999

    fig = plt.figure(num = 0, figsize = (11.69, 2 * 8.27), dpi = None, facecolor = 'white', edgecolor = None, frameon = True)
    fig.clear()

    ax = plt.axes([0.11, 0.92 - 0.80 / math.sqrt(2), 0.80, 0.80 / math.sqrt(2)])

    cdict = {'red': ((0.0, 0.0, 0.0),
                 (0.025, 0.0, 0.0),
                 (0.475, 1.0, 1.0),
                 (0.525, 1.0, 1.0),
                 (0.975, 1.0, 1.0),
                 (1.0, 1.0, 0.0)),
         'green': ((0.0, 0.0, 0.0),
                 (0.025, 0.0, 0.0),
                 (0.475, 1.0, 1.0),
                 (0.525, 1.0, 1.0),
                 (0.975, 0.0, 0.0),
                 (1.0, 0.0, 0.0)),
         'blue': ((0.0, 0.0, 1.0),
                 (0.025, 1.0, 1.0),
                 (0.475, 1.0, 1.0),
                 (0.525, 1.0, 1.0),
                 (0.975, 0.0, 0.0),
                 (1.0, 0.0, 0.0))}
    cmap = colors.LinearSegmentedColormap('cmap', cdict, 256)
    #cmap = plt.cm.get_cmap(name = 'seismic')
    cmap.set_over('black')
    cmap.set_under('black')
    im = ax.imshow(cor_matrix, clim = (-1, 1), cmap = cmap, interpolation = 'nearest')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.set_xticklabels(range(1, len(cor_matrix) + 1))
    ax.set_yticklabels(range(1, len(cor_matrix) + 1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size = "5%", pad = 0.1)
    fig.colorbar(im, cax = cax)
    
    # add corr. coeff test into matrix plot
    for i in xrange(0, nvars):
        for j in xrange(0, nvars):
            if(cor_matrix[i][nvars-1-j] == -999):
                continue
            
            xstep = (0.8 * 0.95) / nvars
            ystep = (0.8 * 0.95 / math.sqrt(2)) / nvars
            
            cor_coeff_string = r"%.2f" % cor_matrix[i][nvars-1-j]
            if(nvars > 10):            
                plt.text(0.11 + xstep/2.0 + i*xstep, (0.92 - 0.8 / math.sqrt(2)) + ystep/6.0 + ystep/2.0 + j*ystep, cor_coeff_string, fontsize = 'small', va = 'center', ha = 'center', transform = fig.transFigure)
            else:
                plt.text(0.11 + xstep/2.0 + i*xstep, (0.92 - 0.8 / math.sqrt(2)) + ystep/6.0 + ystep/2.0 + j*ystep, cor_coeff_string, va = 'center', ha = 'center', transform = fig.transFigure)

    # add page title    
    if(cor_type == "Pearson"):
        plt.text(0.5, 0.97, "Pearson's correlation coefficient matrix of variables", fontsize = 'large', va = 'center', ha = 'center', transform = fig.transFigure)
    elif(cor_type == "Spearman"):
        plt.text(0.5, 0.97, "Spearman's rank correlation coefficient matrix of variables", fontsize = 'large', va = 'center', ha = 'center', transform = fig.transFigure)
    elif(cor_type == "Kendall's tau"):
        plt.text(0.5, 0.97, "Kendall's tau rank correlation coefficient matrix of variables", fontsize = 'large', va = 'center', ha = 'center', transform = fig.transFigure)

    # list variables
    xpos = 0.15
    ypos = 0
    for i in xrange(0, len(header)):
        if(i == 10):
            xpos += 0.2
            ypos -= 9
        else:
            ypos += 1
        plt.text(xpos, 0.82 - 0.80 / math.sqrt(2) - ypos * 0.02, str(i + 1) + ": " + header[i], fontsize = 'large', va = 'center', ha = 'left', transform = fig.transFigure)

    if(type(outfile) == str):
        fig.savefig(outfile)
    else:
        outfile.savefig(fig)


def create_significance_matrix_page(data, header, outfile, verbose):
    """
    Creates a plot of the significance matrix
    """

    nvars = len(data)
    cor_matrix = data

    if(verbose):
        print "Create page with significance matrix"
        
    # set diagonal elements to value out of range
    for i in xrange(0, nvars):
        cor_matrix[i][i] = -999

    fig = plt.figure(num = 0, figsize = (11.69, 2 * 8.27), dpi = None, facecolor = 'white', edgecolor = None, frameon = True)
    fig.clear()

    ax = plt.axes([0.11, 0.92 - 0.80 / math.sqrt(2), 0.80, 0.80 / math.sqrt(2)])

    #cmap = plt.get_cmap("YlOrRd")
    
    cdict = {'red': ((0.0, 1.0, 1.0),
                 (0.1, 1.0, 1.0),
                 (0.2, 1.0, 1.0),
                 (0.3, 1.0, 1.0),
                 (0.4, 1.0, 1.0),
                 (0.5, 1.0, 1.0),
                 (0.6, 1.0, 1.0),
                 (0.7, 1.0, 1.0),
                 (0.8, 1.0, 0.5),
                 (0.9, 0.5, 0.5),
                 (1.0, 0.5, 0.0)),
         'green': ((0.0, 1.0, 1.0),
                 (0.1, 1.0, 1.0),
                 (0.2, 1.0, 1.0),
                 (0.3, 1.0, 0.6),
                 (0.4, 0.6, 0.6),
                 (0.5, 0.6, 0.0),
                 (0.6, 0.0, 0.0),
                 (0.7, 0.0, 0.0),
                 (0.8, 0.0, 0.0),
                 (0.9, 0.0, 0.0),
                 (1.0, 0.0, 0.0)),
         'blue': ((0.0, 1.0, 1.0),
                 (0.1, 1.0, 1.0),
                 (0.2, 1.0, 0.0),
                 (0.3, 0.0, 0.0),
                 (0.4, 0.0, 0.0),
                 (0.5, 0.0, 0.0),
                 (0.6, 0.0, 0.0),
                 (0.7, 0.0, 0.0),
                 (0.8, 0.0, 0.0),
                 (0.9, 0.0, 0.0),
                 (1.0, 0.0, 0.0))}
    cmap = colors.LinearSegmentedColormap('cmap', cdict, 256)
    
    cmap.set_over('black')
    cmap.set_under('black')
    im = ax.imshow(cor_matrix, clim = (0, 10), cmap = cmap, interpolation = 'nearest')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.set_xticklabels(range(1, len(cor_matrix) + 1))
    ax.set_yticklabels(range(1, len(cor_matrix) + 1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size = "5%", pad = 0.1)
    fig.colorbar(im, cax = cax)
    
    # add corr. coeff test into matrix plot
    for i in xrange(0, nvars):
        for j in xrange(0, nvars):
            if(cor_matrix[i][nvars-1-j] == -999):
                continue
            
            xstep = (0.8 * 0.95) / nvars
            ystep = (0.8 * 0.95 / math.sqrt(2)) / nvars
            
            if(cor_matrix[i][nvars-1-j] < 8):
                cor_coeff_string = r"%.2f$\sigma$" % cor_matrix[i][nvars-1-j]
            else:
                cor_coeff_string = r"$>$8$\sigma$"
                
            if(nvars > 10):            
                plt.text(0.11 + xstep/2.0 + i*xstep, (0.92 - 0.8 / math.sqrt(2)) + ystep/6.0 + ystep/2.0 + j*ystep, cor_coeff_string, fontsize = 'small', va = 'center', ha = 'center', transform = fig.transFigure)
            else:
                plt.text(0.11 + xstep/2.0 + i*xstep, (0.92 - 0.8 / math.sqrt(2)) + ystep/6.0 + ystep/2.0 + j*ystep, cor_coeff_string, va = 'center', ha = 'center', transform = fig.transFigure)

    # add page title    
    plt.text(0.5, 0.97, "Significance matrix of the test for two variables being dependent", fontsize = 'large', va = 'center', ha = 'center', transform = fig.transFigure)

    # list variables
    xpos = 0.15
    ypos = 0
    for i in xrange(0, len(header)):
        if(i == 10):
            xpos += 0.2
            ypos -= 9
        else:
            ypos += 1
        plt.text(xpos, 0.82 - 0.80 / math.sqrt(2) - ypos * 0.02, str(i + 1) + ": " + header[i], fontsize = 'large', va = 'center', ha = 'left', transform = fig.transFigure)

    if(type(outfile) == str):
        fig.savefig(outfile)
    else:
        outfile.savefig(fig)


def create_profile_matrix_page(data, header, outfile, verbose):
    """
    Create a page with a profile plot matrix for all variables in data
    """

    if(verbose):
        print "Create profile plot matrix page"

    fig = plt.figure(num = 0, figsize = (11.69, 2 * 8.27), dpi = None, facecolor = 'white', edgecolor = None, frameon = True)
    fig.clear()

    nbins_diagonal = 50

    nvars = len(data)

    # Create plot matrix
    ax_grid = []
    for i in xrange(0, nvars):
        ax_grid.append([])
        for j in xrange(0, nvars):
            width = 0.89 / nvars
            heigth = 0.89 / math.sqrt(2) / nvars
            ax_grid[i].append(plt.axes([0.07 + i * width, 0.92 - (j + 1) * heigth, width, heigth]))

    # Create plots
    y_scale_min, y_scale_max = 0, 0
    for i in xrange(0, nvars):
        i_min, i_max = min(data[i]), max(data[i])
        for j in xrange(0, nvars):
            j_min, j_max = min(data[j]), max(data[j])
            if(i != j):
                plot_profile_plot_to_axes(data[i], data[j], (i_min, i_max), (j_min, j_max), nbins_diagonal, ax_grid[i][j])
            else:
                plot_histogram_to_axes(histogram(*get_bins_and_content(i_min, i_max, nbins_diagonal, data[i])), ax_grid[i][j], 'orange')
                ymax = ax_grid[i][i].get_ylim()[1]
                if(ymax > y_scale_max):
                    y_scale_max = ymax

    # Set y limits to the same scale for all plots on diagonal
    for i in xrange(0, nvars):
        ax_grid[i][i].set_ylim(y_scale_min, y_scale_max)

    # adjust grid elements
    for i in xrange(0, nvars):
        for j in xrange(0, nvars):
            #ax_grid[i][j].text(0.9,0.9,str(i)+","+str(j), fontsize='xx-small', transform=ax_grid[i][j].transAxes)
            ax_grid[i][j].xaxis.set_major_locator(ticker.MaxNLocator(5))
            ax_grid[i][j].yaxis.set_major_locator(ticker.MaxNLocator(5))
            for tickline in ax_grid[i][j].get_xticklines() + ax_grid[i][j].get_yticklines():
                tickline.set_markersize(2)
            for label in ax_grid[i][j].get_xticklabels() + ax_grid[i][j].get_yticklabels():
                label.set_fontsize('xx-small')

            if(j == 0): # top row
                twin = ax_grid[i][j].twiny()
                twin.set_xlim(ax_grid[i][i].get_xlim())
                twin.xaxis.set_major_locator(ticker.MaxNLocator(5))
                twin.yaxis.set_major_locator(ticker.MaxNLocator(5))
                for tickline in twin.get_xticklines() + twin.get_yticklines():
                    tickline.set_markersize(2)
                for label in twin.get_xticklabels() + twin.get_yticklabels():
                    label.set_fontsize('xx-small')

            if(i == nvars - 1): # right coloum
                if(j == nvars - 1):
                    twin = ax_grid[i][j].twinx()
                    twin.set_ylim(ax_grid[i][j].get_ylim())
                    twin.xaxis.set_major_locator(ticker.MaxNLocator(5))
                    twin.yaxis.set_major_locator(ticker.MaxNLocator(5))
                    for tickline in twin.get_xticklines() + twin.get_yticklines():
                        tickline.set_markersize(2)
                    for label in twin.get_xticklabels() + twin.get_yticklabels():
                        label.set_fontsize('xx-small')
                else:
                    twin = ax_grid[i][j].twinx()
                    twin.set_ylim(ax_grid[j][j].get_xlim())
                    twin.xaxis.set_major_locator(ticker.MaxNLocator(5))
                    twin.yaxis.set_major_locator(ticker.MaxNLocator(5))
                    for tickline in twin.get_xticklines() + twin.get_yticklines():
                        tickline.set_markersize(2)
                    for label in twin.get_xticklabels() + twin.get_yticklabels():
                        label.set_fontsize('xx-small')

            if(j != nvars - 1): #none bottom row
                ax_grid[i][j].set_xticklabels([], minor = False, visible = False)

            if(i != 0): # none left coloum
                ax_grid[i][j].set_yticklabels([], minor = False, visible = False)

    # Add other information to plot  
    plt.text(0.5, 0.97, "This page shows a profile plot matrix of all variables", fontsize = 'large', va = 'center', ha = 'center', transform = fig.transFigure)

    # list variables
    xpos = 0.15
    ypos = 0
    for i in xrange(0, len(header)):
        if(i == 10):
            xpos += 0.2
            ypos -= 9
        else:
            ypos += 1
        plt.text(xpos, 0.82 - 0.80 / math.sqrt(2) - ypos * 0.02, str(i + 1) + ": " + header[i], fontsize = 'large', va = 'center', ha = 'left', transform = fig.transFigure)

    if(type(outfile) == str):
        fig.savefig(outfile)
    else:
        outfile.savefig(fig)


def create_correlation_analysis_page(data, nbins, header, variable_x, variable_y, outfile, verbose):
    """
    Create a page with a correlation analysis of one variable_x in nbins of another
    variable_y.    
    """

    if(verbose):
        print "Create page for " + header[variable_x] + " vs. " + header[variable_y]

    # Create Page figsize=(8.27,11.69) for DIN A4 and plot matrix
    fig = plt.figure(num = 0, figsize = (11.69, 2 * 8.27), dpi = None, facecolor = 'white', edgecolor = None, frameon = True)
    fig.clear()

    nbins_diagonal = 50

    # Create plot matrix
    ax_grid = []
    h_grid = []
    for i in xrange(0, nbins):
        ax_grid.append([])
        h_grid.append([])
        for j in xrange(0, nbins):
            width = 0.89 / nbins
            heigth = 0.89 / math.sqrt(2) / nbins
            ax_grid[i].append(plt.axes([0.07 + i * width, 0.92 - (j + 1) * heigth, width, heigth]))
            h_grid[i].append([])

    # Create Distribution of x        
    ax_dist_x = plt.axes([0.07, 0.05, 0.26, 0.18])
    var_x_min, var_x_max = min(data[variable_x]), max(data[variable_x])
    h_dist_x = plot_histogram_to_axes(histogram(*get_bins_and_content(var_x_min, var_x_max, nbins_diagonal, data[variable_x])), ax_dist_x, 'orange')
    ax_dist_x.set_title(header[variable_x])
    ax_dist_x.xaxis.set_major_locator(ticker.MaxNLocator(5))
    for label in ax_dist_x.get_xticklabels() + ax_dist_x.get_yticklabels():
        label.set_fontsize('xx-small')
    for tickline in ax_dist_x.get_xticklines() + ax_dist_x.get_yticklines():
        tickline.set_markersize(3)

    # Create Distribution of y
    ax_dist_y = plt.axes([0.385, 0.05, 0.26, 0.18])
    var_y_min, var_y_max = min(data[variable_y]), max(data[variable_y])
    h_dist_y = plot_histogram_to_axes(histogram(*get_bins_and_content(var_y_min, var_y_max, nbins_diagonal, data[variable_y])), ax_dist_y, 'orange')
    ax_dist_y.set_title(header[variable_y])
    ax_dist_y.xaxis.set_major_locator(ticker.MaxNLocator(5))
    for label in ax_dist_y.get_xticklabels() + ax_dist_y.get_yticklabels():
        label.set_fontsize('xx-small')
    for tickline in ax_dist_y.get_xticklines() + ax_dist_y.get_yticklines():
        tickline.set_markersize(3)

    # Create Profile plot
    ax_prof = plt.axes([0.70, 0.05, 0.26, 0.18])
    plot_profile_plot_to_axes(data[variable_x], data[variable_y], ax_dist_x.get_xlim(), ax_dist_y.get_xlim(), nbins_diagonal, ax_prof)
    ax_prof.set_title(header[variable_x] + " vs. " + header[variable_y])
    ax_prof.set_xlim(ax_dist_x.get_xlim())
    ax_prof.set_ylim(ax_dist_y.get_xlim())
    ax_prof.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax_prof.yaxis.set_major_locator(ticker.MaxNLocator(5))
    for label in ax_prof.get_xticklabels() + ax_prof.get_yticklabels():
        label.set_fontsize('xx-small')
    for tickline in ax_prof.get_xticklines() + ax_prof.get_yticklines():
        tickline.set_markersize(3)

    # Add other information to plot  
    plt.text(0.5, 0.97, "This page shows " + header[variable_x] + " in bins of " + header[variable_y], fontsize = 'large', va = 'center', ha = 'center', transform = fig.transFigure)

    # Sort data for variable y
    data = zip(*data)
    data.sort(key = lambda x: x[variable_y], reverse = False)
    data = zip(*data)
    data = np.array(data, dtype = 'd')

    # Draw x in n bins of y to diagonal elements
    y_scale_min, y_scale_max = 0, 0
    for i in xrange(0, nbins):
        h_grid[i][i] = plot_histogram_to_axes(histogram(*get_bins_and_content(var_x_min, var_x_max, nbins_diagonal, [data[variable_x][j] for j in xrange(i * len(data[variable_x]) / nbins, (i + 1) * len(data[variable_x]) / nbins)])), ax_grid[i][i], 'orange')
        ymax = ax_grid[i][i].get_ylim()[1]
        if(ymax > y_scale_max):
            y_scale_max = ymax
        if(i > 0):
            ax_dist_y.axvline(data[variable_y][i * len(data[variable_x]) / nbins], color = 'red', linestyle = '--')

    # Set y limits to the same scale for all plots on diagonal  
    for i in xrange(0, nbins):
        ax_grid[i][i].set_ylim(y_scale_min, y_scale_max)

    # Subtract two those distributions and divide by error, draw to lower off-diagonal elements
    for i in xrange(0, nbins):
        for j in xrange(0, nbins):
            if(i < j):
                h_grid[i][j] = plot_histogram_to_axes(h_grid[i][i].get_pull_distribution(h_grid[j][j]), ax_grid[i][j], '#32CD32') #lime green color code
                ax_grid[i][j].axhline(0, linestyle = '--', color = 'black')

    # Draw Projection of pulls to upper off-diagnoal elements
    y_scale_min, y_scale_max = 0, 0
    for i in xrange(0, nbins):
        for j in xrange(0, nbins):
            if(i > j):
                h_grid[i][j] = plot_histogram_to_axes(histogram(*get_bins_and_content(-5, 5, 20, h_grid[j][i].get_bin_contents())), ax_grid[i][j], '#007FFF') #azure color code
                ymax = ax_grid[i][j].get_ylim()[1]
                if(ymax > y_scale_max):
                    y_scale_max = ymax

    # Set y limits to the same scale for all plots in the upper half          
    for i in xrange(0, nbins):
        for j in xrange(0, nbins):
            if(i > j):
                ax_grid[i][j].set_ylim(y_scale_min, y_scale_max)

    # adjust grid elements
    for i in xrange(0, nbins):
        for j in xrange(0, nbins):
            #ax_grid[i][j].text(0.9,0.9,str(i)+","+str(j), fontsize='xx-small', transform=ax_grid[i][j].transAxes)
            ax_grid[i][j].xaxis.set_major_locator(ticker.MaxNLocator(5))
            ax_grid[i][j].yaxis.set_major_locator(ticker.MaxNLocator(5))
            for tickline in ax_grid[i][j].get_xticklines() + ax_grid[i][j].get_yticklines():
                tickline.set_markersize(2)
            for label in ax_grid[i][j].get_xticklabels() + ax_grid[i][j].get_yticklabels():
                label.set_fontsize('xx-small')

            if(j != i and i > j): # upper half
                ax_grid[i][j].set_xticks([-4, -2, 0, 2, 4])
                ax_grid[i][j].set_xlim(-5, 5)
                ax_grid[i][j].xaxis.grid(True, 'major')
                ax_grid[i][j].yaxis.grid(True, 'major')

            if(j != i and i < j): # bottom half 
                ax_grid[i][j].set_xticklabels([''] + range(10, nbins_diagonal, 10), minor = False, visible = True)
                ax_grid[i][j].set_yticks([-4, -2, 0, 2, 4])
                ax_grid[i][j].set_ylim(-5, 5)
                ax_grid[i][j].xaxis.grid(True, 'major')
                ax_grid[i][j].yaxis.grid(True, 'major')

            if(j == 0): # top row
                if(i == 0):
                    twin = ax_grid[i][j].twiny()
                    twin.xaxis.set_major_locator(ticker.MaxNLocator(5))
                    twin.yaxis.set_major_locator(ticker.MaxNLocator(5))
                    twin.set_xlim(ax_grid[i][j].get_xlim())
                    for tickline in twin.get_xticklines() + twin.get_yticklines():
                        tickline.set_markersize(2)
                    for label in twin.get_xticklabels() + twin.get_yticklabels():
                        label.set_fontsize('xx-small')
                else:
                    twin = ax_grid[i][j].twiny()
                    twin.set_xlim(ax_grid[i][j].get_xlim())
                    twin.set_xticks(ax_grid[i][j].get_xticks())
                    for tickline in twin.get_xticklines() + twin.get_yticklines():
                        tickline.set_markersize(2)
                    for label in twin.get_xticklabels() + twin.get_yticklabels():
                        label.set_fontsize('xx-small')

            if(j != i and i > j): # upper half
                ax_grid[i][j].set_yticks(range(5, int(y_scale_max), 5))

            if(i == nbins - 1): # right coloum
                if(j == nbins - 1):
                    twin = ax_grid[i][j].twinx()
                    twin.xaxis.set_major_locator(ticker.MaxNLocator(5))
                    twin.yaxis.set_major_locator(ticker.MaxNLocator(5))
                    twin.set_ylim(ax_grid[i][j].get_ylim())
                    for tickline in twin.get_xticklines() + twin.get_yticklines():
                        tickline.set_markersize(2)
                    for label in twin.get_xticklabels() + twin.get_yticklabels():
                        label.set_fontsize('xx-small')
                else:
                    twin = ax_grid[i][j].twinx()
                    twin.set_ylim(ax_grid[i][j].get_ylim())
                    twin.set_yticks(ax_grid[i][j].get_yticks())
                    for tickline in twin.get_xticklines() + twin.get_yticklines():
                        tickline.set_markersize(2)
                    for label in twin.get_xticklabels() + twin.get_yticklabels():
                        label.set_fontsize('xx-small')

            if(j != nbins - 1): #none bottom row
                ax_grid[i][j].set_xticklabels([], minor = False, visible = False)
            if(i != 0): # none left coloum
                ax_grid[i][j].set_yticklabels([], minor = False, visible = False)

    if(type(outfile) == str):
        fig.savefig(outfile)
    else:
        outfile.savefig(fig)


def create_flat_correlation_matix_page(data, header, variable_x, variable_y, outfile, verbose):
    """
    Create a page with a 2D matrix where variable_x and y are flattened.
    
    Also performs a chi2 test to check the null hypothesis. 
    """

    nbins = 3
    while(len(data[0]) / ((nbins + 10) ** 2) > 25):
        nbins += 1
    nbins = min(nbins, 100)

    if(verbose):
        print "Create page with flat correlation matrix"
        print "Using " + str(nbins) + " flat bins in each variable"

    fig = plt.figure(num = 0, figsize = (11.69, 2 * 8.27), dpi = None, facecolor = 'white', edgecolor = None, frameon = True)
    fig.clear()

    matrix = np.zeros((nbins, nbins), dtype = 'd')

    # flatten x
    data = zip(*data)
    data.sort(key = lambda x: x[variable_x], reverse = False)
    data = zip(*data)
    data = np.array(data, dtype = 'd')
    binedges_x = np.zeros(nbins + 1, dtype = 'd')
    for i in xrange(0, nbins):
        binedges_x[i] = data[variable_x][i * len(data[variable_x]) / nbins]
    binedges_x[-1] = data[variable_x][-1]

    # flatten y
    data = zip(*data)
    data.sort(key = lambda x: x[variable_y], reverse = False)
    data = zip(*data)
    data = np.array(data, dtype = 'd')
    binedges_y = np.zeros(nbins + 1, 'd')
    for i in xrange(0, nbins):
        binedges_y[i] = data[variable_y][i * len(data[variable_y]) / nbins]
    binedges_y[-1] = data[variable_y][-1]

    # search for x and y bin
    for i in xrange(0, len(data[variable_x])):
        binx = nbins - 1
        while(data[variable_x][i] < binedges_x[binx]):
            binx -= 1
        biny = nbins - 1
        while(data[variable_y][i] < binedges_y[biny]):
            biny -= 1
        matrix[binx][biny] += 1

    # calc expected bin content and error
    expec = len(data[variable_x]) / float(nbins ** 2)
    error = math.sqrt(expec)

    # calc pulls and chi2
    chi2 = 0
    for i in xrange(0, nbins):
        for j in xrange(0, nbins):
            matrix[i][j] = (matrix[i][j] - expec) / error
            chi2 += matrix[i][j] * matrix[i][j]

    # dgf = nbins*nbins - ((nbins-1)*2 +1) 
    # in each bin of x(y) 1/nbins part of the statistic is stored
    # last bin is determined by all other bins in x(y)
    # subtract over all amount of entries
    prob = stats.chisqprob(chi2, nbins * nbins - ((nbins - 1) + (nbins - 1) + 1))

    if(verbose):
        print "Probability for data " + header[variable_x] + " vs. " + header[variable_y] + " being consistent with flat hypothesis is %.6f" % (prob * 100)

    ax = plt.axes([0.11, 0.92 - 0.80 / math.sqrt(2), 0.80, 0.80 / math.sqrt(2)])

    cdict = {'red': ((0.0, 0.0, 0.0),
                     (0.1, 0.0, 0.0),
                     (0.2, 0.0, 0.0),
                     (0.3, 0.0, 0.0),
                     (0.4, 0.0, 1.0),
                     (0.5, 1.0, 1.0),
                     (0.6, 1.0, 1.0),
                     (0.7, 1.0, 1.0),
                     (0.8, 1.0, 1.0),
                     (0.9, 1.0, 0.5),
                     (1.0, 0.5, 0.0)),
             'green': ((0.0, 0.0, 0.0),
                     (0.1, 0.0, 1.0),
                     (0.2, 1.0, 0.5),
                     (0.3, 0.5, 1.0),
                     (0.4, 1.0, 1.0),
                     (0.5, 1.0, 1.0),
                     (0.6, 1.0, 1.0),
                     (0.7, 1.0, 0.6),
                     (0.8, 0.6, 0.0),
                     (0.9, 0.0, 0.0),
                     (1.0, 0.0, 0.0)),
             'blue': ((0.0, 0.0, 1.0),
                     (0.1, 1.0, 1.0),
                     (0.2, 1.0, 0.0),
                     (0.3, 0.0, 0.0),
                     (0.4, 0.0, 1.0),
                     (0.5, 1.0, 1.0),
                     (0.6, 1.0, 0.0),
                     (0.7, 0.0, 0.0),
                     (0.8, 0.0, 0.0),
                     (0.9, 0.0, 0.0),
                     (1.0, 0.0, 0.0))}
    cmap = colors.LinearSegmentedColormap('cmap', cdict, 256)
    cmap.set_over('#800000')
    cmap.set_under('#0000FF')
    im = ax.imshow(matrix, clim = (-5, 5), cmap = cmap, interpolation = 'nearest')

    ticks_x = []
    ticks_x_loc = []
    ticks_y = []
    ticks_y_loc = []
    for i in xrange(0, nbins + 1):
        ticks_x.append("%.6g" % binedges_x[i])
        ticks_x_loc.append(-0.5 + i)
        ticks_y.append("%.6g" % binedges_y[i])
        ticks_y_loc.append(-0.5 + i)
    ax.xaxis.set_major_locator(ticker.FixedLocator(ticks_x_loc))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText = True))
    ax.yaxis.set_major_locator(ticker.FixedLocator(ticks_y_loc))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText = True))
    ax.ticklabel_format(style = 'scientific', scilimits = (0, 5), axis = 'x')
    ax.ticklabel_format(style = 'scientific', scilimits = (0, 5), axis = 'y')
    ax.set_xticklabels(ticks_x, rotation = 270, fontsize = 'x-small', horizontalalignment = 'center')
    ax.set_yticklabels(ticks_y, fontsize = 'x-small', horizontalalignment = 'right')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size = "5%", pad = 0.1)
    fig.colorbar(im, cax = cax)

    # add probability and significance text
    significance = stats.norm.interval(1-prob)
    significance_string = ""
    if(math.isinf(abs(significance[0]))):
        significance_string = "more than 8 sigma significance."
    else:
        significance_string = "%.2f sigma significance." % abs(significance[0])
        
    plt.text(0.5, 0.25, "Probability of the data\n\n" +
                        "" + header[variable_x] + " vs. " + header[variable_y] + "\n\n" +
                        ("to be consistent with a flat hypothesis is %.6f \n\n" % prob) +
                        "This corresponds to a correlation with " + significance_string + "" 
                        , fontsize = 'large', va = 'center', ha = 'center', transform = fig.transFigure)

    plt.text(0.5, 0.10, "approximate 0.317311 corresponds to 1 sigma \n"
                        "approximate 0.045500 corresponds to 2 sigma \n"
                        "approximate 0.002700 corresponds to 3 sigma \n"
                        "approximate 0.000063 corresponds to 4 sigma \n"
                        "approximate 0.000001 corresponds to 5 sigma \n"
                        , fontsize = 'large', va = 'center', ha = 'center', transform = fig.transFigure)

    # add page title
    plt.text(0.5, 0.95, "Matrix with the deviations in sigma of\n\n"
                        "" + header[variable_x] + " vs. " + header[variable_y] + "\n\n"
                        "from the expectation value of a flat distribution", fontsize = 'large', va = 'center', ha = 'center', transform = fig.transFigure)

    if(type(outfile) == str):
        fig.savefig(outfile)
    else:
        outfile.savefig(fig)
        
    return abs(significance[0])


def compare_version(version_string, major, minor):
    version_list = version_string.split('.')
    
    if(int(version_list[0]) > major):
        return True
    if(int(version_list[0]) < major):
        return False
    if(int(version_list[0]) == major):
        if(int(version_list[1]) >= minor):
            return True
        else:
            return False

def main():
    # Parse options
    parser = OptionParser()
    parser.add_option("-i", "--inputfile", dest = "inputfile", type = 'string',
                      help = "Input filename", metavar = "INPUTFILE")
    parser.add_option("-o", "--outfile", dest = "outfile", type = 'string',
                      help = "Output filename", metavar = "OUTFILE")
    parser.add_option("-t", "--tex", dest = "tex", type = 'string',
                      help = "File that includes code to be used as dictionary for TeX replacement of header names.", metavar = "TEX", default = None)
    parser.add_option("-f", "--fixed", dest = "fixed", type = 'string',
                      help = "Often only correlations of one variable to many others are interesting. "
                             "Pass a variable name as given in the header and fix this variable. This "
                             "will reduce the computing time especially for large sets of variables.", metavar = "FIXED", default = None)
    parser.add_option("-n", "--noheader", dest = "noheader",
                      help = "Activate if no header in the input file is given. This disables -t and -f options.", metavar = "NOHEADER", default = False, action = "store_true")
    parser.add_option("-e", "--epspage", dest = "epspage",
                      help = "Create EPS file for each page and join them afterwards. "
                      "This allows to include each page e.g. in some other document. "
                      "Requires epstopdf and pdfjoin to be installed on the machine. "
                      "As long as there is a bug with export to PdfPages via matplotlib it's forced on.", metavar = "EPSPAGE", default = True, action = "store_true")
    parser.add_option("-p", "--patient", dest = "patient",
                      help = "Create Kendall's tau correlation matrix. This takes some time and you should be (very) patient...", metavar = "PATIENT", default = False, action = "store_true")
    parser.add_option("-c", "--cleanup", dest = "cleanup",
                      help = "Cleanup mode (remove all temporary plots at the end)", metavar = "CLEANUP", default = False, action = "store_true")
    parser.add_option("-v", "--verbose", dest = "verbose",
                      help = "Verbose mode", metavar = "VERBOSE", default = False, action = "store_true")
    (options, args) = parser.parse_args()
    
    # check version requirements
    version_check_failed = False
    if(compare_version(np.version.version, 1, 5) == False):
        print "Numpy version 1.5.0 or newer is required, please update from currently installed " + str(np.version.version)
        version_check_failed = True
    
    if(compare_version(spversion.version, 0, 8) == False):
        print "Scipy version 0.8.0 or newer is required, please update from currently installed " + str(spversion.version)
        version_check_failed = True
    
    if(compare_version(mpl.__version__, 1, 1) == False):
        print "Matplotlib version 1.1.0 or newer is required, please update from currently installed " + str(mpl.__version__)
        version_check_failed = True

    if(options.epspage == False):
        from matplotlib.backends.backend_pdf import PdfPages
        options.outfile = PdfPages(options.outfile)
    else:
        print "Checking for necessary utilities in epspage mode:"
        if(os.system("which epstopdf") > 0):
            print "Could not find epstopdf utiltiy"
            version_check_failed = True
        if(os.system("which pdfjoin") > 0):
            print "Could not find pdfjoin utility"
            version_check_failed = True
    
    if(version_check_failed == True):
        sys.exit(-1)
    
    if(options.inputfile == None or options.outfile == None):
        print ""
        parser.print_help()
        sys.exit(-1)

    # Read input data and header
    reader = csv.reader(open(options.inputfile), delimiter = ';', quotechar = '|')
    header = None
    data = []
    for row in reader:
        if(header == None and options.noheader == False):
            header = row
            for i in xrange(len(row)):
                data.append([])
        elif(header == None and options.noheader == True):
            header = []
            for i in xrange(len(row)):
                header.append("variable_" + str(i))
                data.append([])
        else:
            for i in xrange(len(row)):
                data[i].append(row[i])
    data = np.array(data, dtype = 'd')

    # Check for fixed variable and unfix if noheader is set True
    if(options.fixed != None and options.noheader == False):
        if(options.fixed not in header):
            print "Fixed variable " + str(options.fixed) + " could not be matched to any variable given in header."
            sys.exit(-1)
    else:
        options.fixed = None

    # the raw text header with the names as in the input file, used for filenames
    header_raw = []

    # parse header dictionary
    if(options.tex != None and options.noheader == False):
        rc('font', family = 'serif')
        rc('text', usetex = True)
        header_dict = {}

        try:
            fileheader = open(options.tex)
        except:
            print "Failed to open TeX header file " + str(options.tex)
            sys.exit(-1)

        try:
            for line in fileheader:
                eval(str(line))
        except:
            print "Failed to evaluate given TeX header file... please check input..."
            sys.exit(-1)

    # replace header with nice TeX strings if option is set
    for i in xrange (0, len(header)):
        header_raw.append(header[i])
        if(options.tex != None and options.noheader == False):
            if(header[i] in header_dict):
                header[i] = header_dict[header[i]]
            else:
                print 'Could not find a TeX replacement for ' + header[i] + ' in the given header file'
                sys.exit(-1)

    if(options.verbose):
        print "Input data sample..."
        print header
        for d in data:
            print str(d[0:3]) + ' ... ' + str(d[-3:])

    workingdir = "./CAT_tmpdir_" + str(uuid.uuid4()) + "/"
    print "Working directory is " + workingdir
    os.system("mkdir -p " + workingdir)

    # Determine in how many bins of y each variable x should be plotted
    nbins = get_nbins_for_plot_matrix(data, options.verbose)

    # Create title page
    if(options.epspage):
        create_title_page(header, options.inputfile, workingdir + "ca_page_title.eps", options.verbose)
    else:
        create_title_page(header, options.inputfile, options.outfile, options.verbose)

    # Create collreation matrix of input variables
    if(options.epspage):
        create_correlation_matrix_page(data, header, "Pearson", workingdir + "ca_page_1_correlation_matrix.eps", options.verbose)
        create_correlation_matrix_page(data, header, "Spearman", workingdir + "ca_page_2_correlation_matrix.eps", options.verbose)
        if(options.patient):
            create_correlation_matrix_page(data, header, "Kendall's tau", workingdir + "ca_page_3_correlation_matrix.eps", options.verbose)
    else:
        create_correlation_matrix_page(data, header, "Pearson", options.outfile, options.verbose)
        create_correlation_matrix_page(data, header, "Spearman", options.outfile, options.verbose)
        if(options.patient):
            create_correlation_matrix_page(data, header, "Kendall's tau", options.outfile, options.verbose)

    # Create double flat correlation matrix
    significance_matrix = np.empty((len(header),len(header)), dtype='float64')
    for variable_x in xrange(0, len(header)):
        if(options.fixed != None):
            if(header_raw[variable_x] != options.fixed):
                continue
        for variable_y in xrange(0, len(header)):
            if(variable_x == variable_y):
                continue
            else:
                if(options.epspage):
                    significance_buffer = create_flat_correlation_matix_page(data, header, variable_x, variable_y, workingdir + "ca_page_flat_var_" + header_raw[variable_x] + "_vs_" + header_raw[variable_y] + ".eps", options.verbose)
                else:
                    significance_buffer = create_flat_correlation_matix_page(data, header, variable_x, variable_y, options.outfile, options.verbose)
                if(math.isinf(significance_buffer)):
                    significance_buffer = 9
                significance_matrix[variable_x,variable_y] = float(significance_buffer)
    if(options.verbose):
        print "Significance matrix of dependence (9 is just a placeholder for cutoff at >8 sigma)"
        print significance_matrix
        
    # create the significance matrix page
    create_significance_matrix_page(significance_matrix, header, workingdir + "ca_page_significance_matrix.eps", options.verbose)

    # Create profile plot matrix of input variables
    if(options.epspage):
        create_profile_matrix_page(data, header, workingdir + "ca_page_profile_matrix.eps", options.verbose)
    else:
        create_profile_matrix_page(data, header, options.outfile, options.verbose)

    # Create correlation analysis for each variable 
    for variable_x in xrange(0, len(header)):
        if(options.fixed != None):
            if(header_raw[variable_x] != options.fixed):
                continue
        for variable_y in xrange(0, len(header)):
            if(variable_x == variable_y):
                continue
            else:
                if(options.epspage):
                    create_correlation_analysis_page(data, nbins, header, variable_x, variable_y, workingdir + "ca_page_var_" + header_raw[variable_x] + "_vs_" + header_raw[variable_y] + ".eps", options.verbose)
                else:
                    create_correlation_analysis_page(data, nbins, header, variable_x, variable_y, options.outfile, options.verbose)

    # Join all the pages and create final analysis output file
    if(options.epspage):
        if(options.verbose):
            print "Start with eps to pdf conversion and create final file"
        try:
            os.system("epstopdf " + workingdir + "ca_page_title.eps")
            for i in xrange(1, 4):
                if(i == 3 and options.patient == False):
                    continue
                os.system("epstopdf " + workingdir + "ca_page_" + str(i) + "_correlation_matrix.eps")
            for variable_x in xrange(0, len(header)):
                if(options.fixed != None):
                    if(header_raw[variable_x] != options.fixed):
                        continue
                for variable_y in xrange(0, len(header)):
                    if(variable_x == variable_y):
                        continue
                    else:
                        os.system("epstopdf " + workingdir + "ca_page_flat_var_" + header_raw[variable_x] + "_vs_" + header_raw[variable_y] + ".eps")
            os.system("epstopdf " + workingdir + "ca_page_profile_matrix.eps")
            os.system("epstopdf " + workingdir + "ca_page_significance_matrix.eps")
            for variable_x in xrange(0, len(header)):
                if(options.fixed != None):
                    if(header_raw[variable_x] != options.fixed):
                        continue
                for variable_y in xrange(0, len(header)):
                    if(variable_x == variable_y):
                        continue
                    else:
                        os.system("epstopdf " + workingdir + "ca_page_var_" + header_raw[variable_x] + "_vs_" + header_raw[variable_y] + ".eps")
            os.system("pdfjoin --outfile " + options.outfile + " " + workingdir + "ca_page_title.pdf " + workingdir + "ca_page_*_correlation_matrix.pdf " + workingdir + "ca_page_profile_matrix.pdf " + workingdir + "ca_page_significance_matrix.pdf " + workingdir + "ca_page_flat_var_*.pdf " + workingdir + "ca_page_var_*.pdf")
            os.system("rm " + workingdir + "ca_page_*.pdf")
            if(options.cleanup):
                os.system("rm " + workingdir + "ca_page_*.eps")
                os.system("rm -rf " + workingdir)
        except:
            print "Some error occured during the final joining of all the individual pages to a single correlation analysis file."
            print "Please make sure that epstopdf as well as pdfjoin are installed on your system!"
    else:
        d = options.outfile.infodict()
        d['Title'] = 'A correlation analysis using CAT'
        d['Author'] = 'Michael Prim'
        d['Subject'] = 'A correlation analysis using CAT'
        d['Keywords'] = 'correlation analysis CAT'
        d['CreationDate'] = datetime.datetime.today()
        d['ModDate'] = datetime.datetime.today()
        options.outfile.close()

if __name__ == '__main__':
    main()

