# save files
fullproc_r = './resources/' # for words/names

# where to save the figures that are created
#fake_figs_dir = './example_hists/' # saving in github only for small numbers
# stop overwriting what is there
fake_figs_dir = '~/Downloads/tmp/JCDL2025/visual_qa_histograms/example_hists_complex/' # large dataset, same format
#fake_figs_dir = '~/LLM_VQA_JCDL2025/example_hists/fewshot/' # for fewshot learning


# format for saving images?
#img_format = ['pdf','jpeg']
img_format = ['jpeg']

# save diagnostic plot with all items labeled?
save_diagnostic_plot = True

# randomize names?
randomize_names = True


nHistTotal = 130 # how many to make?
restart = False # set to True to overwrite what is there

grace_ticks = 5 # ignore tick marks that are outside the box by this -- invisible most likely
itriesMax = 10 # start again if too many tries
verbose = True
verbose_qa = True


# ----------------------------------------
# Libraries

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
import pickle
import pandas as pd
#import string
from glob import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import ImageColor
import json
import os
from copy import deepcopy
import copy

from utils.metric_utils.utilities import isRectangleOverlap

import time

import matplotlib as mpl
import os
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}' #for \text command

# written libs -- will update locally
from utils.synthetic_fig_utils import subset_by_percent, \
 get_nrows_and_ncols, normalize_params_prob, get_ticks, get_titles_or_labels, \
 get_font_info

from utils.plot_utils import get_contour_plot, get_histogram_plot, \
   get_line_plot, get_scatter_plot, make_plot, colorbar_mods#, get_line_plot

from utils.text_utils import get_popular_nouns, get_inline_math

# create a bunch of fake figures
from utils.synthetic_fig_utils import normalize_params_prob
from utils.plot_parameters import plot_types_params, panel_params, \
  title_params, xlabel_params, colorbar_params, \
  ylabel_params, aspect_fig_params, dpi_params, tight_layout_params, \
  fontsizes, base

from utils.data_utils import get_data, NumpyEncoder
import numpy as np

import utils.distribution_utils

from importlib import reload
from utils.plot_qa_utils import log_scale_ax
import utils.distribution_utils
import utils.data_utils
reload(utils.distribution_utils)
reload(utils.data_utils)
from utils.data_utils import get_data
import utils.plot_classes_utils
reload(utils.plot_classes_utils)

from utils.synthetic_fig_utils import add_titles_and_labels

import utils.plot_utils
import utils.data_utils
import utils.plot_classes_utils
reload(utils.plot_classes_utils)
reload(utils.data_utils)
reload(utils.plot_utils)
from utils.plot_utils import make_plot, make_base_plot
from utils.data_utils import get_data

from utils.plot_utils import markers, make_base_plot

import utils.synthetic_fig_utils
reload(utils.synthetic_fig_utils)
from utils.synthetic_fig_utils import collect_plot_data_axes

marker_sizes = np.arange(0,10)+1
line_list_thick = np.arange(1,10)

use_uniques = True # use unique inlines
verbose = True

from sys import path
path.append('/Library/TeX/texbin/')

# debug
from importlib import reload

# for seed
from sys import maxsize as maxint

import warnings
warnings.filterwarnings("error")


import utils.plot_qa_utils
reload(utils.plot_qa_utils)
from utils.plot_qa_utils import init_qa_pairs

import utils.histogram_plot_qa_utils
reload(utils.histogram_plot_qa_utils)
from utils.histogram_plot_qa_utils import q_nbars_hist_plot_plotnums, q_stats_hists, q_gmm_ngaussians_hists

import utils.synthetic_fig_utils
reload(utils.synthetic_fig_utils)
from utils.synthetic_fig_utils import collect_plot_data_axes

# newer dealies
from utils.synthetic_fig_utils import reset_all_params, check_aspect, check_labels_titles_off_page, collect_boxes, update_fonts_boxes_overlap, set_all_seeds

# ----------------------------------------------
# Set up
fake_figs_dir = os.path.expanduser(fake_figs_dir)

# check directories
img_dir = fake_figs_dir + '/imgs/'
if not os.path.exists(img_dir):
    os.mkdir(img_dir)
    print('made:', img_dir)
json_dir = fake_figs_dir + '/jsons/'
if not os.path.exists(json_dir):
    os.mkdir(json_dir)
    print('made:', json_dir)


# get fonts -- see "cnn_create_synthetic_ticks" in FullProcess
dfont = pd.read_csv(fullproc_r + 'fonts.csv')

# check that location is there
drop_names = []
for fl in dfont['font location']:
    if not os.path.exists(fl):
        drop_names.append(False)
    else:
        drop_names.append(True)

font_names = dfont.loc[drop_names]['font name'].values

# for plot styles
plot_styles = plt.style.available

# stats for doing calculations
stats = {'minimum':np.min, 'maximum':np.max, 'median':np.median, 'mean':np.mean}

# some things to try
aspect_cut = {'min':0.3, 'max':4.0}

# get popular words for titles/axis
popular_nouns = get_popular_nouns(fullproc_r + 'data/')

# get inline math formulas
inlines = get_inline_math(fullproc_r,
                          recreate_inlines=False,
                         use_uniques=use_uniques)

# --------------------------------------------
# Plotting params

plot_params = plot_types_params.copy()

# prob for getting a histogram
plot_params['histogram']['prob'] = 1

# XYZ: since auto-setting plot_type later, can probably get away with none of the stuff in this cell
# probability of getting a scatter plot
plot_params['scatter']['prob'] = 0
# probability of getting a line
plot_params['line']['prob'] = 0
# prob of getting a contour plot
plot_params['contour']['prob'] = 0

linestyles_hist = ['-'] #, '--', ':'] # only use a subset of the linestyles

plot_params['histogram'] = plot_params['histogram'].copy()

# no horizontal plots
plot_params['histogram']['horizontal prob'] = 0.25

# random distributions
plot_params['histogram']['distribution']['random']['prob'] = 1

# gaussian mixture model
plot_params['histogram']['distribution']['gmm']['prob'] = 1
plot_params['histogram']['distribution']['gmm']['nclusters'] = {'min': 1, 'max': 5}
plot_params['histogram']['distribution']['gmm']['nsamples'] = {'min': 10, 'max': 50}

# linear distributions prob
plot_params['histogram']['distribution']['linear']['prob'] = 1

plot_params['histogram']['nbins'] = {'min':5, 'max':50} # number of bars
plot_params['histogram']['error bars']['elinewidth'] = {'min':1, 'max':3}

# redundent!
plot_type = 'histogram' # fix this

# for ease of things, lets not do equations here
title_params['equation']['prob'] = 0.25 # probability any word will be equation
xlabel_params['equation']['prob'] = 0.25 # probability any word will be equation
ylabel_params['equation']['prob'] = 0.25 # probability any word will be equation

# reload for debug
import utils.synthetic_fig_utils
reload(utils)
reload(utils.synthetic_fig_utils)
from utils.synthetic_fig_utils import normalize_params_prob
# normalize params
plot_params_out, panel_params, \
  title_params, xlabel_params, \
  ylabel_params = normalize_params_prob(plot_params.copy(), panel_params, 
                                        title_params, xlabel_params, 
                                        ylabel_params, colorbar_params)



# ----------------- RUN THE THING ---------------------
# some defaults
tight_layout = True
npanels, panel_style, nrows, ncols = 1, 'square', 1, 1 # keep single image
fontsize_min = 8 # minimum to try to shrink titles
aspect_cut = {'min':0.3, 'max':4.0} # start again if the square is within this
ifigure = 0

plt.ioff() # set off
#for ifigure in range(nHistTotal):
while ifigure < nHistTotal:
    print('')
    if verbose:
        print('*************** Figure', ifigure, '****************')
    # check if there for all formats
    hasFig = []
    for iformat in img_format:
        if os.path.exists(fake_figs_dir + 'imgs/Picture' + str(ifigure+1).zfill(6) + '.'+iformat):
            hasFig.append(iformat)
    # and json
    if os.path.exists(fake_figs_dir + 'jsons/Picture' + str(ifigure+1).zfill(6) + '.json'):
        hasFig.append('json')
    if (len(hasFig) == len(img_format) + 1) and not restart: # extra 1 for json
        if verbose:
            print('  already have:', fake_figs_dir + 'imgs/Picture' + str(ifigure+1).zfill(6) + '.<FMT>')
        ifigure += 1
        continue

    #import sys; sys.exit()
    # set all seeds for this plot
    rng_outer, rng, rng_titles, rng_font, rng_aspect = set_all_seeds(reset_outer = True, 
                                                                     reset_inner = True, 
                                                                     reset_titles=True, 
                                                                     reset_fonts = True, 
                                                                     reset_aspect = True, 
                                                                     verbose=verbose)

    # init plot params
    color_map = rng_outer.choice(plt.colormaps()) # choose a color map
    plot_style = rng_outer.choice(plot_styles) # choose a plotting style
    aspect_fig = rng_outer.uniform(low=aspect_fig_params['min'], high=aspect_fig_params['max'])
    dpi = int(rng_outer.uniform(low=dpi_params['min'], high=dpi_params['max']))

    # get all font stuffs
    title_fontsize, colorbar_fontsize, xlabel_fontsize, ylabel_fontsize, \
        xlabel_ticks_fontsize, ylabel_ticks_fontsize, \
                            csfont = get_font_info(fontsizes, font_names, rng=rng_titles)
    
    # get distribution type
    dist_params = plot_params_out[plot_type]['distribution'] 
    choices_d = []; probs_d = []
    for k,v in dist_params.items():
        choices_d.append(k)
        probs_d.append(v['prob'])

    distribution_type = np.random.choice(choices_d, p=probs_d)
    if verbose: print('Distribution Type:', distribution_type)

    # pull xmin/xmax for hists
    xmin,xmax = log_scale_ax()
    plot_params_out[plot_type]['xmin']=xmin
    plot_params_out[plot_type]['xmax']=xmax
    if verbose: print('xmin, xmax = ', xmin, xmax)

    # get plot data
    success_get_data = False
    while not success_get_data:
        data_for_plot = get_data(plot_params_out[plot_type], 
                                plot_type=plot_type, distribution=distribution_type, rng=rng)
        if len(data_for_plot['xs']) > 0 and plot_type == 'histogram':
            success_get_data = True

    ###### PLOT ########
    success_plot = False # overall plot
    # flags for various things
    success_titles = False
    # keep titles
    xlabels_pull = deepcopy(popular_nouns)
    ylabels_pull = deepcopy(popular_nouns)
    titles_pull = deepcopy(popular_nouns)
    # track figures
    # ifigure = 0
    itries = 0
    while not success_plot and itries <= itriesMax:
        itries += 1
        if itries >= itriesMax: # update everybody
            success_titles, rng_aspect, rng_titles, aspect_fig, xlabels_pull, ylabels_pull, \
                    titles_pull, rng_outer, color_map, plot_style, dist_params, distribution_type, \
                    title_fontsize, colorbar_fontsize, xlabel_fontsize, ylabel_fontsize, \
                    xlabel_ticks_fontsize, ylabel_ticks_fontsize, \
                                    csfont, dpi, xmin,xmax, \
                                        plot_params_in, data_for_plot = reset_all_params(plot_params_out, 
                                                                                         popular_nouns, 
                                                                                         plot_styles, 
                                                                                         fontsizes, 
                                                                                         font_names, 
                                                                                         dpi_params, aspect_fig_params, plot_type, rng)
            success_plot = False
            # get plot data
            success_get_data = False
            while not success_get_data:
                data_for_plot = get_data(plot_params_out[plot_type], 
                                        plot_type=plot_type, distribution=distribution_type, rng=rng)
                if len(data_for_plot['xs']) > 0 and plot_type == 'histogram':
                    success_get_data = True

        #seed = np.random.randint(maxint)
        #rng = np.random.default_rng(seed)
        #print('START: success_titles')
        while not success_titles: # ensure we catch any errors with bad fonts
            try:
                # make figure object
                #print('get fig axes level')
                fig, axes, plot_inds = make_base_plot(plot_style, color_map, dpi, nrows, ncols, aspect_fig,
                                base=2, verbose=True, tight_layout = tight_layout)
                # make plot based on data we already got
                #print('get data from plot level')
                data_from_plot, ax = make_plot(plot_params_out[plot_type], data_for_plot, 
                                        axes[0], plot_type=plot_type, linestyles=linestyles_hist, 
                                        rng=rng)
                # generate x/y labels and titles
                #print('get title x/y label level')
                title, xlabel, ylabel = add_titles_and_labels(axes[0], xlabels_pull, ylabels_pull, titles_pull, 
                                                        title_params, csfont, title_fontsize, 
                                    xlabel_params, ylabel_params, xlabel_fontsize, ylabel_fontsize,
                                    inlines, xlabel_ticks_fontsize, ylabel_ticks_fontsize,
                                    rng=rng_titles)
                #print('end of pulling vars') # HERE OK
                # set "pulls" to save, reset letter as needed
                try:
                    xlabels_pull = xlabel.get_text()
                except:
                    if type(xlabel) == type([]) or type(xlabel) == type('hi'):
                        xlabels_pull = xlabel
                    else:
                        lfkasjl
                try:
                    ylabels_pull = ylabel.get_text()
                except:
                    if type(ylabel) == type([]) or type(ylabel) == type('hi'):
                        ylabels_pull = ylabel
                    else:
                        flasj
                try:
                    titles_pull = title.get_text()
                except:
                    if type(title) == type([]) or type(title) == type('hi'):
                        titles_pull = title
                    else:
                        flasj
                # flag as success after render   
                #print('end of try except')  # HERE?
                success_titles = True
                # HERE is where the errors are occuring -- the SystemErrors null thing
                #plt.draw()
                #plt.pause(0.001)
            except Exception as e:
                plt.close('all')
                #print("HERE")
                if 'missing from font' in str(e):
                    print('[ERROR]: missing font (' + str(e) + '), will try with new font')
                    seed_font = np.random.randint(maxint)
                    rng_font = np.random.default_rng(seed_font)
                    try:
                        _, _, _, _, _, _, csfont = get_font_info(fontsizes, font_names, rng=rng_font)
                    except Exception as e2:
                        pass
                    success_titles = False
                elif 'Tight layout not applied' in str(e): # issue with tight layout, redo
                    print('[ERROR]: tight layout not applied - ', str(e))
                    success_titles, rng_aspect, rng_titles, aspect_fig, xlabels_pull, ylabels_pull, \
                            titles_pull, rng_outer, color_map, plot_style, dist_params, distribution_type, \
                            title_fontsize, colorbar_fontsize, xlabel_fontsize, ylabel_fontsize, \
                            xlabel_ticks_fontsize, ylabel_ticks_fontsize, \
                                            csfont, dpi, xmin,xmax, \
                                                plot_params_in, data_for_plot = reset_all_params(plot_params_out, 
                                                                                                popular_nouns, 
                                                                                                plot_styles, 
                                                                                                fontsizes, 
                                                                                                font_names, 
                                                                                                dpi_params, aspect_fig_params, plot_type, rng)
                else:
                    if verbose: print('[ERROR]: other error - ', str(e))
                    success_titles, rng_aspect, rng_titles, aspect_fig, xlabels_pull, ylabels_pull, \
                            titles_pull, rng_outer, color_map, plot_style, dist_params, distribution_type, \
                            title_fontsize, colorbar_fontsize, xlabel_fontsize, ylabel_fontsize, \
                            xlabel_ticks_fontsize, ylabel_ticks_fontsize, \
                                            csfont, dpi, xmin,xmax, \
                                                plot_params_in, data_for_plot = reset_all_params(plot_params_out, 
                                                                                                popular_nouns, 
                                                                                                plot_styles, 
                                                                                                fontsizes, 
                                                                                                font_names, 
                                                                                                dpi_params, aspect_fig_params, plot_type, rng)


        ###### SAVE FIG and collect bounding boxes #####
        plt.set_cmap(color_map) # do again
        #success_tight = True
        try:
            fig.tight_layout()
            #success_tight = False
        except Exception as e_tight2:
            if verbose: print('[ERROR]: tight layout 2 -- ', str(e_tight2))
            success_titles, rng_aspect, rng_titles, aspect_fig, xlabels_pull, ylabels_pull, \
                    titles_pull, rng_outer, color_map, plot_style, dist_params, distribution_type, \
                    title_fontsize, colorbar_fontsize, xlabel_fontsize, ylabel_fontsize, \
                    xlabel_ticks_fontsize, ylabel_ticks_fontsize, \
                                    csfont, dpi, xmin,xmax, \
                                        plot_params_in, data_for_plot = reset_all_params(plot_params_out, 
                                                                                         popular_nouns, 
                                                                                         plot_styles, 
                                                                                         fontsizes, 
                                                                                         font_names, 
                                                                                         dpi_params, aspect_fig_params, plot_type, rng)


        if not success_titles:
            continue

        # collect data
        try:
            fig.canvas.draw()
        except Exception as e_draw:
            if verbose: print('[ERROR]: in fig.canvas.draw -- ', str(e_draw))
            success_titles, rng_aspect, rng_titles, aspect_fig, xlabels_pull, ylabels_pull, \
                    titles_pull, rng_outer, color_map, plot_style, dist_params, distribution_type, \
                    title_fontsize, colorbar_fontsize, xlabel_fontsize, ylabel_fontsize, \
                    xlabel_ticks_fontsize, ylabel_ticks_fontsize, \
                                    csfont, dpi, xmin,xmax, \
                                        plot_params_in, data_for_plot = reset_all_params(plot_params_out, 
                                                                                         popular_nouns, 
                                                                                         plot_styles, 
                                                                                         fontsizes, 
                                                                                         font_names, 
                                                                                         dpi_params, aspect_fig_params, plot_type, rng)

        if not success_titles:
            continue

        width, height = fig.canvas.get_width_height()
        # save data
        datas = {}
        # figure datas
        success_fill_data = False
        datas['figure'] = {'dpi':dpi, 'base':base, 'aspect ratio': aspect_fig, 
                            'nrows':nrows, 'ncols':ncols, 
                            'plot style':plot_style, 
                            'color map':color_map,
                            'title fontsize':title_fontsize, 
                            'xlabel fontsize':xlabel_fontsize,
                            'ylabel fontsize':ylabel_fontsize, 
                        'plot indexes':plot_inds}
        try:
            for iplot,ax in enumerate(axes): ### XYZ, only 1 axis here
                datas['plot' + str(iplot)], err = collect_plot_data_axes(ax, fig,
                                height, width,
                                data_from_plot, data_for_plot, 
                                plot_type, title, 
                                xlabel, ylabel,
                                distribution_type, verbose=True, error_out=False)
            if not err: success_fill_data = True
        except Exception as e_fill_data:
            if verbose:
                print('[ERROR]: ' + str(e_fill_data))
            if 'Glyph' in str(e_fill_data) and 'missing' in str(e_fill_data): # missing a glyph, try different font
                _, _, _, rng_font, _ = set_all_seeds(reset_fonts = True, verbose=verbose)
            else: # no idea! reset everybody
                success_titles, rng_aspect, rng_titles, aspect_fig, xlabels_pull, ylabels_pull, \
                    titles_pull, rng_outer, color_map, plot_style, dist_params, distribution_type, \
                    title_fontsize, colorbar_fontsize, xlabel_fontsize, ylabel_fontsize, \
                    xlabel_ticks_fontsize, ylabel_ticks_fontsize, \
                                    csfont, dpi, xmin,xmax, \
                                        plot_params_in, data_for_plot = reset_all_params(plot_params_out, 
                                                                                         popular_nouns, 
                                                                                         plot_styles, 
                                                                                         fontsizes, 
                                                                                         font_names, 
                                                                                         dpi_params, aspect_fig_params, plot_type, rng)


        if not success_fill_data or not success_titles:
            continue
            

        ########### CHECKS -- titles off, bounding boxes, etc ##########
        # 1. Check for square with weird aspect ratio
        success_titles, success_aspect, aspect_fig = check_aspect(datas, success_titles, aspect_fig, aspect_cut, aspect_fig_params)
        if not success_aspect:
            continue

        # 2. Check if titles or x/y axis labels are running off the page         
        success_titles, xlabel_fontsize, ylabel_fontsize, title_fontsize, \
            xlabels_pull, ylabels_pull, titles_pull, rng_titles, \
                success_title_label, success_axis_labels = check_labels_titles_off_page(datas, width, height, success_titles, 
                                    xlabels_pull, ylabels_pull, titles_pull, 
                                    xlabel_fontsize, ylabel_fontsize, rng_titles, 
                                    popular_nouns, title_fontsize, fontsizes, font_names,
                                    fontsize_min = fontsize_min, verbose=verbose)

        if not success_title_label or not success_axis_labels:
            continue
                

        # 3. Save the fig, check if we have issues opening it
        # for diagnostics! -- move to after success of fig!
        success_save = False
        try:
            for iformat in img_format:
                fig.savefig(fake_figs_dir + 'imgs/Picture' + str(ifigure+1).zfill(6) + '.'+iformat, dpi=dpi)#, bbox_inches='tight')
                print('saved:', fake_figs_dir + 'imgs/Picture' + str(ifigure+1).zfill(6) + '.'+iformat)
            success_save = True
        except Exception as e_save:
            print('[ERROR]: could not save fig --', str(e_save))
            success_titles, rng_aspect, rng_titles, aspect_fig, xlabels_pull, ylabels_pull, \
                    titles_pull, rng_outer, color_map, plot_style, dist_params, distribution_type, \
                    title_fontsize, colorbar_fontsize, xlabel_fontsize, ylabel_fontsize, \
                    xlabel_ticks_fontsize, ylabel_ticks_fontsize, \
                                    csfont, dpi, xmin,xmax, \
                                        plot_params_in, data_for_plot = reset_all_params(plot_params_out, 
                                                                                         popular_nouns, 
                                                                                         plot_styles, 
                                                                                         fontsizes, 
                                                                                         font_names, 
                                                                                         dpi_params, aspect_fig_params, plot_type, rng)

        if not success_save or not success_titles:
            continue
        # check if issue opening plot
        e = ''
        try:
            for iformat in img_format:
                img = np.array(Image.open(fake_figs_dir + 'imgs/Picture' + str(ifigure+1).zfill(6) + '.'+iformat))
        except Exception as e:
            #success_titles = False
            # redo_gen_fig = True
            # redo_gen_plot = True
            success_titles, rng_aspect, rng_titles, aspect_fig, xlabels_pull, ylabels_pull, \
                    titles_pull, rng_outer, color_map, plot_style, dist_params, distribution_type, \
                    title_fontsize, colorbar_fontsize, xlabel_fontsize, ylabel_fontsize, \
                    xlabel_ticks_fontsize, ylabel_ticks_fontsize, \
                                    csfont, dpi, xmin,xmax, \
                                        plot_params_in, data_for_plot = reset_all_params(plot_params_out, 
                                                                                         popular_nouns, 
                                                                                         plot_styles, 
                                                                                         fontsizes, 
                                                                                         font_names, 
                                                                                         dpi_params, aspect_fig_params, plot_type, rng)
            if verbose: 
                print('[ERROR]: Issue with opening image!')
                if str(e) != '': print('Full error:', str(e))

        if not success_titles:
            continue


        # 4. Check if any bounding boxes are overlapping
        success_boxes, boxes_check, names_overlap = collect_boxes(datas, grace_ticks=grace_ticks)

        if save_diagnostic_plot:
            img_diag = np.array(Image.open(fake_figs_dir + 'imgs/Picture' + str(ifigure+1).zfill(6) + '.'+img_format[0]).convert('RGB'))
            imgplot = add_annotations(img_diag, img_diag, deepcopy(datas))            

        if not success_boxes: # check who overlaps with who
            success_titles, xlabel_fontsize, ylabel_fontsize, ylabel_ticks_fontsize, \
                xlabel_ticks_fontsize, title_fontsize, xlabels_pull, \
                    ylabels_pull, titles_pull, rng_titles = update_fonts_boxes_overlap(names_overlap, success_titles, rng_titles,popular_nouns,
                                    xlabels_pull, ylabels_pull, titles_pull,
                                xlabel_ticks_fontsize, ylabel_ticks_fontsize, 
                                xlabel_fontsize, ylabel_fontsize, title_fontsize,
                                verbose=verbose, fontsize_min=fontsize_min)
            continue


        success_plot = True # hurray if we've made it this far!
        # if all went well, reset seed
        seed = np.random.randint(maxint)

        print('SUCCESS')
        plt.close(fig)

        ######## GENERATE QA BASE #######
        qa_pairs = init_qa_pairs()
        for iplot in range(len((axes))):
            if datas['plot'+str(iplot)]['type'] == 'histogram':
                ############ L1 #############
                # number of bars
                if len(axes) > 1:
                    qa_pairs = q_nbars_hist_plot_plotnums(datas, qa_pairs, plot_num = iplot, use_words=False, verbose=verbose_qa)
                qa_pairs = q_nbars_hist_plot_plotnums(datas, qa_pairs, plot_num = iplot, use_words=True, verbose=verbose_qa)

                ###### L2 #######
                # stats items
                for k,v in stats.items():
                    if len(axes) > 1:
                        qa_pairs = q_stats_hists(datas, qa_pairs, stat={k:v}, plot_num=iplot, use_words=False, verbose=verbose_qa)
                    qa_pairs = q_stats_hists(datas, qa_pairs, stat={k:v}, plot_num=iplot, use_words=True, verbose=verbose_qa)

                ###### L3 ######
                # if GMM -- how many gaussians?
                hasGMM = False
                if 'data params' not in datas['plot'+str(iplot)]['data']:
                    #print('Not a gmm relationship!')
                    pass
                else:
                    if datas['plot'+str(iplot)]['distribution'] == 'gmm':
                        hasGMM = True
                if hasGMM:       
                    qa_pairs = q_gmm_ngaussians_hists(datas, qa_pairs, plot_num=iplot, use_words=True, verbose=verbose_qa)

        datas['VQA'] = qa_pairs


        # dump full data
        dumped = json.dumps(datas, cls=NumpyEncoder)
        with open(fake_figs_dir + 'jsons/Picture' + str(ifigure+1).zfill(6) + '.json', 'w') as f:
            json.dump(dumped, f)


    if success_plot:
        print('DONE MAKING PLOT!')
        plt.close('all') # won't work for parallel
        ifigure += 1
    elif itries >= itriesMax:
        print("--- maxed out tries, plot failed ---")
    else:
        import sys; sys.exit()

    # either way, reset
    rng_outer, rng, rng_titles, rng_font, rng_aspect = set_all_seeds(reset_outer = True, 
                                                                     reset_inner = True, 
                                                                     reset_titles=True, 
                                                                     reset_fonts = True, 
                                                                     reset_aspect = True, 
                                                                     verbose=verbose)
    success_titles = False
