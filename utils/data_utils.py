import numpy as np
import json
from scipy.stats import loguniform

from utils.distribution_utils import get_random_data, \
   get_linear_data, get_gmm_data 


# for saving numpy arrays
# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return 'non serializable entry'
        return json.JSONEncoder.default(self, obj)

##########################################
############## CONTOUR DATA ###############
##########################################
from .plot_classes_utils import Contour

def get_contour_data(plot_params, distribution = 'random',
                     verbose=True, rng=np.random, **kwargs):
    """
    plot_params : directory with plot params
    """
    contour = Contour()
    for k,v in kwargs.items():
        if k in contour.__dict__: # in there
            setattr(contour, k, v)

    data_params = {}
    #print('contour', contour.__dict__)
    if contour.nx is None:
        nx = int(round(rng.uniform(low=plot_params['npoints']['nx']['min'], 
                                          high=plot_params['npoints']['nx']['max'])))
    else:
        nx = contour.nx
    if contour.ny is None:
        ny = int(round(rng.uniform(low=plot_params['npoints']['ny']['min'], 
                                          high=plot_params['npoints']['ny']['max'])))
    else:
        ny = contour.ny
    
    xmin,xmax = plot_params['xmin'],plot_params['xmax']
    ymin,ymax = plot_params['ymin'],plot_params['ymax']

    x1 = rng.uniform(low=xmin, high=xmax)
    x2 = rng.uniform(low=xmin, high=xmax)
    if x1<x2:
        xmin = x1; xmax = x2
    else:
        xmin=x2; xmax=x1
    y1 = rng.uniform(low=ymin, high=ymax)
    y2 = rng.uniform(low=ymin, high=ymax)
    if y1<y2:
        ymin = y1; ymax = y2
    else:
        ymin=y2; ymax=y1
    c1 = rng.uniform(low=plot_params['colors']['min'], 
                           high=plot_params['colors']['max'])
    c2 = rng.uniform(low=plot_params['colors']['min'], 
                           high=plot_params['colors']['max'])
    if c1<c2:
        cmin = c1; cmax = c2
    else:
        cmin=c2; cmax=c1

    if distribution == 'random':
        #print(xmin,xmax)
        xs,ys,colors = get_random_data('contour',xmin,xmax,ymin,ymax,
                                  npoints=(nx,ny),
                                  zmin=cmin, zmax=cmax, rng=rng) 
    elif distribution == 'linear':
        xs,ys,colors, data_params = get_linear_data('contour',plot_params['distribution'][distribution],
                                                    xmin,xmax,ymin,ymax,
                                  npoints=(nx,ny), rng=rng)#,
                                  #zmin=cmin, zmax=cmax) 
    elif distribution == 'gmm':
        xs,ys,colors, data_params = get_gmm_data('contour',
                                                 plot_params['distribution'][distribution],
                                                    xmin,xmax,ymin,ymax,
                                  npoints=(nx,ny),
                                  #zmin=cmin, zmax=cmax) 
                                  cmin=cmin, cmax=cmax, rng=rng, function=rng.uniform) 

    # do we have x/y error bars?
    hasXErr = False; hasYErr = False
    xerr,yerr = [],[] # x/y error maybe later

    return xs, ys, colors, xerr, yerr, data_params



############################################################
######################## LINES DATA ########################
############################################################

from .plot_classes_utils import Line

def get_line_data(plot_params, npoints,nlines,xmin=0, xmax=1, ymin=0, ymax=1, 
                  #prob_same_x=0.1,
                  xordered=True, verbose=True,
                 pick_xrange=True, pick_yrange=True,
                 distribution='random', rng=np.random, **kwargs):
    """
    ymin/ymax : can be a number of a list, if list, needs to be matched up with number of nlines
    xordered : do we want to put the x-points in a monotonic order?
    """
    line = Line()
    for k,v in kwargs.items():
        if k in line.__dict__: # in there
            setattr(line, k, v)

    if line.prob_same_x is None:
        prob_same_x = plot_params['prob same x']
    else:
        prob_same_x = line.prob_same_x

    data_params = {}
    if (type(ymin) not in [list, np.ndarray]) and (type(ymax) not in [list, np.ndarray]):
        pass
    elif ((type(ymin) not in [list, np.ndarray]) and (type(ymax) in [list, np.ndarray])):
        if verbose: print('type of ymin and ymax must be the same!  Will fall back on ints')
        ymax = ymax[0]
    elif ((type(ymin) in [list, np.ndarray]) and (type(ymax) not in [list, np.ndarray])):
        if verbose: print('type of ymin and ymax must be the same!  Will fall back on ints')
        ymin = ymin[0]
    elif len(ymin) != nlines or len(ymax) != nlines:
        if verbose: print('length of ymax, ymin not the same as "nlines", falling back on ints')
        ymin = ymin[0]; ymax = ymax[0]

    nlines = int(round(rng.uniform(low=nlines['min'], high=nlines['max'])))
    npoints = int(round(rng.uniform(low=npoints['min'], high=npoints['max'])))

    # pick randomly
    if pick_xrange:
        x1 = rng.uniform(low=xmin, high=xmax)
        x2 = rng.uniform(low=xmin, high=xmax)
        if x1<x2:
            xmin = x1; xmax = x2
        else:
            xmin=x2; xmax=x1
    if pick_yrange:
        y1 = rng.uniform(low=ymin, high=ymax)
        y2 = rng.uniform(low=ymin, high=ymax)
        if y1<y2:
            ymin = y1; ymax = y2
        else:
            ymin=y2; ymax=y1

    # into arrays
    if (type(ymin) not in [list, np.ndarray]) and (type(ymax) not in [list, np.ndarray]):
        ymin = np.repeat(ymin,nlines)
        ymax = np.repeat(ymax,nlines)

    if distribution == 'random':
        xs,ys = get_random_data('line',xmin,xmax,ymin,ymax,
                    prob_same_x=prob_same_x, 
                            nlines=nlines, npoints=npoints, rng=rng)
    elif distribution == 'linear':
        #print("HI IN GET LINE DATA")
        xs,ys,data_params = get_linear_data('line', plot_params['distribution'][distribution],
                                xmin,xmax,ymin,ymax,
                    prob_same_x=prob_same_x, 
                            nlines=nlines, npoints=npoints, rng=rng)
    elif distribution == 'gmm': # gaussian mixture model
        xs,ys,data_params = get_gmm_data('line', plot_params['distribution'][distribution],
                                xmin,xmax,ymin,ymax,
                    prob_same_x=prob_same_x, 
                            nlines=nlines, npoints=npoints, rng=rng, function=rng.uniform)
        # have to update for binning
        npoints = len(xs[0])
    else:
        print("don't know how to deal with this line distribution!")
        import sys; sys.exit()

    # do we have x/y error bars?
    hasXErr = False; hasYErr = False
    xerrs,yerrs = [],[]

    # for error rate
    #scale = 
    if rng.uniform(0,1) <= plot_params['error bars']['x']['prob']: # yes
        hasXErr = True
        scale = np.max(xmax)-np.min(xmin)
        for i in range(nlines):
            xerr = rng.uniform(low=plot_params['error bars']['x']['size']['min']*scale, 
                                 high=plot_params['error bars']['x']['size']['max']*scale,
                                 size=len(xs[i]))
            xerrs.append(xerr)
    if rng.uniform(0,1) <= plot_params['error bars']['y']['prob']: # yes
        hasYErr = True
        scale = np.max(ymax)-np.min(ymin)
        for i in range(nlines):
            yerr = rng.uniform(low=plot_params['error bars']['y']['size']['min']*scale, 
                                 high=plot_params['error bars']['y']['size']['max']*scale,
                                 size=len(xs[i]))
            yerrs.append(yerr)
    
    return xs, ys, xerrs, yerrs, data_params


############################################################
######################## SCATTER DATA ########################
############################################################

def get_scatter_data(plot_params, distribution = 'random',
                     verbose=True, rng=np.random):
    """
    plot_params : directory with plot params
    """

    data_params = {}
    npoints = int(round(rng.uniform(low=plot_params['npoints']['min'], 
                                          high=plot_params['npoints']['max'])))
    #print('npoints here:', npoints)
    
    xmin,xmax = plot_params['xmin'],plot_params['xmax']
    ymin,ymax = plot_params['ymin'],plot_params['ymax']


    x1 = rng.uniform(low=xmin, high=xmax)
    x2 = rng.uniform(low=xmin, high=xmax)
    if x1<x2:
        xmin = x1; xmax = x2
    else:
        xmin=x2; xmax=x1
    y1 = rng.uniform(low=ymin, high=ymax)
    y2 = rng.uniform(low=ymin, high=ymax)
    if y1<y2:
        ymin = y1; ymax = y2
    else:
        ymin=y2; ymax=y1
    c1 = rng.uniform(low=plot_params['colors']['min'], 
                           high=plot_params['colors']['max'])
    c2 = rng.uniform(low=plot_params['colors']['min'], 
                           high=plot_params['colors']['max'])
    if c1<c2:
        cmin = c1; cmax = c2
    else:
        cmin=c2; cmax=c1

    if distribution == 'random':
        xs,ys,colors = get_random_data('scatter',xmin,xmax,ymin,ymax,
                                  npoints=npoints,
                                  cmin=cmin, cmax=cmax, rng=rng)
    elif distribution == 'linear':
        xs,ys,colors,data_params = get_linear_data('scatter', 
                                    plot_params['distribution'][distribution],
                                    xmin,xmax,ymin,ymax,
                                    cmin=cmin,cmax=cmax,
                                    npoints=npoints, rng=rng)
    elif distribution == 'gmm':
        xs,ys,colors,data_params = get_gmm_data('scatter', 
                                    plot_params['distribution'][distribution],
                                    xmin,xmax,ymin,ymax,
                                    cmin=cmin,cmax=cmax,
                                    npoints=npoints, rng=rng, function=rng.uniform)
        # have to update npoints based on how things where sampled??
        npoints = len(xs)
    else:
        print('no such distribution in "get_scatter_data"!')
        import sys; sys.exit()


    # do we have x/y error bars?
    hasXErr = False; hasYErr = False
    xerr,yerr = [],[]

    if rng.uniform(0,1) <= plot_params['error bars']['x']['prob']: # yes
        hasXErr = True
        scale = xmax-xmin
        xerr = rng.uniform(low=plot_params['error bars']['x']['size']['min']*scale, 
                                 high=plot_params['error bars']['x']['size']['max']*scale,
                                 size=len(xs))
    if rng.uniform(0,1) <= plot_params['error bars']['y']['prob']: # yes
        hasYErr = True
        scale = ymax-ymin
        yerr = rng.uniform(low=plot_params['error bars']['y']['size']['min']*scale, 
                                 high=plot_params['error bars']['y']['size']['max']*scale,
                                 size=len(xs))

    #print(xerr)
    #print(yerr)
    #if data_params == {}:
    #    print('missing data params!!!!!!!')
    return xs, ys, colors, xerr, yerr, data_params



################################################
############ HISTOGRAM PLOTS: PLOT ############
################################################

def get_histogram_data(plot_params, distribution = 'random',
                     verbose=True, rng=np.random):
    """
    plot_params : plot parameters
    """

    data_params = {}
    npoints = int(round(rng.uniform(low=plot_params['npoints']['min'], 
                                          high=plot_params['npoints']['max'])))
    
    xmin,xmax = plot_params['xmin'],plot_params['xmax']

    x1 = rng.uniform(low=xmin, high=xmax)
    x2 = rng.uniform(low=xmin, high=xmax)
    if x1<x2:
        xmin = x1; xmax = x2
    else:
        xmin=x2; xmax=x1

    ys = []
    if distribution == 'random':
        xs = get_random_data('histogram',xmin,xmax,
                                  npoints=npoints, rng=rng)
    elif distribution == 'linear':
        xs, data_params = get_linear_data('histogram',plot_params['distribution'][distribution],
                                          xmin,xmax,
                                  npoints=npoints, rng=rng)
    elif distribution == 'gmm':
        xs, data_params = get_gmm_data('histogram',plot_params['distribution'][distribution],
                                          xmin,xmax,
                                  npoints=npoints, rng=rng, function=rng.uniform)
    else:
        print('no distribution in "get_histogram_data"!')
        import sys; sys.exit()
    
    # do we have x/y error bars?
    hasXErr = False; hasYErr = False
    # xerr,yerr = [],[]

    if rng.uniform(0,1) <= plot_params['error bars']['x']['prob']: # yes
        hasXErr = True
    #     #scale = xmax-xmin
    #     scale = xs
    #     xerr = rng.uniform(low=plot_params['error bars']['x']['size']['min'], 
    #                              high=plot_params['error bars']['x']['size']['max'],
    #                              size=npoints)*scale
        
    if rng.uniform(0,1) < plot_params['horizontal prob']: # probability that we have a horizontal bar plot
        # flip
        ys = xs.copy()
        #yerr = xerr.copy()
        #xs = []; ys = []
        hasYErr = True
        hasXErr = False

    return xs, ys, hasXErr, hasYErr, data_params # this is different than other plots!



###########################################################################
######################## MAIN GET DATA #######################
###########################################################################

##### GENERIC DATA AND PLOTS #####
def get_data(plot_params, plot_type='line', distribution='random', #npoints = 100, xmin=0, xmax=1, ymin=0, ymax=1, 
             #xordered=True, nlines=1, # line params
             verbose=True, xordered=True, # general params
             rng=None, # random number generation
             **kwargs
            ):
    if rng is None:
        rng = np.random
    if plot_type == 'line':
        xs, ys, xerrs,yerrs, data_params = get_line_data(plot_params,  
                                            plot_params['npoints'], # this bad coding, should just pass plot_params
                              plot_params['nlines'],
                              xmin=plot_params['xmin'], 
                              xmax=plot_params['xmax'], 
                              ymin=plot_params['ymin'], 
                              ymax=plot_params['ymax'], 
                              #prob_same_x=plot_params['prob same x'],
                              xordered=xordered, 
                              verbose=verbose, 
                                            distribution=distribution,
                                            rng=rng, **kwargs)
        data = {'xs':xs, 'ys':ys}
        if len(xerrs) > 0:
            data['xerrs'] = xerrs
        if len(yerrs) > 0:
            data['yerrs'] = yerrs
        if data_params != {}:
            data['data params'] = data_params
        return data
    elif plot_type == 'scatter':
        xs, ys, colors_scatter,xerr,yerr, data_params = get_scatter_data(plot_params,
                                                           distribution=distribution, 
                                                           rng=rng,
                                                           verbose=verbose)
        data = {'xs':xs, 'ys':ys, 'colors':colors_scatter}
        if len(xerr) > 0:
            data['xerrs'] = xerr
        if len(yerr) > 0:
            data['yerrs'] = yerr
        if data_params != {}:
            data['data params'] = data_params
        else:
            pass
            #print('in get_data --- NO DATA PARAMS')
        return data
    elif plot_type == 'histogram':
        xs,ys,hasXErr,hasYErr, data_params = get_histogram_data(plot_params,
                                                   distribution=distribution,
                                                   rng=rng,
                                                           verbose=verbose)
        data = {'xs':xs, 'ys':ys}
        #if len(xerr) > 0:
        if hasXErr:
            data['xerrs'] = hasXErr # different!
        #if len(yerr) > 0:
        if hasYErr:
            data['yerrs'] = hasYErr # different!
        if data_params != {}:
            data['data params'] = data_params
        return data
    elif plot_type == 'contour':
        xs, ys, color_grid,xerr,yerr, data_params = get_contour_data(plot_params,
                                                       distribution=distribution, 
                                                       rng=rng,
                                                           verbose=verbose, 
                                                           **kwargs)
        data = {'xs':xs, 'ys': ys, 'colors':color_grid}
        if len(xerr) > 0:
            data['xerrs'] = xerr
        if len(yerr) > 0:
            data['yerrs'] = yerr
        if data_params != {}:
            data['data params'] = data_params
        return data
    else:
        print('not implement for this plot type!', 'Plot type:', plot_type)
        import sys; sys.exit()



