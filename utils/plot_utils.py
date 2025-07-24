# utilities to create plots of different kinds
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}' #for \text command
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from PIL import ImageColor

# classes
from .plot_classes_utils import Histogram, Line

# for scatter plot markers
from matplotlib.lines import Line2D
marker_dir = Line2D.markers

# for astro coord systems
#from astropy.wcs import WCS

markers = []
for m,mn in marker_dir.items():
    if type(m) == str:
        if 'None' not in m.lower() and 'nothing' not in mn.lower():
            #print(m, mn)
            markers.append(m)
    else:
        #print(m,mn)
        markers.append(m)

markers = np.array(markers,dtype=object)

# for line styles
from utils.synthetic_fig_utils import get_line_styles
linestyles = get_line_styles()

# for colors
# how many random colors to generate?
# e.g. colors = colors_(6)
#colors_ = lambda n: list(map(lambda i: "#" + "%06x" % rng.integer(0, 0xFFFFFF),range(n)))
# JPN: should really be picked with a seed but leaving as is for now
colors_ = lambda n: list(map(lambda i: "#" + "%06x" % np.random.randint(0, 0xFFFFFF),range(n)))


# LINES PLOT
def get_line_plot(plot_params, data, ax, linestyles=linestyles, rng=np.random, **kwargs):
    """
    kwargs allows for passing of parameters to be set
    """
    line = Line()
    for k,v in kwargs.items():
        if k in line.__dict__: # in there
            setattr(line, k, v)

    datas = []
    linestyles_here = []; linethicks_here = []; markers_here = []
    marker_sizes_here = []
    xerrs = []; yerrs = []
    if line.hasMarker is None: 
        hasMarker = False
        p = rng.uniform(0,1)
        if p <= plot_params['markers']['prob']:
            hasMarker = True
    else:
        hasMarker = line.hasMarker
    colors_here = []

    if line.elinewidth is None:
        elinewidth = int(round(rng.uniform(low=plot_params['error bars']['elinewidth']['min'], 
                                            high=plot_params['error bars']['elinewidth']['max'])))
    # draw lines
    #xerrs = []; yerrs = []
    # a few tests
    for attr in ['markers', 'lthicks', 'linestyles', 'marker_sizes', 'linecolors']:
        if getattr(line, attr) is not None:
            if len(getattr(line, attr)) != len(data['ys']) or type(getattr(line, attr)) != type([]):
                print('wrong length for '+attr+', setting to default')
                setattr(line, attr, None)


    for i in range(len(data['ys'])):
        if line.marker is None and line.markers is None:
            marker = rng.choice(markers)
        elif line.marker is not None:
            marker = line.marker
        elif line.markers is not None:
            marker = line.markers[i]

        if line.lthick is None and line.lthicks is None:
            lthick = rng.uniform(low=plot_params['line thick']['min'], 
                                   high=plot_params['line thick']['max'])
        elif line.lthick is not None:
            lthick = line.lthick
        elif line.lthicks is not None:
            lthick = line.lthicks[i]

        # choose random linestyle
        if line.linestyle is None and line.linestyles is None:
            linestyle = rng.choice(linestyles)
        elif line.linestyle is not None:
            linestyle = line.linestyle
        elif line.linestyles is not None:
            linestyle = line.linestyles[i]

        if line.linecolor is None and line.linecolors is None:
            try:
                linecolor = ImageColor.getcolor(data_here.get_color(), "RGBA")
                #cols.append(linecolor)
            except:
                #cols.append( (0,0,0) ) # I assume
                linecolor = (0,0,0)
            linecolor = np.array(linecolor)/255.
        elif line.linecolor is not None:
            linecolor = line.linecolor
        elif line.linecolors is not None:
            linecolor = line.linecolors[i]

        if hasMarker:
            if line.marker_size is None and line.marker_sizes is None:
                marker_size = int(round(rng.uniform(low=plot_params['markers']['size']['min'],
                                            high=plot_params['markers']['size']['min'])))
            elif line.marker_size is not None:
                marker_size = line.marker_size
            elif line.marker_sizes is not None:
                marker_size = line.marker_sizes[i]
            data_here, = ax.plot(data['xs'][i],data['ys'][i], linewidth=lthick, 
                                 linestyle = linestyle, marker=marker,
                                markersize=marker_size, color=linecolor)
        else:
            data_here, = ax.plot(data['xs'][i],data['ys'][i], linewidth=lthick, 
                                 linestyle = linestyle, color=linecolor)
            marker = ''
            marker_size = -1

        #cols = []
        plt.draw()
        #print('DRAW WAS CALLED')
        # try:
        #     cols.append(ImageColor.getcolor(data_here.get_color(), "RGBA"))
        # except:
        #     cols.append( (0,0,0) ) # I assume
        # cols = np.array(cols)/255.

        if 'xerrs' in data:# and 'yerrs' not in data: # have x-errors
            (_, caps, bars) = ax.errorbar(data['xs'][i],data['ys'][i],xerr=data['xerrs'][i],
                                         linewidth=0,elinewidth=elinewidth,
                                         markersize=0, ecolor=linecolor, zorder=0)
            xerrs.append(bars)
        if 'yerrs' in data:# and 'xerrs' not in data: # have x-errors
            (_, caps, bars) = ax.errorbar(data['xs'][i],data['ys'][i],yerr=data['yerrs'][i],
                                         linewidth=0, elinewidth=elinewidth,
                                         markersize=0, ecolor=linecolor, zorder=0)
            yerrs.append(bars)
        
        linethicks_here.append(lthick)
        linestyles_here.append(linestyle)
        markers_here.append(marker)
        datas.append(data_here)
        marker_sizes_here.append(marker_size)
        colors_here.append(linecolor)

    data_out = {'data':datas, 'plot params':{'linethick':linethicks_here, 
                                            'linestyles':linestyles_here,
                                            'markers':markers_here,
                                            'marker size':marker_sizes_here,
                                            'colors':colors_here}
               }
    # add in x/y errors, if present
    if 'xerrs' in data:
        data_out['x error bars'] = xerrs
    if 'yerrs' in data:
        data_out['y error bars'] = yerrs
    return data_out, ax




# SCATTERS: PLOTS
def get_scatter_plot(plot_params, data, ax, rng=np.random):
    p = rng.uniform(0,1)
    cax = []; side = ''
    marker = rng.choice(markers)
    marker_size = int(round(rng.uniform(low=plot_params['markers']['size']['min'],
                                    high=plot_params['markers']['size']['min'])))
    if not p <= plot_params['colormap scatter']['prob']: # not have color map
        data_here = ax.scatter(data['xs'],data['ys'],marker=marker, s=marker_size) 
    else:
        data_here = ax.scatter(data['xs'],data['ys'], 
                               c=data['colors'],marker=marker, 
                              s=marker_size) # need to add color
        divider = make_axes_locatable(ax)

        # get probs
        probs = []; choices = []
        for k,v in plot_params['color bar']['location probs'].items():
            probs.append(v); choices.append(k)
        side = rng.choice(choices, p=probs)
        size = rng.uniform(low=plot_params['color bar']['size percent']['min'], 
                     high=plot_params['color bar']['size percent']['max'])
        size = str(int(round(size*100)))+'%'

        pad = rng.uniform(low=plot_params['color bar']['pad']['min'], 
                                 high=plot_params['color bar']['pad']['max'])

        cax = divider.append_axes(side, size=size, pad=pad)
        # the side of the axis
        if side == 'right': # this maybe should become a random selection?
            axis_side = 'right'
            cax.yaxis.set_ticks_position(axis_side)
        elif side == 'left':
            axis_side = 'left'
            cax.yaxis.set_ticks_position(axis_side)
        elif side == 'top':
            axis_side = 'top'
            cax.xaxis.set_ticks_position(axis_side)
        elif side == 'bottom':
            axis_side = 'bottom'
            cax.xaxis.set_ticks_position(axis_side)

    xerrs = []; yerrs = []
    plt.draw()
    #print('DRAW WAS CALLED 2')
    cols = data_here.get_facecolors()
    elinewidth = int(round(rng.uniform(low=plot_params['error bars']['elinewidth']['min'], 
                                                 high=plot_params['error bars']['elinewidth']['max'])))
    if 'xerrs' in data:# and 'yerrs' not in data: # have x-errors
        cols_scatter = cols.reshape(-1,4)#*255
        # print('cols scatter:', cols_scatter.shape)
        # print('cols type:', cols_scatter.dtype)
        # print('cols min/max:', np.min(cols_scatter), np.max(cols_scatter))
        # print('xs:', data['xs'].shape)
        # print('ys:', data['ys'].shape)
        # print('xerrs:', data['xerrs'].shape)
        #cols_scatter = cols_scatter.astype('int')
        cols_scatter[cols_scatter<0] = 0
        cols_scatter[cols_scatter>1] = 1     
        cols_scatter = cols_scatter.astype('float')  
        try:
            (_, caps, bars) = ax.errorbar(data['xs'],data['ys'],xerr=data['xerrs'],
                                     linewidth=0,elinewidth=elinewidth,
                                     markersize=0, 
                                      ecolor=cols_scatter, zorder=0)

        except Exception as e:
            print('Issue with colors in xerrs:')
            print(e)
            (_, caps, bars) = ax.errorbar(data['xs'],data['ys'],xerr=data['xerrs'],
                                     linewidth=0,elinewidth=elinewidth,
                                     markersize=0, zorder=0)
            print('data shape - xs,ys,xerrs:', data['xs'].shape, data['ys'].shape, 
                  data['xerrs'].shape)
            print('colors shape:', cols_scatter.shape)
            print('colors min/max:', cols_scatter.min(), cols_scatter.max())
            
        xerrs.append(bars)
    if 'yerrs' in data:# and 'xerrs' not in data: # have x-errors
        cols_scatter = cols.reshape(-1,4)#*255
        # print('cols scatter:', cols_scatter.shape)
        # print('cols type:', cols_scatter.dtype)
        # print('cols min/max:', np.min(cols_scatter), np.max(cols_scatter))
        # print('xs:',data['xs'].shape)
        # print('ys:',data['ys'].shape)
        # print('yerrs:',data['yerrs'].shape)
        #cols_scatter = cols_scatter.astype('int')
        # make sure no min/max
        cols_scatter[cols_scatter<0] = 0
        cols_scatter[cols_scatter>1] = 1
        try:
            (_, caps, bars) = ax.errorbar(data['xs'],data['ys'],yerr=data['yerrs'],
                                     linewidth=0, elinewidth=elinewidth,
                                     markersize=0, 
                                      ecolor=cols_scatter, zorder=0)
        except Exception as e:
            print('Issue with colors in yerrs:')
            print(e)
            (_, caps, bars) = ax.errorbar(data['xs'],data['ys'],yerr=data['yerrs'],
                                     linewidth=0, elinewidth=elinewidth,
                                     markersize=0, zorder=0)
            print('data shape - xs,ysyxerrs:', data['xs'].shape, data['ys'].shape, 
                  data['yerrs'].shape)
            print('colors shape:', cols_scatter.shape)
            print('colors min/max:', cols_scatter.min(), cols_scatter.max())
        yerrs.append(bars)
        
    # save data
    if cax != []:
        data_out = {'data':data_here, 'color bar':cax, 'marker':marker, 'marker size':marker_size,
                    'color bar params':{'side':side, 'pad':pad, 'size':size, 
                                       'axis side':axis_side, 
                                       'label prob': plot_params['color bar']['label prob']
                                       }
                   }
    else:
        data_out = {'data':data_here, 'marker':marker, 'marker size':marker_size}

    # add in x/y errors, if present
    if 'xerrs' in data:
        data_out['x error bars'] = xerrs
    if 'yerrs' in data:
        #print("YESS TO Y IN DATA")
        data_out['y error bars'] = yerrs
    if 'xerrs' in data or 'yerrs' in data:
        data_out['error bar params']:{'elinewidth':elinewidth}
    
    return data_out, ax



def get_contour_plot(plot_params, data, ax, rng=np.random):
    p = rng.uniform(0,1) # probability that has a colorbar
    #pi = rng.uniform(0,1) # probability that is an image (vs a contour with lines)
    choices = []; probs = []
    for k,v in plot_params['image or contour']['prob'].items():
        choices.append(k)
        probs.append(v)
    plot_type = rng.choice(choices, p=probs)
    cax = []; side = ''
    
    if plot_type == 'contour':
        nlevels = int(round(rng.uniform(low=plot_params['nlines']['min'],
                                              high=plot_params['nlines']['max'])))
        data_here2 = ax.contour(data['xs'], data['ys'], data['colors'], nlevels)
        data_here = {'contour':data_here2}
    elif plot_type == 'image':
        real_x = data['xs']
        real_y = data['ys']
        dx = (real_x[1]-real_x[0])/2.
        dy = (real_y[1]-real_y[0])/2.
        extent = [real_x[0]-dx, real_x[-1]+dx, real_y[0]-dy, real_y[-1]+dy]
        #plt.imshow(data, extent=extent)
        data_here1 = ax.imshow(data['colors'], extent=extent)
        data_here = {'image':data_here1}
    elif plot_type == 'both':
        pg = rng.uniform(0,1)
        grayContours = False
        if pg <= plot_params['image or contour']['both contours']['prob gray']: # probability that contours are gray for "both" situation
            grayContours = True
        cmap = rng.choice(['gray', 'gray_r'])
        real_x = data['xs']
        real_y = data['ys']
        dx = (real_x[1]-real_x[0])/2.
        dy = (real_y[1]-real_y[0])/2.
        extent = [real_x[0]-dx, real_x[-1]+dx, real_y[0]-dy, real_y[-1]+dy]
        data_here1 = ax.imshow(data['colors'], extent=extent)
        nlevels = int(round(rng.uniform(low=plot_params['nlines']['min'],
                                              high=plot_params['nlines']['max'])))
        if not grayContours:
            data_here2 = ax.contour(data['xs'], data['ys'], data['colors'], nlevels)
        else:
            data_here2 = ax.contour(data['xs'], data['ys'], data['colors'], nlevels,
                                   cmap=cmap)

        data_here = {'image':data_here1, 'contour':data_here2}
    else:
        print('not supported plot type!')
        import sys; sys.exit()

    if not p <= plot_params['colormap contour']['prob']: # not have color map
        pass 
    else:
        divider = make_axes_locatable(ax)

        # get probs
        probs = []; choices = []
        for k,v in plot_params['color bar']['location probs'].items():
            probs.append(v); choices.append(k)
        side = rng.choice(choices, p=probs)
        size = rng.uniform(low=plot_params['color bar']['size percent']['min'], 
                     high=plot_params['color bar']['size percent']['max'])
        size = str(int(round(size*100)))+'%'

        pad = rng.uniform(low=plot_params['color bar']['pad']['min'], 
                                 high=plot_params['color bar']['pad']['max'])

        cax = divider.append_axes(side, size=size, pad=pad)
        # the side of the axis
        if side == 'right': # this maybe should become a random selection?
            axis_side = 'right'
            cax.yaxis.set_ticks_position(axis_side)
        elif side == 'left':
            axis_side = 'left'
            cax.yaxis.set_ticks_position(axis_side)
        elif side == 'top':
            axis_side = 'top'
            cax.xaxis.set_ticks_position(axis_side)
        elif side == 'bottom':
            axis_side = 'bottom'
            cax.xaxis.set_ticks_position(axis_side)
    plt.draw()
    #print('DRAW WAS CALLED 3')

    # xerrs = []; yerrs = []
    # plt.draw()
    # cols = data_here.get_facecolors()
    # elinewidth = int(round(rng.uniform(low=plot_params['error bars']['elinewidth']['min'], 
    #                                              high=plot_params['error bars']['elinewidth']['max'])))
    # if 'xerrs' in data:# and 'yerrs' not in data: # have x-errors
    #     (_, caps, bars) = ax.errorbar(data['xs'],data['ys'],xerr=data['xerrs'],
    #                                  linewidth=0,elinewidth=elinewidth,
    #                                  markersize=0, ecolor=cols, zorder=0)
    #     xerrs.append(bars)
    # if 'yerrs' in data:# and 'xerrs' not in data: # have x-errors
    #     (_, caps, bars) = ax.errorbar(data['xs'],data['ys'],yerr=data['yerrs'],
    #                                  linewidth=0, elinewidth=elinewidth,
    #                                  markersize=0, ecolor=cols, zorder=0)
    #     yerrs.append(bars)
        
    # save data
    if cax != []:
        data_out = {'data':data_here, 'color bar':cax, 
                    'color bar params':{'side':side, 'pad':pad, 'size':size, 
                                       'axis side':axis_side,
                                       'label prob': plot_params['color bar']['label prob']
                                        }#, 
                                       #'marker':marker, 'marker size':marker_size}
                   }
    else:
        data_out = {'data':data_here}

    # add in x/y errors, if present
    if 'xerrs' in data:
        data_out['x error bars'] = xerrs
    if 'yerrs' in data:
        #print("YESS TO Y IN DATA")
        data_out['y error bars'] = yerrs
    if 'xerrs' in data or 'yerrs' in data:
        data_out['error bar params']:{'elinewidth':elinewidth}
    
    return data_out, ax









# HISTOGRAMS: PLOTS
def get_histogram_plot(plot_params, data, ax, linestyles=linestyles, rng=np.random, **kwargs):
                    #    elinewidth = None, rwidth=None, orientation=None, axis=None, 
                    #    lthick=None, linestyle=None):
    # datas = []
    # linestyles_here = []; linethicks_here = []; markers_here = []
    # marker_sizes_here = []
    # xerrs = []; yerrs = []
    """
    The "None" parameters allow for the specific setting of parameters of the plot, otherwise, they are chosen from a distribution.
    """
    hist = Histogram()
    for k,v in kwargs.items():
        if k in hist.__dict__: # in there
            setattr(hist, k, v)
    #     if 'elinewidth' in k:
    #         elinewidth = v
    # for k,v in hist.__dict__.items():
    #     if k

       

    if hist.elinewidth is None:
        hist.elinewidth = int(round(rng.uniform(low=plot_params['error bars']['elinewidth']['min'], 
                                            high=plot_params['error bars']['elinewidth']['max'])))
    #print('rwidth before', hist.rwidth)
    if hist.rwidth is None:
        hist.rwidth = rng.uniform(low=plot_params['rwidth']['min'], 
                                            high=plot_params['rwidth']['max'])
    #print('rwidth', hist.rwidth)

    if hist.orientation is None: hist.orientation='vertical' # default
    if hist.axis is None: hist.axis = 'xs'
    #err = 'xerrs'
    if len(data['xs']) == 0: # flipped!
        hist.axis = 'ys'
        #err = 'yerrs'
        hist.orientation='horizontal'


    # line boarder?
    if hist.lthick is None: 
        hist.lthick = int(round(rng.uniform(low=plot_params['line thick']['min'], 
                               high=plot_params['line thick']['max'])))
        if rng.uniform(0,1) > plot_params['line thick']['prob']:
            hist.lthick = 0

    # choose random linestyle
    if hist.linestyle is None: hist.linestyle = rng.choice(linestyles)

    # choose random color
    if hist.linecolor is None: hist.linecolor = np.array(ImageColor.getcolor(colors_(1)[0],'RGBA'))/255
    if hist.barcolor is None: hist.barcolor = np.array(ImageColor.getcolor(colors_(1)[0], 'RGBA'))/255

    # try a thing
    hist.linecolor = np.array(hist.linecolor)
    if len(hist.linecolor.shape) == 1:
        hist.linecolor = [hist.linecolor]

    # number of bins?
    if hist.nbins is None: hist.nbins = int(round(rng.uniform(low=plot_params['nbins']['min'], 
                                            high=plot_params['nbins']['max'])))

    if hist.lthick > 0:
        #try:
        # print('edgecolor=', hist.linecolor, len(hist.linecolor))
        # print('color=', hist.barcolor)
        # print('')
        try:
            data_here = ax.hist(data[hist.axis], orientation=hist.orientation, linewidth=hist.lthick, 
                                        linestyle = hist.linestyle, 
                                        edgecolor=hist.linecolor, 
                                color=hist.barcolor, 
                                rwidth=hist.rwidth, bins=hist.nbins)
        except:
            if len(hist.linecolor) == 1: # just the one color
                hist.linecolor = hist.linecolor[0]
            data_here = ax.hist(data[hist.axis], orientation=hist.orientation, linewidth=hist.lthick, 
                                        linestyle = hist.linestyle, 
                                        edgecolor=hist.linecolor, 
                                color=hist.barcolor, 
                                rwidth=hist.rwidth, bins=hist.nbins)
            
            
                    # except Exception as e:
        #     print(str(e))
        #     print('data:',data[axis])
        #     print('orientation:',orientation)
        #     print('linewidth:', lthick)
        #     print('linestyle:', linestyle)
        #     print('linecolor:', linecolor)
        #     print('rwidth:', rwidth)
        #     print('color:',barcolor)
        #     print('nbins:', nbins)
    else:
        data_here = ax.hist(data[hist.axis], orientation=hist.orientation, 
                                     linestyle = hist.linestyle, edgecolor=hist.linecolor, 
                            color=hist.barcolor, rwidth=hist.rwidth, bins=hist.nbins)

    # add in error bars
    data_heights = data_here[0] # heights of DATA values
    bin_edges = data_here[1] # bin edges of DATA values
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    errs = []
    for b in data_heights:
        if hist.es is None:
            es = rng.uniform(low=plot_params['error bars']['x']['size']['min'], 
                                        high=plot_params['error bars']['x']['size']['max'])
        errs.append(b*es)

    if hist.hasErr is None:
        hasErr = False
        if 'xerrs' in data: # different!
            if data['xerrs']:
                hasErr = True
        elif 'yerrs' in data:
            if data['yerrs']:
                hasErr = True
        hist.hasErr = hasErr

    if hist.hasErr:
        (_, caps, bars) = ax.errorbar(bin_centers,data_heights,yerr=errs,
                                          linewidth=0,elinewidth=hist.elinewidth,
                                          markersize=0, ecolor=hist.linecolor, zorder=10) # error bars on top
    
    xerr_bars = []; yerr_bars = []
    if 'xerrs' in data and hist.hasErr:
        #if len(data['xerrs']) > 0:
        if data['xerrs']: # DIFFERENT
            data['xerrs'] = errs
            #print("YES XERR")
            xerr_bars = bars
            #xerr_bars_bars = bars
    elif 'yerrs' in data and hist.hasErr:
        #if len(data['yerrs']) > 0:
        if data['yerrs']: # DIFFERENT
            #print("YES YERR")
            yerr_bars = bars
            data['yerrs'] = errs
            #yerr_bars_bars = bars

    data_out = {'data':data_here, 'plot params':{
                                            'linethick':hist.lthick, 
                                            'linestyles':hist.linestyle,
                                             'bar color':hist.barcolor,
                                             'edge color':hist.linecolor,
                                             'orientation':hist.orientation,
                                             'rwidth':hist.rwidth,
                                             'nbins':hist.nbins
                                            }
               }
    
    # add in x/y errors, if present
    if len(xerr_bars) > 0:
        data_out['x error bars'] = [xerr_bars] # list for formatting
        #data_out['x error bars plot'] = [xerr_bars_bars] # list for formatting
        data_out['plot params']['elinewidth'] = hist.elinewidth
    if len(yerr_bars) > 0:
        data_out['y error bars'] = [yerr_bars]
        #data_out['y error bars plot'] = [yerr_bars_bars]
        data_out['plot params']['elinewidth'] = hist.elinewidth
    return data_out, ax

###############################################
############## MAIN PLOT #####################
###############################################

def make_plot(plot_params, data, ax, plot_type='line', linestyles=linestyles,
              iplot=None, nrows=None, ncols=None, rng=np.random, **kwargs):#, plot_style='default'):
    if plot_type == 'line':
        data_out, ax = get_line_plot(plot_params, data, ax, linestyles=linestyles, rng=rng, **kwargs)
        return data_out, ax
    elif plot_type == 'scatter':
        data_out, ax = get_scatter_plot(plot_params, data, ax, rng=rng)
        return data_out, ax
    elif plot_type == 'histogram':
        data_out, ax = get_histogram_plot(plot_params, data, ax, linestyles=linestyles, rng=rng, **kwargs)
        return data_out, ax
    elif plot_type == 'contour':
        data_out, ax = get_contour_plot(plot_params, data, ax, rng=rng)
        return data_out, ax
    else:
        print('not implement yet!')
        import sys; sys.exit()



def colorbar_mods(cbar, side, fig):
    # axis labels slide
    bottom = False; top = False; right = False; left = False
    in_or_out = np.random.choice(['in','out'])

    # flip labels?
    if side == 'top' or side == 'bottom':
        # turn off side axis stuff
        cbar.ax.set_ylabel('')
        cbar.ax.tick_params(length=0, axis='y', labelsize=-1, color=fig.get_facecolor(), 
                            labelcolor=fig.get_facecolor())
        # mods for c-bar axis
        if side == 'bottom':
            bottom = True
        else:
            top = True
        cbar.ax.tick_params(axis='x',direction=in_or_out, 
                            labelbottom=bottom, labeltop=top, 
                            labelleft=left, labelright=right)
    else:
        # turn off bottom axis stuff
        cbar.ax.set_xlabel('')
        cbar.ax.tick_params(length=0, axis='x', labelsize=-1, color=fig.get_facecolor(), 
                            labelcolor=fig.get_facecolor())

        # mods for c-bar axis
        if side == 'left':
            left = True
        else:
            right = True
        cbar.ax.tick_params(axis='y',direction=in_or_out, 
                            labelbottom=bottom, labeltop=top, 
                            labelleft=left, labelright=right)
    return cbar



def make_base_plot(plot_style, color_map, dpi, nrows, ncols, aspect_fig,
                   base=5, verbose=True, tight_layout = True):
    plt.close('all')
    plt.style.use(plot_style)
    plt.set_cmap(color_map) 
    figsize = (base*ncols*aspect_fig, base*nrows) # w,h
    if verbose: print('figsize (w,h) =', figsize)

    if tight_layout:
        fig = plt.figure(figsize=figsize, dpi=dpi,layout='tight')
    else:
        fig = plt.figure(figsize=figsize, dpi=dpi)

    axes = []
    plot_inds = []
    for i in range(nrows):
        for j in range(ncols):
            iplot = (i*nrows) + j
            ax = fig.add_subplot(nrows, ncols, iplot + 1)
            axes.append(ax)
            plot_inds.append([i,j])

    return fig, axes, plot_inds


