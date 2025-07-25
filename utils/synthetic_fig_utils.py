import numpy as np



# for plotting, if you want to chop by tolerance
def subset_by_percent(dfin, tol = 0.01, verbose=True, round_off = 2, 
                     tol_count = None, reset_index = True, 
                     replace_insert = True, replace_deletion = True, 
                    track_insert_delete = False):
    """
    tol : in % (so 1.0 will be 1%, 0.1 will be 0.1%)
    tol_count : if not None, will over-write tol and subset by count
    """
    if tol_count is None:
        dfin_subset = dfin.loc[dfin['counts']> tol].copy()
    else:
        dfin_subset = dfin.loc[dfin['counts unnormalized']> tol_count].copy()

    # also, add the tool tip
    names = []
    for i in range(len(dfin_subset)):
        d = dfin_subset.iloc[i]
        names.append(str(round(d['counts'],2))+'%')
    dfin_subset['name']=names
    
    # rename columns for plotting 
    dfin_subset = dfin_subset.rename(columns={"counts": "% of all OCR tokens", 
                                              "counts unnormalized": "Total Count of PDF token"})
    if reset_index:
        dfin_subset = dfin_subset.reset_index(drop=True)
        
    # replace insert
    if replace_insert:
        dfin_subset.loc[(dfin_subset['ocr_letters']=='^')&(dfin_subset['pdf_letters']!='^'),'ocr_letters'] = 'INSERT'
    if replace_deletion:
        dfin_subset.loc[(dfin_subset['pdf_letters']=='@')&(dfin_subset['ocr_letters']!='@'),'pdf_letters'] = 'DELETE'
        
    d = dfin_subset.loc[(dfin_subset['ocr_letters']=='INSERT')&(dfin_subset['pdf_letters']=='DELETE')]
    if track_insert_delete:
        if len(d) > 0:
            print('Have overlap of insert and delete!')
            print(len(d))
    else: # assume error
        dfin_subset.loc[(dfin_subset['ocr_letters']=='INSERT')&(dfin_subset['pdf_letters']=='DELETE'),
                        '% of all OCR tokens'] = np.nan
        dfin_subset.loc[(dfin_subset['ocr_letters']=='INSERT')&(dfin_subset['pdf_letters']=='DELETE'),
                        "Total Count of PDF token"] = np.nan


    if verbose:
        print('shape of output=', dfin_subset.shape)
    return dfin_subset


def get_line_styles():
    # from https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
    linestyle_str = [
         ('solid', 'solid'),      # Same as (0, ()) or '-'
         ('dotted', 'dotted'),    # Same as (0, (1, 1)) or ':'
         ('dashed', 'dashed'),    # Same as '--'
         ('dashdot', 'dashdot')]  # Same as '-.'
    
    linestyle_tuple = [
         ('loosely dotted',        (0, (1, 10))),
         ('dotted',                (0, (1, 1))),
         ('densely dotted',        (0, (1, 1))),
         ('long dash with offset', (5, (10, 3))),
         ('loosely dashed',        (0, (5, 10))),
         ('dashed',                (0, (5, 5))),
         ('densely dashed',        (0, (5, 1))),
    
         ('loosely dashdotted',    (0, (3, 10, 1, 10))),
         ('dashdotted',            (0, (3, 5, 1, 5))),
         ('densely dashdotted',    (0, (3, 1, 1, 1))),
    
         ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
         ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
         ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

    line_styles = []
    for n,l in linestyle_str[::-1]:
        line_styles.append(l)
    for n,l in linestyle_tuple[::-1]:
        line_styles.append(l)
    
    return np.array(line_styles, dtype=object)


def get_nrows_and_ncols(panel_params, verbose=True, rng=np.random):
    npanels = int(round(rng.normal(loc=panel_params['number prob']['median'], 
                                  scale=panel_params['number prob']['std'])))
    if npanels < panel_params['number prob']['min']:
        npanels = panel_params['number prob']['min']
    if npanels > panel_params['number prob']['max']:
        npanels = panel_params['number prob']['max']
    if verbose: print('selected npanels:', npanels)
    
    # panel layout type?
    choices = []; probs = []
    for k,v in panel_params['layout prob'].items():
        choices.append(k)
        probs.append(v)
        
    panel_style = rng.choice(choices, p=probs)
    
    if npanels > panel_params['to even above']:
        panel_style = 'squarish'
    
    # if square, might need to fudge the actual number
    if panel_style == 'squarish':
        n1 = int(np.floor(np.sqrt(npanels)))
        n2 = int(round(npanels/n1))
        npanels = n1*n2
        nrows, ncols = n1,n2
        # flip?
        if rng.uniform(0,1) < 0.5:
            nrows,ncols = n2,n1
    elif panel_style == 'horizontal':
        nrows = 1
        ncols = npanels
    elif panel_style == 'vertical':
        nrows = npanels
        ncols = 1
    
    return npanels, panel_style, nrows, ncols


def normalize_params_prob(plot_types_params, panel_params, 
                          title_params, xlabel_params, 
                          ylabel_params, colorbar_params, verbose=True):
    # create the fake figs
    #print(plot_types_params['scatter'])
    
    p = 0.0
    for k,v in plot_types_params.items():
        p += v['prob']
    if p != 1.0:
        newp = {}
        for k,v in plot_types_params.items():
            newp[k] = v['prob']/p
            plot_types_params[k]['prob'] = v['prob']/p
        if verbose: 
            print('plot_types_params probability did not add to 1! total =', p)
            print('renormalizing...')
            print('now: ', newp)
    
    # layout prob
    p = 0.0
    for k,v in panel_params['layout prob'].items():
        p += v
    if p != 1.0:
        for k,v in panel_params['layout prob'].items():
            panel_params['layout prob'][k] = v/p
        if verbose:
            print('panel_params layout probability did not add to 1! total =', p)
            print('renormalizing...')
            print('now: ', panel_params['layout prob'])
    
    # capitilize
    p = 0.0
    for k,v in title_params['capitalize'].items():
        p += v
    if p != 1.0:
        for k,v in title_params['capitalize'].items():
            title_params['capitalize'][k] = v/p
        if verbose:
            print('title_params capatilize did not add to 1! total =', p)
            print('renormalizing...')
            print('now: ', title_params['capitalize'])
    p = 0.0
    for k,v in xlabel_params['capitalize'].items():
        p += v
    if p != 1.0:
        for k,v in xlabel_params['capitalize'].items():
            xlabel_params['capitalize'][k] = v/p
        if verbose:
            print('xlabel_params capatilize did not add to 1! total =', p)
            print('renormalizing...')
            print('now: ', xlabel_params['capitalize'])
    p = 0.0
    for k,v in ylabel_params['capitalize'].items():
        p += v
    if p != 1.0:
        for k,v in ylabel_params['capitalize'].items():
            ylabel_params['capitalize'][k] = v/p
        if verbose:
            print('ylabel_params capatilize did not add to 1! total =', p)
            print('renormalizing...')
            print('now: ', ylabel_params['capitalize'])
    # colorbar
    p = 0.0
    for k,v in colorbar_params['capitalize'].items():
        p += v
    if p != 1.0:
        for k,v in colorbar_params['capitalize'].items():
            colorbar_params['capitalize'][k] = v/p
        if verbose:
            print('colorbar_params capatilize did not add to 1! total =', p)
            print('renormalizing...')
            print('now: ', colorbar_params['capitalize'])

    if 'scatter' in plot_types_params:
        p = 0.0
        for k,v in plot_types_params['scatter']['color bar']['location probs'].items():
            p += v
        if p != 1.0:
            for k,v in plot_types_params['scatter']['color bar']['location probs'].items():
                plot_types_params['scatter']['color bar']['location probs'][k] = v/p
            if verbose:
                print("plot_types_params['scatter']['color bar']['location probs'] did not add to 1! total =", p)
                print('renormalizing...')
                print('now: ', plot_types_params['scatter']['color bar']['location probs'])

    # images or contours
    # 'image or contour':{'prob':{'image':0.5, 'contour':0.5, 'both':0.5}
    for ptype in ['contour']:
        if ptype in plot_types_params:
            p = 0.0
            for k,v in plot_types_params[ptype]['color bar']['location probs'].items():
                p += v
            if p != 1.0:
                for k,v in plot_types_params[ptype]['color bar']['location probs'].items():
                    plot_types_params[ptype]['color bar']['location probs'][k] = v/p
                if verbose:
                    print("plot_types_params['"+ptype+"']['color bar']['location probs'] did not add to 1! total =", p)
                    print('renormalizing...')
                    print('now: ', plot_types_params[ptype]['color bar']['location probs'])
        
            p = 0.0
            for k,v in plot_types_params[ptype]['image or contour']['prob'].items():
                p += v
            if p != 1.0:
                for k,v in plot_types_params[ptype]['image or contour']['prob'].items():
                    plot_types_params[ptype]['image or contour']['prob'][k] = v/p
                if verbose:
                    print("plot_types_params['"+ptype+"']['image or contour']['prob'] did not add to 1! total =", p)
                    print('renormalizing...')
                    print('now: ', plot_types_params[ptype]['image or contour']['prob'])

    # distribution probabilities
    #print('')
    for k1 in ['line','histogram','scatter','contour']:
        if k1 in plot_types_params:
            p=0
            if 'distribution' in plot_types_params[k1]:
                for dist,vals in plot_types_params[k1]['distribution'].items():
                    p += vals['prob']
                #print('p is:', p)
                if p != 1.0:
                    ps = plot_types_params[k1]['distribution'].copy()
                    for k,v in plot_types_params[k1]['distribution'].items():
                        ps[k]['prob'] = v['prob']/p
                    plot_types_params[k1]['distribution'] = ps
                    if verbose:
                        print("plot_types_params['" + str(k1) + "']['distribution'] probabilities did not add to 1! total =", p)
                        print('renormalizing...')
                        ps = []
                        for k2,v2 in plot_types_params[k1]['distribution'].items():
                            #print(v2)
                            ps.append(v2['prob'])
                        print('now: ', ps)

    return plot_types_params, panel_params, title_params, xlabel_params, ylabel_params




def get_ticks_not_imgOfSky(ticklabels, ticklines, fig=None, dpi=None):
    xticks = []
    # ticks = [t for t in ax.get_xticklabels()]
    # tick_locs = ax.get_xticklines()
    ticks = [t for t in ticklabels]
    tick_locs = ticklines
    modder = len(tick_locs)/len(ticks)
    if int(modder) != modder:
        print('cant divide!')
        import sys; sys.exit()

    modder = int(modder)
    for ip, t in enumerate(ticks):
        if fig is not None:
            if dpi is None:
                tx = 0.5*(tick_locs[ip*modder].get_window_extent(renderer=fig.canvas.get_renderer()).x0+tick_locs[ip*modder].get_window_extent(renderer=fig.canvas.get_renderer()).x1)
                ty = 0.5*(tick_locs[ip*modder].get_window_extent(renderer=fig.canvas.get_renderer()).y0+tick_locs[ip*modder].get_window_extent(renderer=fig.canvas.get_renderer()).y1)
                if t.get_visible():
                    xticks.append( (t.get_text(), t.get_window_extent(renderer=fig.canvas.get_renderer()).x0, t.get_window_extent(renderer=fig.canvas.get_renderer()).y0,
                                    t.get_window_extent(renderer=fig.canvas.get_renderer()).x1, t.get_window_extent(renderer=fig.canvas.get_renderer()).y1, tx,ty) )
            else:
                tx = 0.5*(tick_locs[ip*modder].get_window_extent(renderer=fig.canvas.get_renderer()).x0+tick_locs[ip*modder].get_window_extent(renderer=fig.canvas.get_renderer()).x1)
                ty = 0.5*(tick_locs[ip*modder].get_window_extent(renderer=fig.canvas.get_renderer()).y0+tick_locs[ip*modder].get_window_extent(renderer=fig.canvas.get_renderer()).y1)
                if t.get_visible():
                    xticks.append( (t.get_text(), t.get_window_extent(renderer=fig.canvas.get_renderer(),dpi=dpi).x0, t.get_window_extent(renderer=fig.canvas.get_renderer(),dpi=dpi).y0,
                                    t.get_window_extent(renderer=fig.canvas.get_renderer(),dpi=dpi).x1, t.get_window_extent(renderer=fig.canvas.get_renderer(),dpi=dpi).y1, tx,ty) )
        else:
            if dpi is None:
                tx = 0.5*(tick_locs[ip*modder].get_window_extent().x0+tick_locs[ip*modder].get_window_extent().x1)
                ty = 0.5*(tick_locs[ip*modder].get_window_extent().y0+tick_locs[ip*modder].get_window_extent().y1)
                if t.get_visible():
                    xticks.append( (t.get_text(), t.get_window_extent().x0, t.get_window_extent().y0,
                                    t.get_window_extent().x1, t.get_window_extent().y1, tx,ty) )
            else:
                tx = 0.5*(tick_locs[ip*modder].get_window_extent().x0+tick_locs[ip*modder].get_window_extent().x1)
                ty = 0.5*(tick_locs[ip*modder].get_window_extent().y0+tick_locs[ip*modder].get_window_extent().y1)
                if t.get_visible():
                    xticks.append( (t.get_text(), t.get_window_extent(dpi=dpi).x0, t.get_window_extent(dpi=dpi).y0,
                                    t.get_window_extent(dpi=dpi).x1, t.get_window_extent(dpi=dpi).y1, tx,ty) )
    return xticks



def get_ticks(ax, plot_type, axis, fig=None, dpi=None, minor=False, verbose = False):
    if axis == 'x':
        ticklabels,ticklines = ax.get_xticklabels(), ax.get_xticklines(minor=minor)
    elif axis == 'y':
        ticklabels,ticklines = ax.get_yticklabels(), ax.get_yticklines(minor=minor)
    else:
        print('[ERROR]: in "get_ticks" in synthetic_fig_utils -- no axis type for:', axis)
        import sys; sys.exit()

    # JPN -- default is None DPI and fig!
    ticks = get_ticks_not_imgOfSky(ticklabels, ticklines, fig=None, dpi=None)

    return ticks


def replace_label(w,i,eqs,inlines, 
                 latex_replacements = {r'\.':r'.'}):
    w2 = []
    icount = 0
    for j in range(len(w)):
        if eqs[j]: # yes, replace
            replace = inlines[i[icount]]
            for lr,rp in latex_replacements.items():
                if lr in replace:
                    replace = replace.replace(lr,rp)
            w2.append(replace)
            icount+=1
        else: 
            w2.append(w[j])
    return np.array(w2)




def get_titles_or_labels(words, cap, eq, inlines, nwords=1, rng=np.random):
    """
    Return a title or x/y axes label
    words : list of words to pull from
    cap : if 'first' will just be the first words capitalized, 
          if 'all' the totality of words will be capitalized
    eq : probability of flipping from a word to an equation
    inlines : list of "in line" math formulas in tex
    plot_type : some plots have special tags
    x_or_y : specify x or y for special plots
    """
    if rng == np.random:
        i = rng.randint(0,len(words),size=nwords)
    else:
        i = rng.integers(0,len(words),size=nwords)
    w = np.array(words)[i]
    probs, choices = [],[]
    for k,v in cap.items():
        choices.append(k)
        probs.append(v)
    c = rng.choice(choices, p=probs)
    if c == 'first':
        for i in range(len(w)):
            w[i] = w[i].capitalize()
    elif c == 'all':
        for i in range(len(w)):
            w[i] = w[i].upper()
    # turn any into random equation?
    p = rng.uniform(0,1, size=len(w))
    eqs = p <= eq['prob']
    if len(w[eqs]) > 0: # have some words
        if rng == np.random:
            i = rng.randint(0,len(inlines),size=len(w[eqs])) # grab these inlines
        else:
            i = rng.integers(0,len(inlines),size=len(w[eqs])) # grab these inlines
        w = replace_label(w,i,eqs,inlines)

    w = w.tolist()

    if len(w) > 1:
        wout = r" ".join(w)
    else:
        wout = w[0]
    wout = wout.replace('\n', '')

    return wout


def get_titles_or_labels_ra_dec(plot_params, data, cap, rng=np.random):
    """
    Return a title or x/y axes label
    cap : if 'first' will just be the first words capitalized, 
          if 'all' the totality of words will be capitalized
    """

    # We also gotta think about [coordinate systems](https://github.com/astropy/astropy-api/blob/master/wcs_axes/wcs_api.md#coordinate-systems):

    # * 'fk4' or 'b1950': B1950 equatorial coordinates
    # * 'fk5' or 'j2000': J2000 equatorial coordinates
    # * 'gal' or 'galactic': Galactic coordinates
    # * 'ecl' or 'ecliptic': Ecliptic coordinates
    # * 'sgal' or 'supergalactic': Super-Galactic coordinates
    if 'WCS' in data['data params']:
        wcs = data['data params']['WCS'].wcs
        if wcs.radesys.lower() != 'fk5' and wcs.radesys.lower() != 'fk4':
            print('[ERROR]: in synthtic_fig_utils/get_titles_or_labels_ra_dec -- coord system not supported:', wcs.radesys)
            import sys; sys.exit()
        else: # its one of these
            # choose which format
            xylabel = rng.choice(plot_params['xy labels ra/dec'], size=1)[0]
            ra = xylabel['x']
            dec = xylabel['y']
    else: # its one of these
        # choose which format
        xylabel = rng.choice(plot_params['xy labels ra/dec'], size=1)[0]
        ra = xylabel['x']
        dec = xylabel['y']

    # captialize?
    probs, choices = [],[]
    for k,v in cap.items():
        choices.append(k)
        probs.append(v)
    cc = rng.choice(choices, p=probs)
    #print('CAP:', cc)
    coords = [ra,dec]
    for ic,c in enumerate(coords):
        if '$' not in c: # no inlines
            if cc == 'first':
                coords[ic] = c.title()
            elif cc == 'all':
                coords[ic] = c.upper()

    if 'WCS' in data['data params']:
        if rng.uniform() < plot_params['show temporal']['prob']: # yes
            equinox = str(int(wcs.equinox))
            startend = rng.choice(plot_params['show temporal']['styles'])
            coords[0] = coords[0] + ' ' + startend['start'] + equinox + startend['end']
            coords[1] = coords[1] + ' ' + startend['start'] + equinox + startend['end']
    else: # fake it
        if rng.uniform() < plot_params['show temporal']['prob']: # yes
            equinox = rng.choice(['1900','1950','2000'])
            startend = rng.choice(plot_params['show temporal']['styles'])
            coords[0] = coords[0] + ' ' + startend['start'] + equinox + startend['end']
            coords[1] = coords[1] + ' ' + startend['start'] + equinox + startend['end']
    return coords[0],coords[1]


def get_font_info(fontsizes, font_names, rng=np.random):
    # font sizes
    title_fontsize = int(round(rng.uniform(low=fontsizes['title']['min'], 
                                                 high=fontsizes['title']['max'])))
    colorbar_fontsize = int(round(rng.uniform(low=fontsizes['colorbar']['min'], 
                                                 high=fontsizes['colorbar']['max'])))
    xlabel_fontsize = int(round(rng.uniform(low=fontsizes['xlabel']['min'], 
                                                 high=fontsizes['xlabel']['max'])))
    if not fontsizes['x/y label same']:
        ylabel_fontsize = int(round(rng.uniform(low=fontsizes['ylabel']['min'], 
                                                      high=fontsizes['ylabel']['max'])))
    else:
        ylabel_fontsize = xlabel_fontsize # for consistancy
    xlabel_ticks_fontsize = int(round(rng.uniform(low=fontsizes['ticks']['min'], 
                                                  high=fontsizes['ticks']['max'])))
    if not fontsizes['x/y ticks same']:
        ylabel_ticks_fontsize = int(round(rng.uniform(low=fontsizes['ticks']['min'], 
                                                      high=fontsizes['ticks']['max'])))
    else:
        ylabel_ticks_fontsize = xlabel_ticks_fontsize # for consistancy

    # get fonts
    csfont = {'fontname':rng.choice(font_names)}

    return title_fontsize, colorbar_fontsize, xlabel_fontsize, ylabel_fontsize, xlabel_ticks_fontsize, ylabel_ticks_fontsize, csfont


def add_titles_and_labels(ax, popular_nouns_x, popular_nouns_y, popular_nouns_title, 
                          title_params, csfont, title_fontsize, 
                          xlabel_params, ylabel_params,
                          xlabel_fontsize, ylabel_fontsize,
                          inlines, xlabel_ticks_fontsize, ylabel_ticks_fontsize,
                          rng=np.random):
    """
    Set x/y and title labels based on either randomly drawing from a set of words or as fixed inputs.
    """
    p = rng.uniform(0,1)
    if p < title_params['prob'] or type(popular_nouns_title) == str:
        if type(popular_nouns_title) != str:
            title_words = get_titles_or_labels(popular_nouns_title, title_params['capitalize'],
                                        title_params['equation'], inlines,
                                        nwords=rng.integers(low=title_params['n words']['min'],
                                                                high=title_params['n words']['max']+1), 
                                                                rng=rng)
        else:
            title_words = popular_nouns_title
        title = ax.set_title(title_words, fontsize = title_fontsize, **csfont)
    else:
        title = ''

    if type(popular_nouns_x) != str:
        xlabel_words = get_titles_or_labels(popular_nouns_x, xlabel_params['capitalize'],
                                    xlabel_params['equation'], inlines,
                                    nwords=rng.integers(low=xlabel_params['n words']['min'],
                                                            high=xlabel_params['n words']['max']+1),
                                                            rng=rng)
    else:
        xlabel_words = popular_nouns_x

    if type(popular_nouns_y) != str:
        ylabel_words = get_titles_or_labels(popular_nouns_y, ylabel_params['capitalize'],
                                ylabel_params['equation'], inlines,
                                nwords=rng.integers(low=ylabel_params['n words']['min'],
                                                        high=ylabel_params['n words']['max']+1),
                                                        rng=rng)
    else:
        ylabel_words = popular_nouns_y
    
    xlabel = ax.set_xlabel(xlabel_words, fontsize=xlabel_fontsize, **csfont)
    ylabel = ax.set_ylabel(ylabel_words, fontsize=ylabel_fontsize, **csfont)

    # set ticksizes
    ax.tick_params(axis='x', which='major', labelsize=xlabel_ticks_fontsize, labelfontfamily=csfont['fontname'])
    ax.tick_params(axis='y', which='major', labelsize=ylabel_ticks_fontsize, labelfontfamily=csfont['fontname'])

    return title, xlabel, ylabel



##### not sure if this is the right spot for this.... ####
from .figure_gen_utils.pixel_location_utils import get_data_pixel_locations
import json
from .data_utils import NumpyEncoder


def collect_plot_data_axes(ax, fig,
                           height, width,
                           data_from_plot, data_for_plot, plot_type, title, xlabel, ylabel,
                           distribution_type, verbose=False):
    """
    Collect the data from each plot.  Axes-level (per axis)

    ax : individual axes
    iplot : plot index
    """

    if ax.get_figure() is None:
        if verbose:
            print('[WARNING]: ax has no figure, using "fig"')
        ax.set_figure(fig)
    ###### get data from plot ######
    # data_from_plot = data_from_plots[iplot]
    # data_for_plot = data_for_plots[iplot]
    # plot_type = plot_types[iplot]
    # title = titles[iplot]
    # xlabel = xlabels[iplot]
    # ylabel = ylabels[iplot]
    ### JPN -- add back in char label
    # cbar_label = cbar_labels[iplot]
    # cbar_word = cbar_words[iplot]

    # includes colors
    data_pixels = get_data_pixel_locations(data_from_plot, plot_type, ax, width, height)

    # bounding box of square
    bbox = ax.get_position() # Bbox(x0, y0, x1, y1)
    xpix1 = np.array([bbox.x0,bbox.x1])
    ypix1 = np.array([bbox.y0,bbox.y1])
    xpix1 *= width
    ypix1 *= height
        
    # x-tick locations
    try:
        xticks = get_ticks(ax, plot_type, 'x', fig=fig, verbose=verbose) # fig is not used for "regular" plots
    except Exception as e:
        if verbose:
            print('[ERROR]: issue getting x-ticks')
            print('  ', str(e))
        success_plot = False
        import sys; sys.exit()
        #continue

    # y-tick locations
    try:
        yticks = get_ticks(ax, plot_type, 'y', fig=fig, verbose=verbose)
    except Exception as e:
        if verbose:
            print('[ERROR]: issue getting y-ticks')
            print('  ', str(e))
            import sys; sys.exit()
        success_plot = False
        #continue
    
    #### JPN -- come back and add colorbars!
    # # for colorbars
    # colorbar_ticks = []
    # cbar_bbox = None; cbar_text = None
    # if 'color bar' in data_from_plot:
    #     colorbar = data_from_plot['color bar']
    #     if data_from_plot['color bar params']['side'] == 'left' \
    #         or data_from_plot['color bar params']['side'] == 'right':
    #         cbarax = 'y'
    #     else:
    #         cbarax = 'x'
    #     try:
    #         colorbar_ticks = get_ticks(colorbar, plot_type, cbarax, fig=fig, verbose=verbose)
    #     except Exception as e:
    #         if verbose:
    #             print('[ERROR]: issue getting colorbar ticks')
    #             print('  ', str(e))
    #             success_plot = False

        
    # Get the bounding box of the title in display space
    if title != '':
        title_bbox = title.get_window_extent()#dpi=dpi)
        title_words = title.get_text()
    else:
        title_bbox = -1
        title_words = ''

    # xlabel
    xlabel_bbox = xlabel.get_window_extent()#dpi=dpi)
    xlabel_words = xlabel.get_text()
    # ylabel
    ylabel_bbox = ylabel.get_window_extent()#dpi=dpi)
    ylabel_words = ylabel.get_text()

    # get offset text
    yoffset_text_obj = ax.yaxis.get_offset_text()
    yoffset_text = yoffset_text_obj.get_text()
    yoffset_text_bbox = None
    if yoffset_text != '':
        yoffset_text_bbox = yoffset_text_obj.get_window_extent()
    # also for x
    xoffset_text_obj = ax.xaxis.get_offset_text()
    xoffset_text = xoffset_text_obj.get_text()
    xoffset_text_bbox = None
    if xoffset_text != '':
        xoffset_text_bbox = xoffset_text_obj.get_window_extent()

    ####### SAVE THE DATA ######

    # line plot 
    #plot_name = 'plot' + str(iplot) 
    datas = {}
    # line plot type
    datas['type'] = plot_type # tag for kind of plot
    datas['distribution'] = distribution_type
    datas['data'] = data_for_plot
    if data_pixels != {}:
        datas['data pixels'] = data_pixels
    datas['data from plot'] = json.loads(json.dumps(data_from_plot, cls=NumpyEncoder))
    if (plot_type == 'scatter' or plot_type == 'contour' or plot_type == 'image of the sky') and 'color bar' in data_from_plot:
        #print('yes indeed')
        try:
            w = data_from_plot['color bar'].get_window_extent()#dpi=dpi)
        except:
            w = data_from_plot['color bar'].get_window_extent()
        datas['color bar'] = {'xmin':w.x0,'ymin':w.y0,
                                            'xmax':w.x1,'ymax':w.y1, 
                                            'params':data_from_plot['color bar params']}
        

        ##### JPN -- come back and add colorbars
        # # is it an image of the sky? (WCAxes)
        # if cbar_ax != []: # placeholder for no colorbar
        #     colorbar_label = None
        #     colorbar_offset_text = None
        #     if hasattr(cbar_ax, 'coords'):
        #         # have text
        #         for cbar_axc in cbar_ax.coords:
        #             if cbar_axc._axislabels.get_text() != '':
        #                 cbar_text = cbar_axc._axislabels.get_text()
        #                 cbar_bbox = cbar_axc._axislabels.get_window_extent()
        #                 colorbar_label = {'text':cbar_text, 
        #                                     'xmin':cbar_bbox.x0, 
        #                                     'ymin':cbar_bbox.y0,
        #                                     'xmax':cbar_bbox.x1,
        #                                     'ymax':cbar_bbox.y1}
        #                 print('colorbar_label is (WCAxes):', colorbar_label)
        #                 #print("HAVE TO CHECK FOR OFFSET TEXT")
        #                 #import sys; sys.exit()
        #         # try this
        #         yoff = cbar_ax.yaxis.get_offset_text() #get_text()
        #         xoff = cbar_ax.yaxis.get_offset_text()
        #         if xoff.get_text() != '' and yoff.get_text() != '':
        #             print('both x & y have offset text and I dont know how to deal!')
        #             import sys; sys.exit()
        #         elif xoff.get_text() != '':
        #             cbar_offset_text = xoff
        #         else:
        #             cbar_offset_text = yoff # either something or nothing
        #         if cbar_offset_text.get_text() != '':
        #             cbar_ot_bb = cbar_offset_text.get_window_extent()
        #             colorbar_offset_text = {'text':cbar_offset_text.get_text(), 
        #                                     'xmin':cbar_ot_bb.x0, 
        #                                     'ymin':cbar_ot_bb.y0,
        #                                     'xmax':cbar_ot_bb.x1,
        #                                     'ymax':cbar_ot_bb.y1}
        #     elif hasattr(cbar_ax, '_colorbar'):
        #         # check both x & y
        #         if cbar_ax.yaxis.label.get_text() != '':
        #             cbar_text = cbar_ax.yaxis.label.get_text()
        #             cbar_bbox = cbar_ax.yaxis.label.get_window_extent()
        #             cbar_offset_text = cbar_ax.yaxis.get_offset_text()
        #         elif cbar_ax.xaxis.label.get_text() != '':
        #             cbar_text = cbar_ax.xaxis.label.get_text()
        #             cbar_bbox = cbar_ax.xaxis.label.get_window_extent()
        #             cbar_offset_text = cbar_ax.xaxis.get_offset_text()
        #         else:
        #             print('no label for colorbar!')
        #             cbar_text = ''
        #             cbar_offset_text = cbar_ax.xaxis.get_offset_text() # placeholder
        #         # cbar_text = cbar_ax.get_ylabel()
        #         # cbar_bbox = cbar_ax.get_window_extent()
        #         if cbar_text != '':
        #             colorbar_label = {'text':cbar_text, 
        #                                     'xmin':cbar_bbox.x0, 
        #                                     'ymin':cbar_bbox.y0,
        #                                     'xmax':cbar_bbox.x1,
        #                                     'ymax':cbar_bbox.y1}
        #             print('colorbar_label is (matplotlib):', colorbar_label)
        #         #import sys; sys.exit()
        #         if cbar_offset_text.get_text() != '':
        #             cbar_ot_bb = cbar_offset_text.get_window_extent()
        #             colorbar_offset_text = {'text':cbar_offset_text.get_text(), 
        #                                     'xmin':cbar_ot_bb.x0, 
        #                                     'ymin':cbar_ot_bb.y0,
        #                                     'xmax':cbar_ot_bb.x1,
        #                                     'ymax':cbar_ot_bb.y1}
        #     else:
        #         print('not sure what kind of colorbar this is!')
        #         import sys; sys.exit()

        #     if colorbar_label is None:
        #         print('colorbar_label is None for iplot =', iplot, '!')
        #         #import sys; sys.exit()
        #     else:
        #         datas['color bar']['label'] = colorbar_label.copy()

        #     if colorbar_offset_text is None:
        #         pass
        #     else:
        #         datas['color bar']['offset text'] = colorbar_offset_text.copy()

    xtmp = []
    for xt in xticks:
        l = {'data':xt[0], 'xmin': xt[1], 
                'ymin': xt[2], 
                'xmax':xt[3], 'ymax':xt[4],
                'tx':xt[5], 'ty':xt[6]}
        xtmp.append(l)
    datas['xticks'] = xtmp.copy()
    # 
    xtmp = []
    for xt in yticks:
        l = {'data':xt[0], 'xmin': xt[1], 
                'ymin': xt[2], 
                'xmax':xt[3], 'ymax':xt[4], 
            'tx':xt[5], 'ty':xt[6]}
        xtmp.append(l)
    datas['yticks'] = xtmp.copy()
    ##### JPN -- come back for colorbars
    # if len(colorbar_ticks) > 0:
    #     xtmp = []
    #     for xt in colorbar_ticks:
    #         l = {'data':xt[0], 'xmin': xt[1], 
    #                 'ymin': xt[2], 
    #                 'xmax':xt[3], 'ymax':xt[4], 
    #             'tx':xt[5], 'ty':xt[6]}
    #         xtmp.append(l)
    #     datas['color bar ticks'] = xtmp.copy()
        
    # axis box
    datas['square'] = {'xmin':xpix1[0], 'ymin':ypix1[0], 
                                        'xmax':xpix1[1], 'ymax':ypix1[1]}
    # title
    if title_bbox != -1:
        datas['title'] = {'xmin':title_bbox.x0, 'ymin':title_bbox.y0, 
                                        'xmax':title_bbox.x1, 'ymax':title_bbox.y1,
                                        'words':title_words}
    else:
        pass
    datas['xlabel'] = {'xmin':xlabel_bbox.x0, 'ymin':xlabel_bbox.y0, 
                                    'xmax':xlabel_bbox.x1, 'ymax':xlabel_bbox.y1,
                                    'words':xlabel_words}
    datas['ylabel'] = {'xmin':ylabel_bbox.x0, 'ymin':ylabel_bbox.y0, 
                                    'xmax':ylabel_bbox.x1, 'ymax':ylabel_bbox.y1,
                                    'words':ylabel_words}
    # offset text
    for lt,lbb,t in zip([xoffset_text,yoffset_text],
                        [xoffset_text_bbox,yoffset_text_bbox], ['x','y']):
        if lt != '': # have something
            datas[t + '-offset text'] = {'xmin':lbb.x0, 
                                                    'ymin':lbb.y0, 
                                    'xmax':lbb.x1, 'ymax':lbb.y1,
                                    'words':lt}
            
    return datas