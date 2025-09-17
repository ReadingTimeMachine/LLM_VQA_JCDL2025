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

    # XYZ -- default is None DPI and fig!
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
                           distribution_type, verbose=False,
                           #cbar_label = None, cbar_word=None, 
                           cbar_ax=None, 
                           colorbar_verbose=False, error_out=True):
    """
    Collect the data from each plot.  Axes-level (per axis)

    ax : individual axes
    iplot : plot index
    """
    err = False
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
        if error_out:
            import sys; sys.exit()
        else:
            return '', True
        #continue

    # y-tick locations
    try:
        yticks = get_ticks(ax, plot_type, 'y', fig=fig, verbose=verbose)
    except Exception as e:
        if verbose:
            print('[ERROR]: issue getting y-ticks')
            print('  ', str(e))
            if error_out:
                import sys; sys.exit()
            else:
                return '', True
        success_plot = False
        #continue
    
    # for colorbars
    colorbar_ticks = []
    cbar_bbox = None; cbar_text = None
    if 'color bar' in data_from_plot:
        colorbar = data_from_plot['color bar']
        if data_from_plot['color bar params']['side'] == 'left' \
            or data_from_plot['color bar params']['side'] == 'right':
            cbarax = 'y'
        else:
            cbarax = 'x'
        try:
            colorbar_ticks = get_ticks(colorbar, plot_type, cbarax, fig=fig, verbose=verbose)
        except Exception as e:
            if verbose:
                print('[ERROR]: issue getting colorbar ticks')
                print('  ', str(e))
                success_plot = False
                return '', True

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
        

        # is it an image of the sky? (WCAxes)
        if cbar_ax != []: # placeholder for no colorbar
            colorbar_label = None
            colorbar_offset_text = None
            if hasattr(cbar_ax, 'coords'):
                # have text
                for cbar_axc in cbar_ax.coords:
                    if cbar_axc._axislabels.get_text() != '':
                        cbar_text = cbar_axc._axislabels.get_text()
                        cbar_bbox = cbar_axc._axislabels.get_window_extent()
                        colorbar_label = {'text':cbar_text, 
                                            'xmin':cbar_bbox.x0, 
                                            'ymin':cbar_bbox.y0,
                                            'xmax':cbar_bbox.x1,
                                            'ymax':cbar_bbox.y1}
                        if colorbar_verbose: print('colorbar_label is (WCAxes):', colorbar_label)
                        #print("HAVE TO CHECK FOR OFFSET TEXT")
                        #import sys; sys.exit()
                # try this
                yoff = cbar_ax.yaxis.get_offset_text() #get_text()
                xoff = cbar_ax.yaxis.get_offset_text()
                if xoff.get_text() != '' and yoff.get_text() != '':
                    print('both x & y have offset text and I dont know how to deal!')
                    if error_out:
                        import sys; sys.exit()
                    else:
                        return '', True
                elif xoff.get_text() != '':
                    cbar_offset_text = xoff
                else:
                    cbar_offset_text = yoff # either something or nothing
                if cbar_offset_text.get_text() != '':
                    cbar_ot_bb = cbar_offset_text.get_window_extent()
                    colorbar_offset_text = {'text':cbar_offset_text.get_text(), 
                                            'xmin':cbar_ot_bb.x0, 
                                            'ymin':cbar_ot_bb.y0,
                                            'xmax':cbar_ot_bb.x1,
                                            'ymax':cbar_ot_bb.y1}
            elif hasattr(cbar_ax, '_colorbar'):
                # check both x & y
                if cbar_ax.yaxis.label.get_text() != '':
                    cbar_text = cbar_ax.yaxis.label.get_text()
                    cbar_bbox = cbar_ax.yaxis.label.get_window_extent()
                    cbar_offset_text = cbar_ax.yaxis.get_offset_text()
                elif cbar_ax.xaxis.label.get_text() != '':
                    cbar_text = cbar_ax.xaxis.label.get_text()
                    cbar_bbox = cbar_ax.xaxis.label.get_window_extent()
                    cbar_offset_text = cbar_ax.xaxis.get_offset_text()
                else:
                    if colorbar_verbose: print('no label for colorbar!')
                    cbar_text = ''
                    cbar_offset_text = cbar_ax.xaxis.get_offset_text() # placeholder
                # cbar_text = cbar_ax.get_ylabel()
                # cbar_bbox = cbar_ax.get_window_extent()
                if cbar_text != '':
                    colorbar_label = {'text':cbar_text, 
                                            'xmin':cbar_bbox.x0, 
                                            'ymin':cbar_bbox.y0,
                                            'xmax':cbar_bbox.x1,
                                            'ymax':cbar_bbox.y1}
                    if colorbar_verbose: print('colorbar_label is (matplotlib):', colorbar_label)
                #import sys; sys.exit()
                if cbar_offset_text.get_text() != '':
                    cbar_ot_bb = cbar_offset_text.get_window_extent()
                    colorbar_offset_text = {'text':cbar_offset_text.get_text(), 
                                            'xmin':cbar_ot_bb.x0, 
                                            'ymin':cbar_ot_bb.y0,
                                            'xmax':cbar_ot_bb.x1,
                                            'ymax':cbar_ot_bb.y1}
            else:
                print('not sure what kind of colorbar this is!')
                if error_out:
                    import sys; sys.exit()
                else:
                    return '', True

            if colorbar_label is None:
                if colorbar_verbose: print('colorbar_label is None for this plot!')
                #import sys; sys.exit()
            else:
                datas['color bar']['label'] = colorbar_label.copy()

            if colorbar_offset_text is None:
                pass
            else:
                datas['color bar']['offset text'] = colorbar_offset_text.copy()


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
    if len(colorbar_ticks) > 0:
        xtmp = []
        for xt in colorbar_ticks:
            l = {'data':xt[0], 'xmin': xt[1], 
                    'ymin': xt[2], 
                    'xmax':xt[3], 'ymax':xt[4], 
                'tx':xt[5], 'ty':xt[6]}
            xtmp.append(l)
        datas['color bar ticks'] = xtmp.copy()
        
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
            
    return datas, err


# -----------------------------------------------
# ------------ FOR TESTING and RERUNS ----------
# -----------------------------------------------

def get_new_title(title_fontsize, rng, fontsizes, font_names, fontsize_min=8):

    title_fontsize -= 1
    err = False

    if title_fontsize < fontsize_min:
        err = True
        print("[ERROR]: can't make font size smaller, gonna grab new words for title and try again...")
        title_fontsize, _, _, _, _, _, _ = get_font_info(fontsizes, font_names, rng=rng)

    return title_fontsize, err


def get_new_xylabels(xlabel_fontsize, ylabel_fontsize, rng, fontsizes, font_names, fontsize_min = 8):
    xlabel_fontsize -= 1
    ylabel_fontsize -= 1    
    err = False

    if xlabel_fontsize < fontsize_min or ylabel_fontsize < fontsize_min:
        err = True
        print("[ERROR]: can't make font size smaller, gonna grab new words for x/y axis labels and try again...")
        _, _, xlabel_fontsize, ylabel_fontsize, _, _, _ = get_font_info(fontsizes, font_names, rng=rng)
    return xlabel_fontsize, ylabel_fontsize, err



import matplotlib.pyplot as plt
from copy import deepcopy

from sys import maxsize as maxint
from .plot_qa_utils import log_scale_ax
from .data_utils import get_data

def reset_all_params(plot_params_input, popular_nouns, 
                     plot_styles, fontsizes, font_names, dpi_params,
                     aspect_fig_params, plot_type, rng,
                     colormaps = plt.colormaps(), verbose=True):
    """
    If there is major failure, reset all randomized params and try to remake the plot.
    """
    plot_params_in = deepcopy(plot_params_input)
    success_titles = False
    seed_aspect = np.random.randint(maxint)
    rng_aspect = np.random.default_rng(seed_aspect)
    seed_titles = np.random.randint(maxint)
    rng_titles = np.random.default_rng(seed_titles)
    aspect_fig = rng_aspect.uniform(low=aspect_fig_params['min'], high=aspect_fig_params['max'])
    xlabels_pull = deepcopy(popular_nouns)
    ylabels_pull = deepcopy(popular_nouns)
    titles_pull = deepcopy(popular_nouns)
    seed_outer= np.random.randint(maxint)
    rng_outer = np.random.default_rng(seed_outer)
    color_map = rng_outer.choice(colormaps) # choose a color map
    plot_style = rng_outer.choice(plot_styles) # choose a plotting style
    # dist params
    dist_params = plot_params_in[plot_type]['distribution'] 
    choices_d = []; probs_d = []
    for k,v in dist_params.items():
        choices_d.append(k)
        probs_d.append(v['prob'])
    distribution_type = rng_outer.choice(choices_d, p=probs_d)
    # get fonts
    title_fontsize, colorbar_fontsize, xlabel_fontsize, ylabel_fontsize, \
        xlabel_ticks_fontsize, ylabel_ticks_fontsize, \
                            csfont = get_font_info(fontsizes, font_names, rng=rng_titles)
    # get DPI
    dpi = int(rng_outer.uniform(low=dpi_params['min'], high=dpi_params['max']))
    # get data
    # pull xmin/xmax for hists
    xmin,xmax = log_scale_ax()
    plot_params_in[plot_type]['xmin']=xmin
    plot_params_in[plot_type]['xmax']=xmax
    if verbose: print('xmin, xmax = ', xmin, xmax)

    # get plot data
    success_get_data = False
    while not success_get_data:
        data_for_plot = get_data(plot_params_in[plot_type], 
                                plot_type=plot_type, distribution=distribution_type, rng=rng)
        if len(data_for_plot['xs']) > 0 and plot_type == 'histogram':
            success_get_data = True


    return success_titles, rng_aspect, rng_titles, aspect_fig, xlabels_pull, ylabels_pull, \
        titles_pull, rng_outer, color_map, plot_style, dist_params, distribution_type, \
    title_fontsize, colorbar_fontsize, xlabel_fontsize, ylabel_fontsize, \
        xlabel_ticks_fontsize, ylabel_ticks_fontsize, \
                            csfont, dpi, xmin,xmax, plot_params_in, data_for_plot 



# shorten checks
def check_aspect(datas, success_titles, aspect_fig, aspect_cut, aspect_fig_params):
    """
    Check if the resulting aspect ratio of the figure is super small.  If so, flag to re-run stuff.
    """
    # 1. Check for square with weird aspect ratio
    success_aspect = True
    # check for thin/fat squares
    for k,v in datas.items():
        if 'plot' in k: # a plot key
            w = v['square']['xmax']-v['square']['xmin']
            h = v['square']['ymax']-v['square']['ymin']
            if w/h > aspect_cut['max'] or w/h < aspect_cut['min']:
                success_aspect = False

    if not success_aspect:
        print('[ERROR]: aspect ratio off')
        # regenerate figure with new aspect ratio
        success_titles = False
        seed_aspect = np.random.randint(maxint)
        rng_aspect = np.random.default_rng(seed_aspect)
        aspect_fig = rng_aspect.uniform(low=aspect_fig_params['min'], high=aspect_fig_params['max'])

    return success_titles, success_aspect, aspect_fig


def check_labels_titles_off_page(datas, width, height, success_titles, 
                                 xlabels_pull, ylabels_pull, titles_pull,
                                 xlabel_fontsize, ylabel_fontsize, rng_titles, 
                                 popular_nouns, title_fontsize, 
                                 fontsizes, font_names,
                                 fontsize_min = 8, verbose=True):
    """
    Check if any of the x-axis labels, y-axis labels, or titles is off the page of the figure.  
    If so, try to make the fontsize smaller and re-run.  If the fontsize is smaller than 
    the `fontsize_min` parameter, flag to re-pull new x/y axis labels and titles.
    """
    # check for overlaps of x/y axis labels, tickmarks or anything outside of the figbox
    success_axis_labels = True
    success_title_label = True
    for k,v in datas.items():
        if 'plot' in k: # a plot key
            # check if x is out of bounds
            for axislabel in ['xlabel', 'ylabel']:
                if v[axislabel]['xmin'] < 0 or v[axislabel]['xmax'] > width or \
                    v[axislabel]['ymin'] < 0 or v[axislabel]['ymax'] > height:
                    success_axis_labels = False
            # also title
            if 'title' in v:
                if v['title']['xmin'] < 0 or v['title']['xmax'] > width or \
                    v['title']['ymin'] < 0 or v['title']['ymax'] > height:
                    success_title_label = False

        if not success_axis_labels:
            if verbose: 
                print('[ERROR]: x/y axis off page, making smaller...')
                print('   x/y axis labels:', xlabels_pull, ylabels_pull)
            xlabel_fontsize, ylabel_fontsize, err = get_new_xylabels(xlabel_fontsize, ylabel_fontsize, rng_titles, fontsizes, font_names,
                                                                                fontsize_min = fontsize_min)
            # regenerate x/y titles if fontsize too small
            success_titles = False
            if err: # regenerate labels
                xlabels_pull = deepcopy(popular_nouns)
                ylabels_pull = deepcopy(popular_nouns)
                # reset RNG for labels
                seed_titles = np.random.randint(maxint)
                rng_titles = np.random.default_rng(seed_titles)
            else:
                if verbose: print('   new fontsizes (x,y):', xlabel_fontsize, ylabel_fontsize)

        if not success_axis_labels and err:
            return success_titles, xlabel_fontsize, ylabel_fontsize, title_fontsize, xlabels_pull, ylabels_pull, titles_pull, rng_titles, success_title_label, success_axis_labels

        if not success_title_label:
            if verbose:
                print('[ERROR]: title axis off page, making smaller...')
            title_fontsize, err = get_new_title(title_fontsize, rng_titles, fontsizes, font_names,
                                                fontsize_min=fontsize_min)
            # regenerate titles if fontsize too small
            success_titles = False
            if err:
                titles_pull = deepcopy(popular_nouns)
                # reset RNG for labels
                seed_titles = np.random.randint(maxint)
                rng_titles = np.random.default_rng(seed_titles)
            else:
                if verbose: print('   new fontsize:', title_fontsize)

            #break # I think...         
    return success_titles, xlabel_fontsize, ylabel_fontsize, title_fontsize, xlabels_pull, ylabels_pull, titles_pull, rng_titles, success_title_label, success_axis_labels


from .metric_utils.utilities import isRectangleOverlap

def collect_boxes(datas, grace_ticks=5):
    """
    Collect all potential bounding boxes in a figure (can be for multi-panel figures too).
    """
    # now check overlapping boxes
    # first make boxes
    boxes_check = []
    success_boxes = True
    for k,v in datas.items():
        if 'plot' in k: # a plot key
            # square!
            boxes_check.append(([v['square']['xmin'], v['square']['ymin'], 
                                        v['square']['xmax'], v['square']['ymax']], 'square'))
            if 'title' in v:
                boxes_check.append( ([v['title']['xmin'], v['title']['ymin'], 
                                        v['title']['xmax'], v['title']['ymax']], 'title') )
            # xlabel
            boxes_check.append( ([v['xlabel']['xmin'], v['xlabel']['ymin'], 
                                        v['xlabel']['xmax'], v['xlabel']['ymax']], 'xlabel') )
            # ylabel
            boxes_check.append( ([v['ylabel']['xmin'], v['ylabel']['ymin'], 
                                        v['ylabel']['xmax'], v['ylabel']['ymax']],'ylabel')  )
            # x/yticks
            for t in ['x','y']:
                for tick in v[t+'ticks']:
                    # ignore things that are outside square
                    if tick['tx'] < v['square']['xmin']-grace_ticks or tick['tx'] > v['square']['xmax']+grace_ticks or \
                      tick['ty'] < v['square']['ymin']-grace_ticks or tick['ty'] > v['square']['ymax']+grace_ticks:
                        continue
                    boxes_check.append( ([tick['xmin'],tick['ymin'],tick['xmax'],tick['ymax']],t+'-tick labels') )
            # x/y offset labels
            for t in ['x','y']:
                if t + '-offset text' in v:
                    tick = v[t + '-offset text']
                    boxes_check.append( ([tick['xmin'],tick['ymin'],tick['xmax'],tick['ymax']],t+'-offset text') )

            # if colorbar, add this
            if 'color bar' in v:
                boxes_check.append(([v['color bar']['xmin'], v['color bar']['ymin'], 
                                        v['color bar']['xmax'], v['color bar']['ymax']],'colorbar'))
                # also check for label
                if 'label' in v['color bar']:
                    xmin = v['color bar']['label']['xmin']
                    ymin = v['color bar']['label']['ymin']
                    xmax = v['color bar']['label']['xmax']
                    ymax = v['color bar']['label']['ymax']
                    boxes_check.append(([xmin,ymin,xmax,ymax],'colorbar label'))
                # and offset text
                if 'offset text' in v['color bar']:
                    xmin = v['color bar']['offset text']['xmin']
                    ymin = v['color bar']['offset text']['ymin']
                    xmax = v['color bar']['offset text']['xmax']
                    ymax = v['color bar']['offset text']['ymax']
                    boxes_check.append(([xmin,ymin,xmax,ymax],'colorbar offset text'))
                    
            # colorbar ticks
            if 'color bar ticks' in v:
                for tick in v['color bar ticks']:
                    boxes_check.append( ([tick['xmin'],tick['ymin'],tick['xmax'],tick['ymax']], 'colorbar tick') )

    # now run and check all boxes -- look for overlap of all boxes
    names_overlap = []
    for ib1,(box1,name1) in enumerate(boxes_check):
        for ib2,(box2,name2) in enumerate(boxes_check):
            if ib1 != ib2: # ib1 < ib2?
                if isRectangleOverlap( box1, box2 ):
                    names_overlap.append( (name1, name2) )
                    success_boxes = False

    return success_boxes, boxes_check, names_overlap


def update_fonts_boxes_overlap(names_overlap, success_titles, rng_titles,popular_nouns,
                            xlabels_pull, ylabels_pull, titles_pull,
                            xlabel_ticks_fontsize, ylabel_ticks_fontsize, 
                            xlabel_fontsize, ylabel_fontsize, title_fontsize,
                               verbose=True, fontsize_min=8):
    """
    Check if there are any overlapping bounding boxes.  If so, try to update fonts 
    if possible, and if not, flag to regenerate the figure.
    """
    if verbose: print('[ERROR]: overlapping boxes!')
    # figure out what to do for each box
    # get unique overlaps
    s1 = np.unique(names_overlap, axis=0)
    # sort
    s2 = np.unique(np.sort(s1, axis=1),axis=0)

    ######## check boxes ##########
    reset_fonts = False
    for b1,b2 in s2:
        # for ticks overlapping with things
        if 'tick' in b1 and 'tick' in b2: # overlapping ticks, smallen
            xlabel_ticks_fontsize -= 1
            ylabel_ticks_fontsize -= 1
            success_titles = False
            break
        elif ( ('tick' in b1) and ('tick' not in b2) and  ('xlabel' in b2) ) or ( ('tick' in b2) and ('tick' not in b1) and  ('xlabel' in b1) ): # xlabel cross over
            if xlabel_fontsize > xlabel_ticks_fontsize: # axis label still bigger
                xlabel_ticks_fontsize -= 1
                ylabel_ticks_fontsize -= 1
            else:
                xlabel_fontsize -= 1
            success_titles = False
            break            
        elif ( ('tick' in b1) and ('tick' not in b2) and  ('ylabel' in b2) ) or ( ('tick' in b2) and ('tick' not in b1) and  ('ylabel' in b1) ): # ylabel cross over
            if ylabel_fontsize > ylabel_ticks_fontsize: # axis label still bigger
                xlabel_ticks_fontsize -= 1
                ylabel_ticks_fontsize -= 1
            else:
                ylabel_fontsize -= 1
            success_titles = False
            break   
        elif ( ('tick' in b1) and ('tick' not in b2) and  ('title' in b2) ) or ( ('tick' in b2) and ('tick' not in b1) and  ('title' in b1) ): # title cross over
            if title_fontsize > ylabel_ticks_fontsize or title_fontsize > xlabel_ticks_fontsize: # axis label still bigger
                xlabel_ticks_fontsize -= 1
                ylabel_ticks_fontsize -= 1
            else:
                title_fontsize -= 1
            success_titles = False
            break   
        elif ( ('xlabel' in b1) or ('xlabel' in b2) or ('ylabel' in b1) or ('ylabel' in b2) ): # overlap
            ylabel_fontsize -= 1
            xlabel_fontsize -= 1
            success_titles = False
            break
        elif ('title' in b1) or ('title' in b2):
            title_fontsize -= 1
            success_titles = False
            break
        else: # no idea!
            reset_fonts = True

    # full rest of fonts
    for fs in [xlabel_fontsize, ylabel_fontsize, ylabel_ticks_fontsize, xlabel_ticks_fontsize, title_fontsize]:
        if fs < fontsize_min:
            reset_fonts = True    

    if reset_fonts:
        xlabels_pull = deepcopy(popular_nouns)
        ylabels_pull = deepcopy(popular_nouns)
        titles_pull = deepcopy(popular_nouns)
        # reset RNG for labels
        seed_titles = np.random.randint(maxint)
        rng_titles = np.random.default_rng(seed_titles)\
        
    return success_titles, xlabel_fontsize, ylabel_fontsize, ylabel_ticks_fontsize, xlabel_ticks_fontsize, title_fontsize, xlabels_pull, ylabels_pull, titles_pull, rng_titles



# based on seed, make random number generator, see: https://numpy.org/doc/2.2/reference/random/generator.html
def set_all_seeds(rng_outer=np.random, rng=np.random, rng_titles=np.random, rng_font=np.random, rng_aspect=np.random,
                  reset_outer = False, reset_inner = False, reset_titles=False, reset_fonts = False, reset_aspect = False,
                  verbose=True):
    if reset_outer:
        seed_outer = np.random.randint(maxint)
        if verbose: print('seed_outer =',seed_outer)
        rng_outer = np.random.default_rng(seed_outer)

    # "Inner" seed -- for things like distributions and whatnot
    if reset_inner:
        seed = np.random.randint(maxint)
        if verbose: print('seed, inner = ', seed)
        rng = np.random.default_rng(seed)

    # for titles
    if reset_titles:
        seed_titles = np.random.randint(maxint)
        if verbose: print('seed_titles:', seed_titles)
        rng_titles = np.random.default_rng(seed_titles)

    # for fonts
    if reset_fonts:
        seed_font = np.random.randint(maxint)
        if verbose: print('seed_font:', seed_font)
        rng_font = np.random.default_rng(seed_font)

    # aspect ratio
    if reset_aspect:
        seed_aspect = np.random.randint(maxint)
        if verbose: print('seed_aspect:', seed_aspect)
        rng_aspect = np.random.default_rng(seed_aspect)

    return rng_outer, rng, rng_titles, rng_font, rng_aspect