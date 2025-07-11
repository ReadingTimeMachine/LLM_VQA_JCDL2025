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
    for ptype in ['contour', 'image of the sky']:
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

        # # image of the sky
        # p = 0.0
        # if 'image of the sky' in plot_types_params['contour']['distribution']:
        #     for dist,pval in plot_types_params['contour']['distribution']['image of the sky']['distribution or sky'].items():
        #         p += pval
        #     if p != 1.0:
        #         ps = plot_types_params['contour']['distribution']['image of the sky']['distribution or sky'].copy()
        #         for k,v in plot_types_params['contour']['distribution']['image of the sky']['distribution or sky'].items():
        #             ps[k]['prob'] = v['prob']/p
        #         plot_types_params['contour']['distribution']['image of the sky']['distribution or sky'] = ps
    # distribution probabilities
    #print('')
    for k1 in ['line','histogram','scatter','contour', 'image of the sky']:
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


# image of the sky, from coordinate helper of astropy
def get_xy_dx_dy(tl, fig, i, bb, axis='b'):
    """
    i is number of the tick
    """                  
    #renderer = fig.canvas.renderer 
    # Set initial position and find bounding box
    # self.set_text(self.text[axis][i])
    # self.set_position((x, y))
    # bb = super().get_window_extent(renderer)
    pad = fig.canvas.renderer.points_to_pixels(tl.get_pad() + tl._tick_out_size)
    x, y = tl._frame.parent_axes.transData.transform(tl.data[axis][i])


    # Find width and height, as well as angle at which we
    # transition which side of the label we use to anchor the
    # label.
    width = bb.width
    height = bb.height

    # Project axis angle onto bounding box
    ax = np.cos(np.radians(tl.angle[axis][i]))
    ay = np.sin(np.radians(tl.angle[axis][i]))

    # Set anchor point for label
    if np.abs(tl.angle[axis][i]) < 45.0:
        dx = width
        dy = ay * height
    elif np.abs(tl.angle[axis][i] - 90.0) < 45:
        dx = ax * width
        dy = height
    elif np.abs(tl.angle[axis][i] - 180.0) < 45:
        dx = -width
        dy = ay * height
    else:
        dx = ax * width
        dy = -height

    dx *= 0.5
    dy *= 0.5

    # Find normalized vector along axis normal, so as to be
    # able to nudge the label away by a constant padding factor

    dist = np.hypot(dx, dy)

    ddx = dx / dist
    ddy = dy / dist

    dx += ddx * pad
    dy += ddy * pad

    x = x - dx
    y = y - dy

    return x, y, dx, dy


def get_ticks_imgOfSky(ax1, axis, fig, verbose=False, 
                       subtracty = False):

    # how many axes for this ax object
    ncoords = len(ax1.coords._coords)

    # determine x/y axis
    if axis == 'x':
        visible_coords = ['b','t']
    elif axis == 'y':
        visible_coords = ['l','r']
    else:
        print('[ERROR]: in get_ticks_imgOfSky in synthetic_fig_utils -- no axis for:', axis)
        import sys; sys.exit()

    # get x/y size of image
    xsize,ysize = fig.get_size_inches()*fig.dpi # size in pixels

    ticks_out_full = []

    for icoord in range(ncoords): # all coords (x/y) in axis
        # is axis visible? get list
        is_vis = ax1.coords[icoord]._ticklabels.get_visible_axes()

        for sidecoord in visible_coords:
            if sidecoord not in is_vis or sidecoord == '#':
                #if verbose: print('coord not visible:', sidecoord)
                continue

            # tick labels
            ticklabels1 = ax1.coords[icoord]._ticklabels

            # tick locations
            ticks = ax1.coords[icoord]._ticks

            # text
            texts = ax1.coords[icoord]._ticklabels.text[sidecoord]

            nticks = len(ticklabels1.text[sidecoord])

            # format: (text, xmin,ymin, xmax, ymax, tick_x, tick_y)
            ticks_out = []

            for i in range(nticks):
                #tout = {}
                bb = ticklabels1._get_bb(sidecoord,i,ax1.get_figure().canvas.renderer)
                # if bb is None, this means that there is no tick label there (empty)
                if bb is None:
                    continue
                xmin,ymin, dx, dy = get_xy_dx_dy(ticklabels1, fig, i, bb, axis=sidecoord)
                xmin -= bb.width/2
                ymin -= bb.height/2

                xmax = xmin + bb.width
                ymax = ymin + bb.height

                if subtracty:
                    ymin = ysize-ymin
                    ymax = ysize-ymax

                # get tick locations too
                tickxy = ticks._frame.parent_axes.transData.transform(ticks.ticks_locs[sidecoord][i][0])
                xt = tickxy[0]
                yt = tickxy[1]
                if subtracty:
                    yt = yt-tickxy[1]

                # format: (text, xmin,ymin, xmax, ymax, tick_x, tick_y)
                ticks_out.append((texts[i],xmin,ymin,xmax,ymax,xt,yt))

            if len(ticks_out)>0:
                ticks_out_full.append(ticks_out)
                #print('ticksout', ticks_out)

    # check
    if len(ticks_out_full) > 1:
        print('[ERROR]: more than 1 x/y axis not implemented!')
        print('  In: get_ticks_imgOfSky in synthetic_fig_utils')
        print(len(ticks_out_full))
        print(ticks_out_full)
        import sys; sys.exit()

    ticks = ticks_out_full[0]
    return ticks




def get_ticks(ax, plot_type, axis, fig=None, dpi=None, minor=False, verbose = False):
    if plot_type != 'image of the sky':
        if axis == 'x':
            ticklabels,ticklines = ax.get_xticklabels(), ax.get_xticklines(minor=minor)
        elif axis == 'y':
            ticklabels,ticklines = ax.get_yticklabels(), ax.get_yticklines(minor=minor)
        else:
            print('[ERROR]: in "get_ticks" in synthetic_fig_utils -- no axis type for:', axis)
            import sys; sys.exit()

        # JPN -- default is None DPI and fig!
        ticks = get_ticks_not_imgOfSky(ticklabels, ticklines, fig=None, dpi=None)
    else:
        ticks = get_ticks_imgOfSky(ax, axis, fig, verbose=verbose)


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
    #if plot_type != 'image of the sky':
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


