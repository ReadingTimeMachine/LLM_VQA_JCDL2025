# utils to get locations of different objects in pixel coords

import numpy as np
import cv2 as cv

def get_errorbar_pixels(datain,ax):
    # x/y error bars
    xerror_bars = []
    error_caps_out = []
    # Loop through each of the error bars
    for bbar in datain:
        for bar in bbar[0].get_segments():
            # Get the position of the bar in data coordinates
            xy_data = bar[0]  # lower cap
            xy_data2 = bar[1]  # upper cap
        
            # Convert the position from data coordinates to pixel coordinates
            xy_pixels = ax.transData.transform([xy_data])
            xy_pixels2 = ax.transData.transform([xy_data2])
        
            error_caps_out.append((xy_pixels, xy_pixels2))

    for bottom, top in error_caps_out:
        xmin,ymin = bottom.flatten()
        xmax,ymax = top.flatten()
        xerror_bars.append( [xmin,ymin,xmax,ymax] )

    return xerror_bars


def transform(x,y,ax,height, isImage=False):
    xy_pixels = ax.transData.transform(np.vstack([x,y]).T)
    xpix, ypix = xy_pixels.T
    
    # In matplotlib, 0,0 is the lower left corner, whereas it's usually the upper 
    # left for most image software, so we'll flip the y-coords...
    if not isImage:
        ypix = height - ypix
    else:
        #print('hi')
        #ypix = -1.0*(height-ypix)
        #ypix = height + ypix # WHY??
        pass
    return xpix,ypix

def get_data_pixel_locations(data_from_plot, plot_type, ax, width, height):
    datas = data_from_plot['data']
    if plot_type == 'line':
        x1,y1 = [], []
        # Get the x and y data and transform it into pixel coordinates
        for data in datas:
            #if not isScatter:
            x, y = data.get_data()

            xpix,ypix = transform(x,y,ax,height)
        
            x1.append(xpix)
            y1.append(ypix)
                            
        dataout = {'xs':x1, 'ys':y1}

        xerrs, yerrs = [], []
        if 'x error bars' in data_from_plot: # have x-error bars
            for i in range(len(datas)):
                xerrs.append(get_errorbar_pixels([data_from_plot['x error bars'][i]], ax))
            dataout['x error bars'] = xerrs
        if 'y error bars' in data_from_plot: # have y-error bars
            for i in range(len(datas)):
                yerrs.append(get_errorbar_pixels([data_from_plot['y error bars'][i]], ax))
            dataout['y error bars'] = yerrs

        return dataout
    elif plot_type == 'scatter':
        #x1,y1 = [], []
        x = datas.get_offsets().data[:,0]
        y = datas.get_offsets().data[:,1]
        
        # # In matplotlib, 0,0 is the lower left corner, whereas it's usually the upper 
        # # left for most image software, so we'll flip the y-coords...
        # ypix = height - ypix
        xpix,ypix = transform(x,y,ax,height)

        # get colors
        colors = datas.get_facecolors()

        dataout = {'xs':xpix, 'ys':ypix, 'colors':colors}

        # # x/y error bars
        if 'x error bars' in data_from_plot: # have x-error bars
            dataout['x error bars'] = get_errorbar_pixels(data_from_plot['x error bars'], ax)
        if 'y error bars' in data_from_plot: # have y-error bars
            dataout['y error bars'] = get_errorbar_pixels(data_from_plot['y error bars'], ax)
        
        return dataout
    elif plot_type == 'histogram':
        # derive plot points
        data_heights = datas[0] # heights of DATA values
        bin_edges = datas[1] # bin edges of DATA values
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        rwidth = data_from_plot['plot params']['rwidth']

        xpix_center,ypix_center = transform(bin_centers, data_heights, ax,height)

        x_left = bin_centers - 0.5*rwidth*np.abs(bin_edges[:-1] - bin_edges[1:])
        x_right = bin_centers + 0.5*rwidth*np.abs(bin_edges[:-1] - bin_edges[1:])
        
        xpix_right,ypix_right = transform(x_right, data_heights, ax,height)
        xpix_left,ypix_left = transform(x_left, data_heights, ax,height)

        # get colors
        bars = datas[2]
        colors = []
        for b in bars:
            colors.append(b.get_facecolor())

        dataout = {'xs': xpix_center, 'ys': ypix_center,
                   'xs_right':xpix_right, 'ys_right':ypix_right,
                   'xs_left':xpix_left, 'ys_left':ypix_left,
                  'colors':colors}

        # # x/y error bars
        if 'x error bars' in data_from_plot: # have x-error bars
            dataout['x error bars'] = get_errorbar_pixels(data_from_plot['x error bars'], ax)
        if 'y error bars' in data_from_plot: # have y-error bars
            dataout['y error bars'] = get_errorbar_pixels(data_from_plot['y error bars'], ax)
        return dataout
    elif plot_type == 'contour':
        xpix_c = []; ypix_c = []
        xpix_im = []; ypix_im = []
        image = {}; contour = {}
        if 'image' in datas: # has image
            xmin,xmax,ymin,ymax = datas['image'].get_extent()
            sh = datas['image'].get_array().shape
            xs1 = np.linspace(xmin,xmax, sh[1]+1) # bin edges
            ys1 = np.linspace(ymin,ymax, sh[0]+1) # bin edges
            xsc1 = 0.5*(xs1[1:] + xs1[:-1])
            ysc1 = 0.5*(ys1[1:] + ys1[:-1])
            # unravel
            xs = []; ys = []
            for x in xs1:
                for y in ys1:
                    xs.append(x); ys.append(y)
            xpix_im,ypix_im = transform(xs,ys, ax, height)
            xsc = []; ysc = []
            for x in xsc1:
                for y in ysc1:
                    xsc.append(x); ysc.append(y)
            xpix_imc,ypix_imc = transform(xsc,ysc, ax, height)
            # colors
            #data_from_plot['data']['image'].cmap
            colors = datas['image'].cmap(datas['image'].norm(np.unique(datas['image'].get_array())))
            image = {'xs':xpix_im, 'ys':ypix_im, 'xsc':xpix_imc, 'ysc':ypix_imc, 'colors':colors}.copy()
        if 'contour' in datas: # has image
            #xmin,xmax,ymin,ymax = datas['contour'].get_extent()
            #sh = datas['contour'].get_array().shape
            #>> cs = plt.contour(x,y,m, [9.5])
            # get contour verticies
            # paths = datas['contour'].collections[0].get_paths()
            # xs1 = np.linspace(xmin,xmax, sh[1]+1) # bin edges
            # ys1 = np.linspace(ymin,ymax, sh[0]+1) # bin edges
            # xsc1 = 0.5*(xs1[1:] + xs1[:-1])
            # ysc1 = 0.5*(ys1[1:] + ys1[:-1])
            xs,ys = [], []
            for d in datas['contour'].allsegs:
                if len(d) > 1:
                    for dd in d:
                        for x,y in dd:
                            xs.append(x); ys.append(y)  # for element j, in level i

            xpix_imv,ypix_imv = transform(xs,ys, ax, height)
            # colors
            #data_from_plot['data']['image'].cmap
            colors = datas['contour'].cmap(datas['contour'].norm(np.unique(datas['contour'].get_array())))
            contour = {'xs':xpix_imv, 'ys':ypix_imv, 'colors':colors}.copy()

        dataout = {
                'image':image,
                'contour':contour
                  }
        return dataout
    else:
        print('get_data_pixel_locations: not implemented!', 'Plot type:', plot_type)
        import sys; sys.exit()



def plot_color_bar(v,img,imgplot):
# colormap
    xmin,ymin = int(round(v['color bar']['xmin'])), int(round(img.shape[0]-v['color bar']['ymin']))
    xmax,ymax = int(round(v['color bar']['xmax'])), int(round(img.shape[0]-v['color bar']['ymax']))
    cv.rectangle(imgplot, (xmin,ymin), (xmax,ymax), (255, 0, 0), 3)
    # colormap ticks
    if 'color bar ticks' in v:
        for d in v['color bar ticks']:
            xmin,ymin,xmax,ymax = int(d['xmin']),int(img.shape[0]-d['ymin']),int(d['xmax']),int(img.shape[0]-d['ymax'])
            # check if we should have it or not
            if v['color bar']['params']['side'] == 'bottom' or v['color bar']['params']['side'] == 'top':
                if d['tx']>=v['color bar']['xmin'] and d['tx']<=v['color bar']['xmax']:
                    cv.rectangle(imgplot, (xmin,ymin), (xmax,ymax), (255, 0, 0), 3)
                    cv.circle(imgplot, (int(d['tx']), int(img.shape[0]-d['ty'])), 10, (255,0,0), -1)
            else: # side
                if d['ty']>=v['color bar']['ymin'] and d['ty']<=v['color bar']['ymax']:
                    cv.rectangle(imgplot, (xmin,ymin), (xmax,ymax), (255, 0, 0), 3)
                    cv.circle(imgplot, (int(d['tx']), int(img.shape[0]-d['ty'])), 10, (255,0,0), -1)
    return imgplot





