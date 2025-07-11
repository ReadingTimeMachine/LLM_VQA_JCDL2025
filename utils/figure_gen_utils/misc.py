import numpy as np
import cv2 as cv

def log_scale_ax(scale_exp = {'min':0.01, 'max':5.0}, 
                 scale_factor = {'min':0.1, 'max':10}, 
                 npoints=1,
                 verbose=False):
    """
    npoints : how many objects do we want?  only 1 is implemented right now
    """
    if npoints != 1:
        print('only npoints=1 is implemented!')
        import sys; sys.exit()
    scale_input = np.random.uniform(low=scale_exp['min'], high=scale_exp['max'])
    n = np.random.exponential(scale_input)
    scale = 10**n
    scale_factor_low = np.random.uniform(low=scale_factor['min'], high=scale_factor['max'])
    xmin = scale - scale_factor_low*scale
    scale_factor_high = np.random.uniform(low=scale_factor['min'], high=scale_factor['max'])
    xmax = scale + scale_factor_high*scale
    if verbose: print(scale_input, n, scale, xmin,xmax)
    return xmin,xmax

def plot_color_bar(v,img,imgplot, color=(255,0,0), lthick=3, verbose=True):
# colormap
    xmin,ymin = int(round(v['color bar']['xmin'])), int(round(img.shape[0]-v['color bar']['ymin']))
    xmax,ymax = int(round(v['color bar']['xmax'])), int(round(img.shape[0]-v['color bar']['ymax']))
    cv.rectangle(imgplot, (xmin,ymin), (xmax,ymax), color, lthick)
    # colormap ticks
    if 'color bar ticks' in v:
        for d in v['color bar ticks']:
            xmin,ymin,xmax,ymax = int(d['xmin']),int(img.shape[0]-d['ymin']),int(d['xmax']),int(img.shape[0]-d['ymax'])
            # check if we should have it or not
            if v['color bar']['params']['side'] == 'bottom' or v['color bar']['params']['side'] == 'top':
                if d['tx']>=v['color bar']['xmin'] and d['tx']<=v['color bar']['xmax']:
                    cv.rectangle(imgplot, (xmin,ymin), (xmax,ymax), color, lthick)
                    cv.circle(imgplot, (int(d['tx']), int(img.shape[0]-d['ty'])), 10, color, -1)
            else: # side
                if d['ty']>=v['color bar']['ymin'] and d['ty']<=v['color bar']['ymax']:
                    cv.rectangle(imgplot, (xmin,ymin), (xmax,ymax), color, lthick)
                    cv.circle(imgplot, (int(d['tx']), int(img.shape[0]-d['ty'])), 10, color, -1)
    # check for label
    if 'label' in v['color bar']:
        xmin = int(round(v['color bar']['label']['xmin']))
        xmax = int(round(v['color bar']['label']['xmax']))
        ymin = img.shape[0]-int(round(v['color bar']['label']['ymin']))
        ymax = img.shape[0]-int(round(v['color bar']['label']['ymax']))
        #ymin = int(round(v['color bar']['label']['ymin']))
        #ymax = int(round(v['color bar']['label']['ymax']))
        cv.rectangle(imgplot, (xmin,ymin), (xmax,ymax), color, lthick)
        # print("I AM IN PLOT COLOR BAR")
        # print(xmin,ymin,xmax,ymax)
    # check for offset text
    if 'offset text' in v['color bar']:
        xmin = int(round(v['color bar']['offset text']['xmin']))
        xmax = int(round(v['color bar']['offset text']['xmax']))
        ymin = img.shape[0]-int(round(v['color bar']['offset text']['ymin']))
        ymax = img.shape[0]-int(round(v['color bar']['offset text']['ymax']))
        cv.rectangle(imgplot, (xmin,ymin), (xmax,ymax), color, lthick)
    return imgplot


def add_annotations(imgplot, img, datas_plot, verbose = True, csize = 5, 
                    color_annotations = [(255,0,0),(255,0,125),(0,255,0),(0,255,255)]):
    """
    imgplot : image to plot on
    img : orig image 

    returns : image with annotations plotted in "color_annotations", or other colors
    """

    color1 = color_annotations[0]
    color2 = color_annotations[1]
    color3 = color_annotations[2]
    color4 = color_annotations[3]

    iplot_count = 0
    for p,v in datas_plot.items():
        iplot_count += 1
        if 'plot' in p: # not figure stuffs... just yet
            out_string = ' ------ Plot #' + str(p.split('plot')[-1]) + ' ------ \n'
            out_string += 'Plot type:' + v['type'] + '\n'
            out_string += 'Distribution:' + v['distribution'] + '\n'
            if 'data params' in v['data']:
                if 'nclusters' in v['data']['data params']:
                    out_string += '  n_clusters = ' + str(v['data']['data params']['nclusters']) + '\n'
                if 'm1' in v['data']['data params']:
                    out_string += 'm_1 * x + b_1: m_1 & b_1 = ' + str(v['data']['data params']['m1']) + \
                      ' & ' + str(v['data']['data params']['a1']) + '\n'
                if 'm2' in v['data']['data params']:
                    out_string += 'm_2 * y + b_2: m_2 & b_2 = ' + str(v['data']['data params']['m2']) + \
                      ' & ' + str(v['data']['data params']['a2']) + '\n'
                if 'm' in v['data']['data params'] and 'a' in v['data']['data params']:
                    out_string += 'm * x + b: m & b = ' + str(v['data']['data params']['m']) + \
                      ' & ' + str(v['data']['data params']['a']) + '\n'
            if verbose: print(out_string)
            
            # square
            #print(v)
            d = v['square']
            xmin,ymin = int(d['xmin']),int(img.shape[0]-d['ymin'])
            xmax,ymax = int(d['xmax']),int(img.shape[0]-d['ymax'])
            cv.rectangle(imgplot, (xmin,ymin), (xmax,ymax), color1, 3)
            # title bounding box
            if 'title' in v: # has a title?
                d = v['title']
                xmin,ymin = int(d['xmin']),int(img.shape[0]-d['ymin'])
                xmax,ymax = int(d['xmax']),int(img.shape[0]-d['ymax'])
                cv.rectangle(imgplot, (xmin,ymin), (xmax,ymax), color1, 3)
            # xlabel bounding box
            d = v['xlabel']
            xmin,ymin = int(d['xmin']),int(img.shape[0]-d['ymin'])
            xmax,ymax = int(d['xmax']),int(img.shape[0]-d['ymax'])
            cv.rectangle(imgplot, (xmin,ymin), (xmax,ymax), color1, 3)
            # ylabel bounding box
            d = v['ylabel']
            xmin,ymin = int(d['xmin']),int(img.shape[0]-d['ymin'])
            xmax,ymax = int(d['xmax']),int(img.shape[0]-d['ymax'])
            cv.rectangle(imgplot, (xmin,ymin), (xmax,ymax), color1, 3)
    
            #print('PLOT:', p)
            if v['type'] == 'line':
                xs = v['data pixels']['xs']
                ys = v['data pixels']['ys']
                for xx,yy in zip(xs,ys):
                    for x,y in zip(xx,yy):
                        cv.circle(imgplot, (int(x), int(y)), csize, color1, -1)
            elif v['type'] == 'scatter':
                xs = v['data pixels']['xs']
                ys = v['data pixels']['ys']
                for x,y in zip(xs,ys):
                    cv.circle(imgplot, (int(x), int(y)), csize, color1, -1)
                # colorbar
                if 'color bar' in v:
                    imgplot = plot_color_bar(v,img,imgplot)
            elif v['type'] == 'histogram':
                # middle of bar
                xs = np.array(v['data pixels']['xs'])
                ys = np.array(v['data pixels']['ys'])
                for x,y in zip(xs,ys): # middle
                    cv.circle(imgplot, (int(x), int(y)), csize, color1, -1)
                # right
                xsr = np.array(v['data pixels']['xs_right'])
                ysr = np.array(v['data pixels']['ys_right'])
                for x,y in zip(xsr,ysr): # right
                    cv.circle(imgplot, (int(x), int(y)), csize, color1, -1)
                # left
                xsl = np.array(v['data pixels']['xs_left'])
                ysl = np.array(v['data pixels']['ys_left'])
                for x,y in zip(xsl,ysl): # left
                    cv.circle(imgplot, (int(x), int(y)), csize, color2, -1)
            elif v['type'] == 'contour':
                #xs = np.array
                #xs = []; ys = []
                #if 'image' in v['data pixels']:
                if v['data pixels']['image'] != {}:
                    xs = v['data pixels']['image']['xsc']
                    ys = v['data pixels']['image']['ysc']
                    #print('image')
                    #print(xs[:10], ys[:10])
                    for x,y in zip(xs,ys):
                        #print(x,y)
                        cv.circle(imgplot, (int(x), int(y)), csize, color3, -1)
                #if 'contour' in v['data pixels']:
                if v['data pixels']['contour'] != {}:
                    xs = v['data pixels']['contour']['xs']
                    ys = v['data pixels']['contour']['ys']
                    #print('contour')
                    #print(xs[:10], ys[:10])
                    for x,y in zip(xs,ys):
                        #print(x,y)
                        cv.circle(imgplot, (int(x), int(y)), csize, color4, -1)
                if 'color bar' in v:
                    #print('ding')
                    imgplot = plot_color_bar(v,img,imgplot)
            elif v['type'] == 'image of the sky':
                # get mods
                if 'sky image params' in v['data']['data params']:
                    ymod = v['data']['data params']['sky image params']['original img size'][1]
                    xmod = v['data']['data params']['sky image params']['original img size'][0]
                    mx = 1 # should this be 1 or 0?
                    #print('have xy mod:', xmod, ymod)
                if v['data pixels']['image'] != {}:
                    xs = v['data pixels']['image']['xsc']
                    ys = v['data pixels']['image']['ysc']
                    #print('len xs, ys:', len(xs), len(ys))
                    if 'x pixel limits' not in v['data from plot']: # no limits
                        for ixy,(x,y) in enumerate(zip(xs,ys)):
                            cv.circle(imgplot, (int(x), int(y)), csize, color3, -1)
                    else:
                        #print('checking limits')
                        for ixy,(x,y) in enumerate(zip(xs,ys)):
                            # check limits
                            # JPN -- not sure about the -1!
                            if (ixy-mx)//xmod > v['data from plot']['x pixel limits'][0] and \
                                (ixy-mx)//xmod < v['data from plot']['x pixel limits'][1] and \
                                    ixy%ymod > v['data from plot']['y pixel limits'][0] and \
                                        ixy%ymod < v['data from plot']['y pixel limits'][1]:
                                cv.circle(imgplot, (int(x), int(y)), csize, color3, -1)                                
                if v['data pixels']['contour'] != {}:
                    xs = v['data pixels']['contour']['xs']
                    ys = v['data pixels']['contour']['ys']
                    if 'x pixel limits' not in v['data from plot']: # no limits
                        for x,y in zip(xs,ys):
                            cv.circle(imgplot, (int(x), int(y)), csize, color4, -1)
                    else:
                        for ixy,(x,y) in enumerate(zip(xs,ys)):
                            # check limits
                            # JPN -- not sure about the -1!
                            if (ixy-mx)//xmod > v['data from plot']['x pixel limits'][0] and \
                                (ixy-mx)//xmod < v['data from plot']['x pixel limits'][1] and \
                                    ixy%ymod > v['data from plot']['y pixel limits'][0] and \
                                        ixy%ymod < v['data from plot']['y pixel limits'][1]:
                                cv.circle(imgplot, (int(x), int(y)), csize, color4, -1)    
                if 'color bar' in v:
                    imgplot = plot_color_bar(v,img,imgplot)                
            else:
                print('no idea how to deal with this plot type! Plot type = ', v['type'])
                import sys; sys.exit()
    
            # these are things for every plot
            for d in v['xticks']: # draw x-ticks
                xmin,ymin,xmax,ymax = int(d['xmin']),int(img.shape[0]-d['ymin']),int(d['xmax']),int(img.shape[0]-d['ymax'])
                if d['tx']>=v['square']['xmin'] and d['tx']<=v['square']['xmax']:
                    cv.rectangle(imgplot, (xmin,ymin), (xmax,ymax), color1, 3)
                # also ticks
                if d['tx']>=v['square']['xmin'] and d['tx']<=v['square']['xmax']:
                    cv.circle(imgplot, (int(d['tx']), int(img.shape[0]-d['ty'])), 10, color1, -1)
            for d in v['yticks']: # draw y-ticks
                xmin,ymin,xmax,ymax = int(d['xmin']),int(img.shape[0]-d['ymin']),int(d['xmax']),int(img.shape[0]-d['ymax'])
                if d['ty']>=v['square']['ymin'] and d['ty']<=v['square']['ymax']:
                    cv.rectangle(imgplot, (xmin,ymin), (xmax,ymax), color1, 3)
                if d['ty']>=v['square']['ymin'] and d['ty']<=v['square']['ymax']:
                    cv.circle(imgplot, (int(d['tx']), int(img.shape[0]-d['ty'])), 10, color1, -1)
            # any offset labels?
            for t in ['x','y']:
                if t + '-offset text' in v:
                    xmin = int(round(v[t + '-offset text']['xmin']))
                    ymin = img.shape[0]-int(round(v[t + '-offset text']['ymin']))
                    xmax = int(round(v[t + '-offset text']['xmax']))
                    ymax = img.shape[0]-int(round(v[t + '-offset text']['ymax']))
                    cv.rectangle(imgplot, (xmin,ymin), (xmax,ymax), color1, 3)

            if v['type'] == 'line':
                if 'x error bars' in v['data pixels']:
                    for il,l in enumerate(v['data pixels']['x error bars']):
                        for ie,(xmin,ymin,xmax,ymax) in enumerate(l):
                            xmin,ymin = int(round(xmin)),int(round(img.shape[0]-ymin))
                            xmax,ymax = int(round(xmax)),int(round(img.shape[0]-ymax))
                            # only take parts within square
                            xmin = max(xmin,int(round(v['square']['xmin'])))
                            xmax = min(xmax,int(round(v['square']['xmax'])))
                            ymin = max(ymin,int(round(img.shape[0]-v['square']['ymax'])))
                            ymax = min(ymax,int(round(img.shape[0]-v['square']['ymin'])))
                            cv.line(imgplot, (xmin,ymin),(xmax,ymax), color1, 2)
                if 'y error bars' in v['data pixels']:
                    for l in v['data pixels']['y error bars']:
                        for xmin,ymin,xmax,ymax in l:
                            xmin,ymin = int(round(xmin)),int(round(img.shape[0]-ymin))
                            xmax,ymax = int(round(xmax)),int(round(img.shape[0]-ymax))
                            # only take parts within square
                            xmin = max(xmin,int(round(v['square']['xmin'])))
                            xmax = min(xmax,int(round(v['square']['xmax'])))
                            ymin = max(ymin,int(round(img.shape[0]-v['square']['ymax'])))
                            ymax = min(ymax,int(round(img.shape[0]-v['square']['ymin'])))
                            cv.line(imgplot, (xmin,ymin),(xmax,ymax), color1, 2)
            elif v['type'] == 'scatter' or v['type'] == 'histogram':
                if 'x error bars' in v['data pixels']:
                    for xmin,ymin,xmax,ymax in v['data pixels']['x error bars']:
                        xmin,ymin = int(round(xmin)),int(round(img.shape[0]-ymin))
                        xmax,ymax = int(round(xmax)),int(round(img.shape[0]-ymax))
                        # only take parts within square
                        xmin = max(xmin,int(round(v['square']['xmin'])))
                        xmax = min(xmax,int(round(v['square']['xmax'])))
                        ymin = max(ymin,int(round(img.shape[0]-v['square']['ymax'])))
                        ymax = min(ymax,int(round(img.shape[0]-v['square']['ymin'])))
                        cv.line(imgplot, (xmin,ymin),(xmax,ymax), color1, 2)
                if 'y error bars' in v['data pixels']:
                    for xmin,ymin,xmax,ymax in v['data pixels']['y error bars']:
                        #print(ymin,ymax)
                        xmin,ymin = int(round(xmin)),int(round(img.shape[0]-ymin))
                        xmax,ymax = int(round(xmax)),int(round(img.shape[0]-ymax))
                        xmin = max(xmin,int(round(v['square']['xmin'])))
                        xmax = min(xmax,int(round(v['square']['xmax'])))
                        ymin = max(ymin,int(round(img.shape[0]-v['square']['ymax'])))
                        ymax = min(ymax,int(round(img.shape[0]-v['square']['ymin'])))
                        cv.line(imgplot, (xmin,ymin),(xmax,ymax), color1, 2)

    return imgplot