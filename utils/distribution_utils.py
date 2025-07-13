import numpy as np

######### "RAW" DISTRIBUTIONS ##########

def get_random(xmin,xmax,ymin=0,ymax=0, zmin=0,zmax=1,hmin=0,hmax=1,
               ndims=1,
               npoints=10,grid=False,
              function=np.random.uniform): #, rng=np.random):
    """
    Generic random distribution
    ndims : 1-3, how many dimensions for the data
    xmin/xmax : ranges for dimension 1
    ymin/ymax : ranges for dimension 2
    zmin/zmax : ranges for dimension 3
    hmin/hmax : ranges for dimension 3, grid=True (3d "heatmap")
    npoints : number of random points (can be a tuple for different values)
    function : how to generate the random numbers
    grid : if ndim>1 and grid is set to True instead of individual points, 
           you will get a data on a uniform grid
    """
    if ndims == 1:
        #print(xmin)
        #print(xmax)
        #print(npoints)
        return function(low=xmin, high=xmax, size=npoints)
    elif ndims == 2:
        if not grid:
            return function(low=np.array([xmin,ymin]),
                            high=np.array([xmax,ymax]), size=[npoints,2])
        else:
            x = np.linspace(xmin,xmax,npoints[0])
            y = np.linspace(ymin,ymax,npoints[1])
            #grid = function(low=zmin,high=zmax,size=(npoints,npoints))
            grid = function(low=zmin,high=zmax,size=npoints) # npoints is a tuple (nx,ny)
            return {'x':x, 'y':y, 'z':grid.T} # .T for right shape
    elif ndims == 3:
        if not grid:
            return function(low=np.array([xmin,ymin,zmin]),
                            high=np.array([xmax,ymax,zmax]), size=[npoints,3])
        else:
            x = np.linspace(xmin,xmax,npoints[0])
            y = np.linspace(ymin,ymax,npoints[1])
            z = np.linspace(zmin,zmax,npoints[2])
            #grid = function(low=hmin,high=hmax,size=(npoints,npoints,npoints))
            grid = function(low=hmin,high=hmax,size=npoints) # tuple (nx,ny,nz)
            return {'x':x, 'y':y, 'z':z, 'h':grid.T} # .T for right shape

    else:
        print('ndims=', ndims, 'not supported for "get_random" function!')
        import sys; sys.exit()
        


##### SPECIFIC DISTRIBUTIONS #######
def get_random_data(plot_type,xmin,xmax,ymin=0,ymax=1,zmin=0,zmax=0,
                    prob_same_x=0.5, nlines=1, npoints=1,
                   cmin=0, cmax=1, rng=np.random):
    """
    npoints : can be tuple if multi dimension
    """
    # do we have all the same x for all points?
    if plot_type == 'line':
        p = rng.uniform(0,1)
        xs = []
        if p <= prob_same_x: # same x for all y
            x = get_random(xmin,xmax,ndims=1,npoints=npoints, 
                           function=rng.uniform)#, rng=rng)
            x = np.sort(x)
            # repeat for all
            for i in range(nlines):
                xs.append(x)
        else: # different
            for i in range(nlines):
                x = get_random(xmin,xmax,ndims=1,npoints=npoints, function=rng.uniform)#, rng=rng)
                x = np.sort(x)
                xs.append(x)
            
        ys = []
        for i in range(nlines):
            y = get_random(ymin[i],ymax[i],ndims=1,npoints=npoints, function=rng.uniform)#, rng=rng)
            ys.append(y)
        return xs,ys
    elif plot_type == 'scatter':
        xs = get_random(xmin,xmax,ndims=1,npoints=npoints, function=rng.uniform)#, rng=rng)
        ys = get_random(ymin,ymax,ndims=1,npoints=npoints, function=rng.uniform)#, rng=rng)
        colors = get_random(cmin,cmax,ndims=1,npoints=npoints, function=rng.uniform)#, rng=rng)
        return xs,ys,colors
    elif plot_type == 'histogram':
        x = get_random(xmin,xmax,ndims=1,npoints=npoints, function=rng.uniform)#, rng=rng)
        return x
    elif plot_type == 'contour':
        data = get_random(xmin,xmax,ymin=ymin,ymax=ymax, ndims=2, grid=True, 
                         npoints=npoints, zmin=zmin,zmax=zmax, function=rng.uniform)#, rng=rng) # {'x':x, 'y':y, 'z':grid}
        xs = data['x']; ys = data['y']
        colors = data['z']
        return xs,ys,colors


########################################################################
############################# LINEAR ############################
#######################################################################


def get_linear(x,y=[],z=[],h=[],
               ndims=1,
                    a1=(-1,1), a2=(-1,1), a3=(-1,1), a4=(-1,1), # mx + a
                    m1=(-1,1), m2=(-1,1), m3=(-1,1), m4=(-1,1), # mx + a
                    noise1=(0,1), noise2=None,noise3=None, noise4=None,
               npoints=10,grid=False,
              function=np.random.uniform, rng=np.random):
    """
    a1-a4 : the "a's" in mx + a, will be randomly selected from range
    m1-m4 : the "m's" in mx + a, will be randomly selected from range
    noise1-noise4 : noise as a % of calculated "y", will be randomly selected from range
    """
    if ndims == 1:
        a = function(a1[0],a1[1])
        m = function(m1[0],m1[1])
        y = m*x + a
        # now add noise --> I think maybe multiply?
        noise_level = function(noise1[0],noise1[1])
        noise = rng.normal(0,1,npoints)*noise_level
        y = y*(1+noise)
        data = {'m':m, 'a':a, 'noise level':noise_level}
        return y, data
    if ndims == 2:
        if not grid:
            a11 = function(a1[0],a1[1])
            m11 = function(m1[0],m1[1])
            a22 = function(a2[0],a2[1])
            m22 = function(m2[0],m2[1])
            #noise_level1 = function(noise1[0],noise1[1])
#            if len(x.shape)>1 and len(y.shape)>1: # have 2d x/y
            noise_level1 = function(noise1[0],noise1[1])
            #print('noise level shape', noise_level1)
#            elif len(x) == len(y):
#                noise_level1 = 
                
            noise1 = rng.normal(0,1,npoints[0])*noise_level1
            #print('noise1 shape', noise1.shape)
            if noise2 is not None: # different noise in different directions
                noise_level2 = function(noise2[0],noise2[1])
                noise2 = rng.normal(0,1,npoints[1])*noise_level2
                z = (m11*x + a11)*(1+noise1) + (m22*y + a22)*(1+noise2)
            else:
                if len(x) == len(y):
                    noise1 = rng.normal(0,1,npoints[0])*noise_level1
                else:
                    noise1 = rng.normal(0,1,npoints)*noise_level1
                    
                #print('noise1 shape here', noise1.shape)
                z = (m11*x + a11 + m22*y + a22)*(1+noise1)

            #print('z shape', z.shape)
                
            data = {'m1':m11, 'a1':a11, 
                    'm2':m22, 'a2':a22, 
                   'noise level 1':noise_level1}
            if noise2 is not None: 
                data['noise level 2'] = noise_level2
            return z, data
        else:
            a11 = function(a1[0],a1[1])
            m11 = function(m1[0],m1[1])
            a22 = function(a2[0],a2[1])
            m22 = function(m2[0],m2[1])
            noise_level1 = function(noise1[0],noise1[1],len(x))
            grid = np.zeros([len(x),len(y)])
            # repeat x/y
            xr = np.repeat(x[:,np.newaxis], len(y), axis=1).T
            yr = np.repeat(y[:,np.newaxis], len(x), axis=1)
            if noise2 is not None:
                noise_level2 = function(noise2[0],noise2[1],len(y))
                nxr = np.repeat(noise_level1[:,np.newaxis], len(y), axis=1).T
                nyr = np.repeat(noise_level2[:,np.newaxis], len(x), axis=1)
                grid = (m11*xr + a11)*(1+nxr) + (m22*yr + a22)*(1+nyr)
            else:
                noise_level1 = function(noise1[0],noise1[1],npoints)
                #print('shape1:', noise_level1.shape)
                noise1 = rng.normal(0,1,npoints)*noise_level1
                grid = (m11*xr + a11 + m22*yr + a22)*(1+noise1.T)
                
            data = {'m1':m11, 'a1':a11, 
                    'm2':m22, 'a2':a22, 
                   'noise level 1':noise_level1}
            if noise2 is not None:
                   data['noise level 2'] = noise_level2
            zout = {'x':x, 'y':y, 'z':grid}
            return zout, data



def get_linear_data(plot_type, dist_params, 
                    xmin,xmax,ymin=0,ymax=1,zmin=0,zmax=0,
                    prob_same_x=0.5, nlines=1, npoints=1,
                   cmin=0, cmax=1, 
                   function=np.random.uniform, rng=np.random):

    """
    npoints : can be tuple if multi dimension
    """
    #print('in get_linear_data:', dist_params)
    # do we have all the same x for all points?
    if plot_type == 'line':
        p = rng.uniform(0,1)
        xs = []
        if p <= prob_same_x: # same x for all y
            x = function(low=xmin, high=xmax, size=npoints) # x-values
            x = np.sort(x)
            # repeat for all
            for i in range(nlines):
                xs.append(x)
        else: # different
            for i in range(nlines):
                x = function(low=xmin, high=xmax, size=npoints) # x-values
                x = np.sort(x)
                xs.append(x)
            
        ys = []
        data_line = {}
        for i in range(nlines):
            #y = get_random(ymin[i],ymax[i],ndims=1,npoints=npoints)
            y, data = get_linear(xs[i],ndims=1,npoints=npoints,
                          a1=dist_params['intersect'],
                          m1=dist_params['slope'],
                          noise1=dist_params['noise'], function=rng.uniform)
            ##y = data['y']
            ys.append(y)
            data_line['line'+str(i)] = data.copy()
        return xs,ys, data_line
    elif plot_type == 'scatter':
        # x/y
        xs = get_random(xmin,xmax,ndims=1,npoints=npoints, function=rng.uniform)#, rng=rng) # this will be random still
        ys, data_points = get_linear(xs,ndims=1,npoints=npoints,
                      a1=dist_params['intersect'],
                      m1=dist_params['slope'],
                      noise1=dist_params['noise'], function=rng.uniform)    
        # colors linear?
        if rng.uniform(0,1) <= dist_params['color noise prob']:
            colors, data_color = get_linear(xs,ys,ndims=2, 
                               npoints=(npoints,npoints),
                      a1=dist_params['intersect'],
                      m1=dist_params['slope'],
                      noise1=dist_params['noise'], rng=rng, function=rng.uniform)#,  
                      # a2=dist_params['intersect'],
                      # m2=dist_params['slope'],
                      # noise2=dist_params['noise'])  
            # colors is now 2D at each combo of x/y
            data_color['type'] = 'linear'
        else:
            colors = get_random(cmin,cmax,ndims=1,npoints=npoints, function=rng.uniform)#, rng=rng)
            data_color = {'type':'random'}

        #ys = get_random(ymin,ymax,ndims=1,npoints=npoints)
        #colors = get_random(cmin,cmax,ndims=1,npoints=npoints)
        # is color linear?
        data = {'points':data_points, 'colors':data_color}
        return xs,ys,colors, data
    elif plot_type == 'histogram': # I feel like this doesn't make too 
                                   # much sense for histograms, but lets just do it
        x = get_random(xmin,xmax,ndims=1,npoints=npoints, function=rng.uniform)#, rng=rng) 
        y, data = get_linear(x,ndims=1,npoints=npoints,
                      a1=dist_params['intersect'],
                      m1=dist_params['slope'],
                      noise1=dist_params['noise'], rng=rng, function=rng.uniform)
        return y, data
    elif plot_type == 'contour':
        xs = get_random(xmin,xmax,ndims=1,npoints=npoints[0], function=rng.uniform)#, rng=rng) # this will be random still
        xs = np.sort(xs)
        ys = get_random(xmin,xmax,ndims=1,npoints=npoints[1], function=rng.uniform)#, rng=rng) # this will be random still
        ys = np.sort(ys)
        data, data_params = get_linear(xs,ys,ndims=2, 
                               npoints=npoints,
                      a1=dist_params['intersect'],
                      m1=dist_params['slope'],
                      noise1=dist_params['noise'],  
                      # a2=dist_params['intersect'],
                      # m2=dist_params['slope'],
                      # noise2=dist_params['noise'],
                                      grid=True, rng=rng, function=rng.uniform)

        xs = data['x']; ys = data['y']
        colors = data['z']
        return xs,ys,colors,data_params



########################################################################
############################# GMM (in progress!) ############################
#######################################################################


# nclusters = 3
# cluster_std = ([1,0.05],
#                [1,1],
#                [2,2])
# nsamples = 500

from sklearn.datasets import make_blobs
from scipy.stats import binned_statistic_2d

def get_gmm(xmin,xmax,ymin=0,ymax=1,zmin=0,zmax=1,
            cmin=0,cmax=1, ndims=1, 
            nclusters = {'min':1,'max':10}, 
            cluster_std = {'min':0.01, 'max':1.0},
            nsamples = {'min':10,'max':1000},
            noise = {'min':0, 'max':1.0},
            grid=False,
           function=np.random.uniform, 
           small_data_params = True, 
           down_sample_max=None, fit_to_bounds = True,
            min_points=5, iloop_max = 8, max_sample_size = 1000000, 
            seed = None, verbose=False, rng=np.random):

    """
    small_data_params : won't save X array and y_true for contour plots (big stuffs)
    seed : set seed to a number to keep things consitent for debugging
    """

    #seed = 42
    #rng.seed(seed=seed)

    if type(nsamples) == type({}): # will pull randomly
        if nsamples['min'] == nsamples['max']:
            nsamples1 = nsamples['min']
        elif rng == np.random:
            nsamples1 = rng.randint(nsamples['min'],nsamples['max'])
        else:
            nsamples1 = rng.integers(nsamples['min'],nsamples['max'])
        nsamples_min = min(nsamples['min'],min_points)
    elif type(nsamples) == type([]) or type(nsamples) == type(()):
        nsamples1 = nsamples[0]
        nsamples_min=min(nsamples[0],min_points)
    else:
        nsamples1 = nsamples # if fixed
        nsamples_min = min(nsamples,min_points)
    # number of clusters
    if nclusters['min'] != nclusters['max']:
        if rng == np.random:
            nclusters1 = rng.randint(nclusters['min'],nclusters['max'])
        else:
            nclusters1 = rng.integers(nclusters['min'],nclusters['max'])
    else: # same
        nclusters1 = nclusters['min']
    if ndims > 1:
        cluster_std1 = rng.uniform(cluster_std['min'],cluster_std['max'], (nclusters1,ndims))
        # power
        #print('power:', cluster_std1)
        cluster_std1 = np.power(10,cluster_std1)
        #print('after power:', cluster_std1)
        #cluster_std1 *= (xmax-xmin)
    else:
        #print('ding ding!')
        cluster_std1 = rng.uniform(cluster_std['min'],cluster_std['max'], nclusters1)
        cluster_std1 = np.power(10,cluster_std1)
        cluster_std1 *= (xmax-xmin)

    #print(cluster_std1)
        
    data_params = {'nsamples':nsamples1, 'nclusters':nclusters1}#,
                  #'cluster_std':cluster_std1}
    # moving for consistancy?
    centers = np.zeros((nclusters1,ndims))
    noise_level = function(noise['min'],noise['max'])

    # n_features = n_dim
    # centers : int or array-like of shape (n_centers, n_features), default=None
    #    The number of centers to generate, or the fixed center locations.
    # center_box : tuple of float (min, max), default=(-10.0, 10.0)
    if ndims == 1: # along a line 
        #print('xmin,xmax', xmin,xmax)
        #print('nclusters1', nclusters1)
        #centers_x = rng.uniform(xmin,xmax,(nclusters1,2))
        centers_x = rng.uniform(xmin,xmax,nclusters1).reshape(-1, 1)
        # for reproducable random state
        random_state = rng.bit_generator.state['state']['state']%(2**32 - 1)
        #print('random state 1:', random_state)
        X, y_true,centers_ret = make_blobs(n_samples=nsamples1, n_features=1,
                               cluster_std=cluster_std1, centers=centers_x,
                              center_box=(xmin,xmax),
                                      return_centers=True, 
                                      random_state=random_state)
        #**HERE** maybe we need an upsampling factor again
        # collapse
        #X = np.sum(X, axis=1)
        # labeling starts at 0
        y_true += 1
        
        #noise1 = rng.normal(0,1,nsamples1)*noise_level*(xmax-xmin)
        #print('distribution utils - noise_level, nsamples, ndims', noise_level, nsamples,ndims)
        noise1 = rng.normal(loc=0,
                                scale=noise_level/2, # the 1/2 is just a sort of random scalier
                                size=(nsamples1,ndims))#*noise_level  
        shifts = noise1*(xmax-xmin)
        Xout = X.flatten() + shifts.flatten()  
        # print('Xflatten =', X.shape)
        # print('shifts', shifts.shape) 
        # print('Xout:', Xout.shape)
        # print('xmin,xmax', xmin,xmax)
        #noise1[noise1<0] = 0.0 # allow for negatives
        # multiply
        #Xout = X.flatten()*(1+noise1.flatten())
        #shifts = noise1*(ma-mi)
        #Xout = X.flatten() + noise1
        # drop outside
        if fit_to_bounds:
            mask = (Xout > xmax) | (Xout < xmin)
            #print('mask', mask.shape)
            #print('x,y before:', Xout.shape, y_true.shape)
            y_true_out = y_true.copy()[~mask]
            Xout = Xout[~mask]
            #print('x,y after:', Xout.shape, y_true.shape)

        if not small_data_params:
            data_params['X'] = X
            data_params['Xout'] = Xout
            data_params['labels'] = y_true
        data_params['centers'] = centers_x
        data_params['cluster_std'] = cluster_std1
        data_params['noise level'] = noise_level
        
        return Xout, data_params 
    elif ndims > 1: # multi-d
        if ndims > 1: # 2
            #mi = np.min([xmin,ymin])
            mi = np.array([xmin,ymin])
            #ma = np.max([xmax,ymax])
            ma = np.array([xmax,ymax])
        if ndims > 2:
            #mi = np.min([mi,zmin])
            #ma = np.max([ma,zmax])
            #pass
            mi = np.array([xmin,ymin,zmin])
            ma = np.array([xmax,ymax,zmax])
        if ndims > 3:
            print('ndims > 3 not supported!')
            import sys; sys.exit()

        #print('mi,ma:', mi,ma)
        #print('xmin,xmax,ymin,ymax:',xmin,xmax, ymin,ymax)
        y_true_out = []
        isamples_mul = 0
        maxReached = True
        #print('HERE gmm_data 3')
        #print(nsamples_min, iloop_max, max_sample_size, nsamples1)
        if nsamples1 > max_sample_size:
            nsamples1 = max_sample_size
        # y_true_out 
        while (len(y_true_out) < nsamples_min) or ((isamples_mul<=iloop_max) and (nsamples1 <= max_sample_size)): # loop until we get something
            if isamples_mul > 1: # we are on a new loop
                if verbose: print('    on loop:', isamples_mul, nsamples1, len(y_true_out))
            isamples_mul += 1
            nsamples1 *= isamples_mul
            centers_x = rng.uniform(xmin,xmax,nclusters1)
            centers_y = rng.uniform(ymin,ymax,nclusters1)
            #centers = np.zeros((nclusters1,ndims))
            centers[:,0] = centers_x
            centers[:,1] = centers_y
            if ndims > 2:
                centers_z = rng.uniform(zmin,zmax,nclusters1)
                centers[:,2] = centers_z
            # print('centers',centers)
            # print("ma,mi", ma,mi)
            # print('std',cluster_std1*(ma-mi))
            random_state = rng.bit_generator.state['state']['state']%(2**32 - 1)
            #print('random state 2:', random_state)
            # print('nsamles1:', nsamples1)
            # print('nfeatures', ndims)
            # print('cluster std min/max', cluster_std1.min(), cluster_std1.max())
            # print('ma,mi', ma,mi)
            # print('centers min/max:', centers.min(), centers.max())
            X, y_true,centers_ret = make_blobs(n_samples=nsamples1, n_features=ndims,
                                   cluster_std=cluster_std1*(ma-mi), centers=centers,
                                  center_box=(mi,ma),
                                          return_centers=True, 
                                          random_state=random_state)
            #print('xmin/max', X.min(), X.max())
            
            noise_level = function(noise['min'],noise['max'])
            #print('noise_level:', noise_level)
            #noise1 = rng.normal(0,1,(nsamples1,ndims))*noise_level
            noise1 = rng.normal(loc=0,
                          scale=noise_level/2, # the 1/2 is just a scaler
                          size=(nsamples1,ndims))
            # starts at 0
            y_true += 1
                                        
            # multiply
            #Xout = X*(1+noise1)
            #**HERE** -- small shift?
            shifts = noise1*(ma-mi)
            Xout = X + shifts
            #print('xmin,ymin,xmax,ymax:', xmin,ymin,xmax,ymax)
            # drop outside
            #print('X:', X)
            #print('Xout:', Xout)
            mask = (Xout[:,0] > xmax) | (Xout[:,0] < xmin) | (Xout[:,1] > ymax) | (Xout[:,1] < ymin)
            Xout = Xout[~mask]
            y_true_out = y_true.copy()[~mask]
            mask = (X[:,0] > xmax) | (X[:,0] < xmin) | (X[:,1] > ymax) | (X[:,1] < ymin)
            #print('X before:', X.shape)
            X = X[~mask]
            #print('X after:', X.shape)
            y_true = y_true[~mask]
            if len(y_true_out) >= nsamples_min: maxReached = False
            # print('y_true, 1:', y_true)
            # print('y_true_out:', y_true_out)

        #print('HERE gmm_data 4')
        if not maxReached: # we have maxed out! drop condition
            if verbose: print('    maxed out!')
            centers_x = rng.uniform(xmin,xmax,nclusters1)
            centers_y = rng.uniform(ymin,ymax,nclusters1)
            centers = np.zeros((nclusters1,ndims))
            centers[:,0] = centers_x
            centers[:,1] = centers_y
            if ndims > 2:
                centers_z = rng.uniform(zmin,zmax,nclusters1)
                centers[:,2] = centers_z
            #print('centers',centers)
            #print("ma,mi", ma,mi)
            #print('std',cluster_std1*(ma-mi))
            random_state = rng.bit_generator.state['state']['state']%(2**32 - 1)
            #print('random state 3:', random_state)
            #print('function:', function)
            #print('rng:', rng)
            X, y_true,centers_ret = make_blobs(n_samples=nsamples1, n_features=ndims,
                                   cluster_std=cluster_std1*(ma-mi), centers=centers,
                                  center_box=(mi,ma),
                                          return_centers=True, 
                                          random_state=random_state)
            
            noise_level = function(noise['min'],noise['max'])
            noise1 = rng.normal(0,1,(nsamples1,ndims))*noise_level
            # starts at 0
            y_true += 1
                                        
            # multiply
            Xout = X*(1+noise1)
            y_true_out = y_true.copy()#[~mask]


        #print('HERE gmm_data 5')
        if not small_data_params:
            data_params['X'] = X
            #data_params['Xout'] = Xout
            #data_params['y_true_out'] = y_true_out
            data_params['labels'] = y_true
        data_params['centers'] = centers
        data_params['cluster_std'] = cluster_std1
        data_params['noise level'] = noise_level
        #print('HERE gmm_data 6')

        if not grid:
            # colors
            #print(nclusters1)
            #print(y_true_out)
            
            if nclusters1 > 1 and np.max(y_true_out) != np.min(y_true_out):
                colors = (y_true_out-np.min(y_true_out))/(np.max(y_true_out)-np.min(y_true_out))*(cmax-cmin)+cmin
            else:
                colors = (y_true_out)*(cmax-cmin)+cmin # ytrue = 1 now
            # noise here as well
            noise2 = rng.normal(0,1,len(y_true_out))*noise_level
            colors = colors*(1+noise2)

            # downsample if needed
            if down_sample_max is not None:
                if len(colors) > down_sample_max:
                    ints = rng.choice(np.arange(0,len(colors)),
                                            size=down_sample_max,
                                            replace=False)
                    Xout = Xout[ints,:]
                    colors = colors[ints]
            return Xout, colors, data_params
        else: # we have more work to do (contours)
            #print('HERE gmm_data 7')
            # just the shape
            #y_true[y_true>0] = 1
            y_true[:] = 1
            #print('HERE gmm_data 7.1')
            #y_true = y_true[~mask]
            nbinsx,nbinsy = nsamples[1],nsamples[2]
            
            binx = np.linspace(xmin,xmax, nbinsx)
            biny = np.linspace(ymin,ymax, nbinsy)

            #print("HERE gmm_data 7.1.1")
            #print('y_true:', y_true)
                
            ret = binned_statistic_2d(X[:,1], X[:,0], y_true, 
                                    'sum', bins=[biny,binx], 
                expand_binnumbers=True)
            #print("HERE gmm_data 7.2")
            
            colors = ret.statistic
            # renorm
            if np.max(colors) != np.min(colors):
                div1 = np.max(colors)-np.min(colors)
            else:
                div1 = 1
            colors = (colors-np.min(colors))/div1*(cmax-cmin)+cmin
            # noise here as well
            noise2 = rng.normal(0,1,colors.shape)*noise_level
            colors = colors*(1+noise2)

            xs = (binx[:-1] + binx[1:]) / 2
            ys = (biny[:-1] + biny[1:]) / 2
            #print("HERE gmm_data 8")
            # print('xs:',xs)
            # print('ys:',ys)
            # print('colors:', colors)
            # print('data_params:', data_params)
            #print('shape(xs):', xs.shape, 'shape(colors):', colors.shape)

            return xs,ys, colors, data_params


def get_gmm_data(plot_type, dist_params,
                 xmin,xmax,ymin=0,ymax=1,zmin=0,zmax=0,
                    prob_same_x=0.5, nlines=1, npoints=1,
                   cmin=0, cmax=1, 
                   function=np.random.uniform, small_data_params = True, 
                   seed = None, verbose=False, rng=np.random):
    """
    npoints : can be tuple if multi dimension
    seed : set seed to number for randomization
    """

    #seed = 42
    #rng.seed(seed)
        
    if plot_type == 'line':
        p = rng.uniform(0,1)
        xs = []
        isSameX = False
        if p <= prob_same_x: # same x for all y
            x = function(low=xmin, high=xmax, size=npoints) # x-values
            x = np.sort(x)
            # repeat for all
            for i in range(nlines):
                xs.append(x)
            isSameX = True
        else: # different
            for i in range(nlines):
                x = function(low=xmin, high=xmax, size=npoints) # x-values
                x = np.sort(x)
                xs.append(x)

        # histograms actually but as line plot?
        isHisto = False
        if rng.uniform(0,1) <= dist_params['histogram as line']['prob']:
            isHisto = True
        ys = []
        data_params = {}
        for i in range(nlines):
            if not isHisto:
                y, data_params1 = get_gmm(ymin[i],ymax[i],
                                         ndims=1,
                                         #npoints=npoints,
                              nclusters=dist_params['nclusters'],
                              nsamples=len(xs[i]), # keep same # of x values
                              cluster_std=dist_params['cluster std'],
                                        noise=dist_params['noise'],
                                        function=function,
                                         small_data_params = small_data_params,
                                         fit_to_bounds = not isSameX, 
                                         verbose=verbose, rng=rng)
                #print('isSameX', isSameX)
                if not isSameX and len(y) != len(xs[i]):
                    #print('hey')
                    inds = rng.choice(np.arange(0,len(xs[i])),len(y),replace=False)
                    inds = np.sort(inds)
                    xs[i] = xs[i][inds]
            else:
                ypoints, data_params1 = get_gmm(np.min(xs[i]),np.max(xs[i]),
                                             ndims=1,
                                  nclusters=dist_params['nclusters'],
                                  nsamples=len(xs[i])*dist_params['histogram as line']['factor'],
                                  cluster_std=dist_params['cluster std'],
                                            noise=dist_params['noise'],
                                            function=function,
                                         small_data_params = small_data_params,
                                         verbose=verbose, rng=rng)
                # repeat from centers to edges
                #bins = xs[i].copy()
                #rol
                y,bin_edges = np.histogram(ypoints, bins=xs[i]) 
                xs[i] = (bin_edges[:-1] + bin_edges[1:]) / 2 # update bins
                # rescale
                y = y.astype('float')
                #print('ymin,ymax, ymin[i],ymax[i]', np.min(y), np.max(y), ymax[i], ymin[i])
                y = (y-np.min(y))/(np.max(y)-np.min(y))*(float(ymax[i])-float(ymin[i]))+ymin[i]
                #print((float(ymax[i])-float(ymin[i]))+ymin[i])

            ys.append(y)
            data_params['line'+str(i)] = data_params1.copy()
        return xs,ys, data_params
    elif plot_type == 'scatter':
        # x/y
        npoints = rng.uniform(dist_params['upsample factor log']['min'], 
                                    dist_params['upsample factor log']['max'])
        npoints = int(np.max(npoints+1)*round(np.power(10,npoints)))

        data_params_out = {'points':"","colors":''}
        X, colors_dist, data_params = get_gmm(xmin,xmax,ymin=ymin,ymax=ymax,
                                     ndims=2,
                          nclusters=dist_params['nclusters'],
                          nsamples=npoints,#dist_params['nsamples'], # keep same # of x values
                          cluster_std=dist_params['cluster std'],
                                    noise=dist_params['noise'],
                                    function=function,
                                                 cmin=cmin,
                                                 cmax=cmax,
                                             down_sample_max=dist_params['max points'],
                                         small_data_params = small_data_params,
                                         verbose=verbose, rng=rng)  
        xs = X[:,0]
        ys = X[:,1]
        # colors linear?
        if rng.uniform(0,1) <= dist_params['color noise prob']: 
            colors = colors_dist
            data_params_out['colors'] = {'type':'gmm, by cluster'}
        else: #otherwise, make random
            colors = get_random(cmin,cmax,ndims=1,npoints=len(colors_dist), function=rng.uniform)#, rng=rng)
            data_params_out['colors'] = {'type':'gmm, random'}

        data_params_out['points'] = data_params.copy()

        return xs,ys,colors, data_params_out
    elif plot_type == 'histogram': # I feel like this doesn't make too 
                                   # much sense for histograms, but lets just do it
        y, data_params = get_gmm(xmin,xmax,
                                     ndims=1,
                          nclusters=dist_params['nclusters'],
                          nsamples=dist_params['nsamples'], # keep same # of x values
                          cluster_std=dist_params['cluster std'],
                                    noise=dist_params['noise'],
                                    function=function,
                                         small_data_params = small_data_params,verbose=verbose,
                                         rng=rng)

        return y, data_params
    elif plot_type == 'contour':
        # upsample the number of points per grid
        nx = npoints[0] # silly reformatting stuffs
        ny = npoints[1]
        npoints = rng.uniform(dist_params['upsample factor log']['min'], 
                                    dist_params['upsample factor log']['max'])
        npoints = int(np.max(npoints+1)*round(np.power(10,npoints)))

        xs,ys, colors, data_params = get_gmm(xmin,xmax,ymin=ymin,ymax=ymax,
                                     ndims=2,
                          nclusters=dist_params['nclusters'],
                          nsamples=(npoints,nx,ny), # keep same # of x values
                          cluster_std=dist_params['cluster std'],
                                    noise=dist_params['noise'],
                                    function=function,
                                                 cmin=cmin,
                                                 cmax=cmax, 
                                        grid=True,
                                         small_data_params = small_data_params, 
                                         seed = seed, verbose=verbose, rng=rng)  

        return xs,ys,colors,data_params
    else:
        print('[ERROR]: No such plot type for:', plot_type)
        
        return xs,ys,colors,data_params


##########################################
########## Image of the Sky ##############
###### (right now only for contours!) ####
##########################################
# JPN -- support for scatters too?
# JPN -- what about histograms?

# # xs,ys,colors, data_params = get_sky_image_data(npoints=(nx,ny),
# #                         cmin=cmin, cmax=cmax, rng=rng)

# #from glob import glob
# import pickle
# import time
# #import numpy as np
# #from copy import deepcopy
# #import matplotlib.pyplot as plt
# import operator

# # for parsing
# #from astroquery.simbad import Simbad
# import time
# from astroquery.skyview import SkyView
# import os

# from astropy.wcs import WCS
# from astropy.io import fits
# from astropy.utils.data import get_pkg_data_filename

# import pandas as pd

# # best guesses for how surveys match up with wavelength tags
# surveys_by_wl = {'O':['Optical:SDSS', 'OtherOptical', 'Optical:DSS class=', 'Allbands:GOODS/HDF/CDF'], 
#                  # repeat IR for other far ir/near ir/etc
#                  'I':['IR:IRAS', 'IR:2MASS class=', 'IR:UKIDSS class=', 'IR:WISE class=', 'IR:AKARI class=', 'IR:Planck', 'IR:WMAP&COBE'], 
#                  'F':['IR:IRAS', 'IR:2MASS class=', 'IR:UKIDSS class=', 'IR:WISE class=', 'IR:AKARI class=', 'IR:Planck', 'IR:WMAP&COBE'], 
#                  'M':['IR:IRAS', 'IR:2MASS class=', 'IR:UKIDSS class=', 'IR:WISE class=', 'IR:AKARI class=', 'IR:Planck', 'IR:WMAP&COBE'], 
#                  'N':['IR:IRAS', 'IR:2MASS class=', 'IR:UKIDSS class=', 'IR:WISE class=', 'IR:AKARI class=', 'IR:Planck', 'IR:WMAP&COBE'], 
#                  # out of IR repeats
#                  'X':['HardX-ray', 'X-ray:SwiftBAT',  'SoftX-ray class=', 'ROSATw/sources class=', 'ROSATDiffuse class='],
#                  'G':['GammaRay'],
#                  'R':['Radio:GHz', 'Radio:MHz class=', 'Radio:GLEAM class='],
#                  'U':['UV', 'SwiftUVOT'],
#                  # repeats for "m" and "s"
#                  'm':['Radio:GHz', 'Radio:MHz class=', 'Radio:GLEAM class='],
#                  's':['IR:IRAS', 'IR:2MASS class=', 'IR:UKIDSS class=', 'IR:WISE class=', 'IR:AKARI class=', 'IR:Planck', 'IR:WMAP&COBE'] 
#                  }

# # other guesses
# # 'F' 'G' 'I' 'M' 'N' 'O' 'R' 'U' 'X' 'm' 's'

# # from docs above
# # survey_keys = {'R':['Radio:GHz', 'Radio:MHz', 'Radio:GLEAM'], 
# #                'I':['IR:IRAS', 'IR:2MASS', 'IR:UKIDSS', 'IR:WISE', 'IR:AKARI', 'IR:Planck', 'IR:WMAP&COBE'], 
# #                'V':['Optical:DSS', 'Optical:SDSS'], 
# #                'U':['UV', 'SwiftUVOT'], 
# #                'X':['HardX-ray', 'X-ray:SwiftBAT', 'SoftX-ray class', 'ROSATw/sources','ROSATDiffuse'], 
# #                'G':['GammaRay']
# #                }

# ## 'F' 'G' 'I' 'M' 'N' 'O' 'R' 'U' 'X' 'm' 's'
# # I think "F" means Far IR -- get some IRAS
# # I think "M" means Mid-IR -- IRAS again
# # N I think means near-IR (RAFGL)
# # m -- maybe millimeter (like Radio)
# # s -- TeV? actually might be submillimeter, like Far IR and Microwave bands

# def get_images_survey(surveys_wl, object_id, save_img_dir, pdfname, missing_list,
#                       missing_list_file,
#                       verbose=True, sleep_time = 2, overwrite = False, 
#                       pick_random_survey = True, rng=np.random, 
#                       height=300, width=300):
#     pdfs = []; objs = []; surveys = []; reasons = []
#     for s in surveys_wl:
#         if verbose: print('  Main survey:', s)
#         try:
#             sub_surveys_wl = SkyView.survey_dict[s]
#         except Exception as e:
#             if verbose:
#                 print(str(e))
#                 print('survey tried:', s)
#             return None
#         if pick_random_survey:
#             sub_surveys_wl = [rng.choice(sub_surveys_wl)]
#         # get all images associated with this survey
#         filenames = []
#         for ss in sub_surveys_wl:
#             subdf = missing_list[(missing_list['object']==object_id) & (missing_list['survey']==ss) & (missing_list['pdf']==pdfname) ]
#             if len(subdf) > 0:
#                 if verbose: print('   missing, skipping:', ss)
#                 continue
#             if verbose: print('    sub survey:', ss)
#             try:
#                 img_list = SkyView.get_images(position=object_id, survey = ss, 
#                                               pixels=(width,height))
#             except:
#                 if verbose: print('      no survey for:', ss)
#                 # add to list
#                 img_list = []
#                 pdfs.append(pdfname); objs.append(object_id); surveys.append(ss)
#                 reasons.append('no survey')
#                 continue
#             if len(img_list) == 0:
#                 if verbose:
#                     print('      no images')
#                 pdfs.append(pdfname); objs.append(object_id); surveys.append(ss)
#                 reasons.append('no images')
#             alreadyHave = False
#             for iimg, img in enumerate(img_list):
#                 print('here0')
#                 filename = 'image_' + pdfname + '_OBJ_' + object_id.replace(' ', '_').replace('/','-') + \
#                                 '_SURVEY_' + str(ss).replace(' ','_').replace('/','-') + \
#                                 '_height' + str(height) + '_width' + str(width) + \
#                                     '_NUM_' + str(iimg).zfill(4) + '.fits'
#                 filenames.append(filename)
#                 if os.path.exists(save_img_dir + filename) and not overwrite:
#                     if verbose:
#                         print('already have:', filename)
#                     alreadyHave = True
#                     continue
#                 if verbose: print('      on image number:', iimg)
#                 try:
#                     print('here1')
#                     img.writeto(save_img_dir + filename)
#                     print('here2')
#                 except Exception as e:
#                     pdfs.append(pdfname); objs.append(object_id); surveys.append(ss)
#                     reasons.append('exception')
#                     if verbose:
#                         print('          cannot write image:', filename)
#                         print('          ' + str(e))
#             if not alreadyHave: time.sleep(sleep_time)

#         if len(pdfs) > 0: # update list
#             print('[WARNING]: writing of missing list is not parallizable!!')
#             missing_addon = pd.DataFrame({'object':objs, 
#                                           'survey':surveys, 
#                                           'pdf':pdfs, 'reason':reasons})
#             missing_out = pd.concat([missing_list,missing_addon],ignore_index=True)
#             missing_out.to_csv(missing_list_file, index=False)
    
#         # choose an image
#         if len(filenames) > 0:
#             img_pick = rng.choice(filenames)
#             return img_pick
#         else:
#             return None


# # JPN -- here you want function as rng.uniform when
# def get_sky_image_data(plot_params,
#                    cmin=0, cmax=1, 
#                    verbose=False, rng=np.random, timer_pause=1.0, 
#                    overwrite = False, pick_random_survey = False, 
#                    height=300, width=300):
#     """
#     npoints : can be tuple if multi dimension
#     timer_pause : if pulling from astroquery, how much of a pause to take
#     height/width : the default resolution of the images returned (astroquery default is 300x300)
#     """

#     # get sky images list
#     if verbose:
#         print("[WARNING]: loading of sky images not optimized! (distribution_utils/get_sky_image_data)")
#     output_dir = plot_params['distribution']['sky']['object wavelength table']
#     #print(output_dir)
#     with open(output_dir,'rb') as f:
#         object_wavelengths = pickle.load(f)

#     #print(object_wavelengths) # don't print, too big

#     # missing obj/surveys
#     missing_list_file = plot_params['distribution']['sky']['missing obj/surveys list']
#     if not os.path.exists(missing_list_file): # make new one
#         missing_list = pd.DataFrame({'object':[], 'survey':[], 'pdf':[], 'reason':[]})
#         if verbose:
#             print('writing missing items file:', missing_list_file)
#     else:
#         try:
#             missing_list = pd.read_csv(missing_list_file)
#         except Exception as e:
#             print('[ERROR]: opening missing_list file in distribution_utils/get_sky_image_data')
#             print(str(e))
#             import sys; sys.exit()


#     filename = None
#     #icount = 0; icount_max = 10
#     # get image height/width
#     height = plot_params['distribution']['sky']['image height']
#     width = plot_params['distribution']['sky']['image width']
#     while filename is None:
#         # grab random object
#         # object, wavelength(s), pdf, itable
#         obj = rng.choice(object_wavelengths)
#         # if verbose:
#         #     print('object keys:')
#         #     print(obj)
#         #     print('')

#         # if only wavelength is emtpy, just guess optical
#         if obj['wavelength'] == '':
#             survey_keys = ['Optical:DSS class='] #surveys_by_wl['O']
#             if verbose:
#                 print('No wavelength, picking optical ( wavelength empty )')
#         else:
#             survey_keys = surveys_by_wl[obj['wavelength']]

#         # pick random survey
#         if not pick_random_survey:
#             survey = [rng.choice(survey_keys)]
#             if verbose: print('picking random survey:', survey)
#         else:
#             survey = survey_keys

#         if verbose:
#             print('object, survey, pdf:', obj['object'], survey, obj['pdf'])

#         # try all surveys
#         isurvey = 0
#         n_surveys = len(survey_keys)
#         while filename is None and isurvey < n_surveys:
#             if verbose: print('on survey', survey, '(', isurvey+1, 'of', n_surveys, ')')
#             filename = get_images_survey(survey, obj['object'], 
#                             plot_params['distribution']['sky']['query images dir'], 
#                             obj['pdf'], missing_list, missing_list_file,
#                         verbose=verbose, sleep_time = timer_pause, overwrite = overwrite, 
#                         pick_random_survey = pick_random_survey, rng=rng, 
#                         height=height, width=width)
#             # try different survey if no file
#             if filename is None:
#                 survey = [survey_keys[isurvey]]
#                 isurvey += 1

#         # try to open
#         if filename is not None:
#             try:
#                 if verbose:
#                     print('will try to load fits file:', plot_params['distribution']['sky']['query images dir']+filename)
#                 filename2 = plot_params['distribution']['sky']['query images dir']+filename
#                 # add a with?
#                 with open(filename2,'rb') as ff:
#                     hdu = fits.open(ff, output_verify='fix')[0]
#                     wcs = WCS(hdu.header)
#                     img_data = hdu.data
#             except Exception as e:
#                 if verbose:
#                     print('Issue opening fits file')
#                     print(str(e))
#                 filename = None

#         # also check if only NaN's
#         if filename is not None:
#             if len(img_data[~np.isnan(img_data)]) == 0:
#                 if verbose:
#                     print('[WARNING]: img_data is all NaNs for file:', filename)
#                 filename = None


#     # if verbose:    
#     #     print('')
#     #     #print('filename = ', filename)
#     #     print('will try to load fits file:', plot_params['distribution']['sky']['query images dir']+filename)

#     # # I DON'T KNOW WHY THIS LINE WON'T WORK!
#     # #filename2 = get_pkg_data_filename(str(plot_params['distribution']['sky']['query images dir'])+str(filename))
#     # filename2 = plot_params['distribution']['sky']['query images dir']+filename

#     # hdu = fits.open(filename2)[0]
#     # wcs = WCS(hdu.header)
#     # img_data = hdu.data

#     data_params = {}
#     data_params['WCS'] = wcs
#     # also save simple string version
#     data_params['WCS header string'] = wcs.to_header_string()
#     data_params['sky image params'] = {'filename':filename, 
#                                        'header':hdu.header,
#                                        'object':obj['object'], 
#                                        'pdf':obj['pdf'],
#                                        'survey':survey, 
#                                        'original img size':img_data.shape}

#     # xs,ys are just the pixel coords
#     xs = np.arange(0,img_data.shape[1])
#     ys = np.arange(0,img_data.shape[0])

#     #return xs,ys,colors,data_params
#     return xs, ys, img_data, data_params
        
