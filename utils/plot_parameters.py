
###### MAIN FIGURE #######
base = 5
aspect_fig_params = {'min':0.25, 'max':1.5}#, 'prob function':np.random.uniform} # w/h
dpi_params = {'min':75, 'max':500}#, 'prob function':np.random.uniform} #300
tight_layout_params = {True:0.9} # probability of having a "tight layout"

panel_params = {
    'number prob': {
        'min':1, # minimum number of panels
        'max':25, # maximum number of panels
        'median':1,
        'std':10
    },
    'layout prob':{'horizontal':0.5,'vertical':0.25, 'squarish':0.25}, # probability things are horizontal, vertical, square-ish
    'to even above':5 # above this amount, panels will be put into a square-ish shape
    }

fontsizes = {'title':{'min':10, 'max':20},
             'colorbar':{'min':6, 'max':18},
            'xlabel':{'min':6, 'max':20},
            'ylabel':{'min':6, 'max':20},
            'ticks':{'min':4, 'max':20},
            'x/y label same': True, # fontsize of x&y labels are the same
            'x/y ticks same': True} # fontsize of the x&y tick labels are the same


###### TITLE & X/Y LABELS #######
title_params = {'prob':0.5, 'n words':{'min':1, 'max':5},
               'capitalize':{'none':0.1, 'first':0.5, 'all':0.7}, # none is all lower case, first just capitalizes first word, all is all uppercase
                'equation':{'prob':0.1}, # probability that any one word will be an equation (~0.25)
               }
colorbar_params = {'prob':0.5, 'n words':{'min':1, 'max':3},
               'capitalize':{'none':0.1, 'first':0.5, 'all':0.7}, # none is all lower case, first just capitalizes first word, all is all uppercase
                'equation':{'prob':0.1}, # probability that any one word will be an equation (~0.25)
                }
xlabel_params = {'prob':0.90, 'n words':{'min':1, 'max':3},
                'capitalize':{'none':0.1, 'first':0.5, 'all':0.7}, # none is all lower case, first just capitalizes first word, all is all uppercase
                'equation':{'prob':0.25}, # probability that any one word will be an equation (~0.25)
                }
ylabel_params = {'prob':0.90, 'n words':{'min':1, 'max':3},                 
                 'capitalize':{'none':0.1, 'first':0.5, 'all':0.7}, # none is all lower case, first just capitalizes first word, all is all uppercase
                'equation':{'prob':0.25}, # probability that any one word will be an equation
                }

######## PLOT PARAMS ############
plot_types_params = {
                     'line':{
                         'prob':1, # probability of getting this plot
                         'npoints':{'min':10,'max':11}, 
                         'line thick':{'min':1, 'max':5},
                         'nlines':{'min':2, 'max':3}, 
                         'xmin':-10000,
                         'xmax':10000,
                         'ymin':-10000,
                         'ymax':10000,
                         'prob same x': 0.1, # probability that all points have same x-value
                         'markers':{'prob':0.5, # probability that we have a marker on this line (types randomly selected from all matplotlib)
                                    'size':{'min':1, 'max':20}
                                   },
                         'error bars':{
                             'elinewidth':{'min':1, 'max':5},
                             'x':{
                                 'prob':0.25, # probability for x-axis error bars
                                 'size':{'min':0.01, 'max':0.1}, # each error will be sampled randomly from these percentages of the x-axis width
                               }, 
                             'y':{
                                 'prob':0.25, # probability for x-axis error bars
                                 'size':{'min':0.01, 'max':0.1}, # each error will be sampled randomly from these percentages of the x-axis width
                               }, 
                            }, # end errorbars
                         'distribution': {
                             'random':{'prob':1},
                             'linear':{'prob':1, 
                                       'intersect':(-100,100), # range of "a" in mx + a
                                       'slope':(-5, 5), # range of "m" in mx + a
                                       'noise':(0, 0.25) # noise % range
                                      },
                             'gmm':{ # gaussian mixture model
                                 'prob':1, 
                                 'histogram as line':{ # plot this histo as a line?
                                     'prob':0.25, 
                                     'factor':1000
                                 }, 
                                 'nclusters':{'min':1, 'max':20},
                                 'nsamples':{'min':10, 'max':500},
                                 'cluster std':{'min':-2, 'max':0.5}, # in terms of factors of the x/y ranges
                                 'noise':{'min':0.05,'max':0.25} # for noise in distribution and color when applicable
                             } # end gmm distribution for line plots
                         } # end distribution for line plots
                     }, # end line plots
                     'histogram':{'prob':1,
                         'npoints':{'min':10,'max':10000}, # points for distribution
                         'nbins':{'min':1, 'max':100}, # number of bars
                         'rwidth':{'min':0.2,'max':1.0}, # bin width
                         'line thick':{'prob':0.5, 'min':1, 'max':5}, # 0 means no lines
                         'nlines':{'min':2, 'max':3}, 
                         'xmin':-10000, # only xmin/xmax for 
                         'xmax':10000,
                         'error bars':{ # if vertical, this will be on y-axis, of horizontal, then x-axis
                             'elinewidth':{'min':1, 'max':5},
                             'x':{
                                 'prob':1, # probability for x-axis error bars
                                 'size':{'min':0.01, 'max':0.1}, # each error will be sampled randomly from these percentages of the x-axis width
                               }
                            },
                         'horizontal prob':0.25, # probability that we have a horizontal bar plot
                         'distribution': {
                             'random':{'prob':1},
                             'linear':{'prob':1, 
                                       'intersect':(-100,100), # range of "a" in mx + a
                                       'slope':(-5, 5), # range of "m" in mx + a
                                       'noise':(0, 0.25) # noise % range 
                                      },
                             'gmm':{ # gaussian mixture model
                                 'prob':1, 
                                 'xmin':-10000, # ranges
                                 'xmax':10000,
                                 'ymin':-10000,
                                 'ymax':10000, 
                                 'nclusters':{'min':1, 'max':20},
                                 'nsamples':{'min':10, 'max':500},
                                 'cluster std':{'min':-1, 'max':1.5}, # in terms of factors of the x/y ranges
                                 'noise':{'min':0.05,'max':0.25} # for noise in distribution and color when applicable
                             }
                         } # end distribution for histogram plots
                    },  # end histogram plots
                     'scatter':{'prob':1,                        
                         'npoints':{'min':10,'max':100}, 
                         'markers':{
                               'size':{'min':1, 'max':30}
                                },
                         'colors':{'min':-100, 'max':100}, # values for the colors to have
                         'colormap scatter':{'prob':0.95}, # prob for scatter plot to have a colormap
                         'color bar':{'location probs':{'right':0.5, 'left':0.05, 'top':0.25, 'bottom':0.05},
                                      'label prob':0.5, # probability has label
                                     'size percent':{'min':0.05, 'max':0.15},
                                     'pad':{'min':0.01, 'max':0.2}
                                     },
                         'xmin':-10000,
                         'xmax':10000,
                         'ymin':-10000,
                         'ymax':10000,
                         'error bars':{
                             'elinewidth':{'min':1, 'max':5},
                             'x':{
                                 'prob':0.25, # probability for x-axis error bars
                                 'size':{'min':0.01, 'max':0.1}, # each error will be sampled randomly from these percentages of the x-axis width
                               }, 
                             'y':{
                                 'prob':0.25, # probability for x-axis error bars
                                 'size':{'min':0.01, 'max':0.1}, # each error will be sampled randomly from these percentages of the x-axis width
                               },
                             },
                         'distribution': {
                             'random':{'prob':1},
                             'linear':{'prob':1, 
                                       'intersect':(-100,100), # range of "a" in mx + a
                                       'slope':(-5, 5), # range of "m" in mx + a
                                       'noise':(0, 0.25), # noise % range
                                       'color noise prob': 0.5 # linear relationship between x/y and color?
                                      },
                             'gmm':{ # scatter gaussian mixture model
                                 'prob':1, 
                                 'upsample factor log':{'min':2, 'max':6}, # upsample number of points, 10^X
                                 'nclusters':{'min':1, 'max':5},
                                 'nsamples':{'min':1000, 'max':50000},
                                 'cluster std':{'min':-1, 'max':0.5}, # in terms of factors of the x/y ranges
                                 'noise':{'min':0.05,'max':0.25}, # for noise in distribution and color when applicable
                                 'color noise prob': 0.5, # gmm relationship between x/y and color?
                                 'max points': 200 # maximum number of points

                             } # end gmm for scatter plots
                         } # end distribution for line plots
                       }, # end scatter plots
                    # contour plots
                     'contour':{'prob':1,                        
                         'npoints':{'nx':{'min':10,'max':500}, 'ny':{'min':10,'max':500}}, 
                         'nlines':{'min':3, 'max':10}, # number of contour lines
                         # 'markers':{
                         #       'size':{'min':1, 'max':30}
                         #        },
                         'colors':{'min':-100, 'max':100}, # values for the colors to have
                         'colormap contour':{'prob':0.95}, # prob for contour plot to have a colormap
                         'color bar':{'location probs':{'right':0.5, 
                                                        'left':0.05, 
                                                        'top':0.25, 
                                                        'bottom':0.05},
                                        'label prob': 0.5, # prob has label
                                     'size percent':{'min':0.05, 'max':0.15},
                                     'pad':{'min':0.01, 'max':0.2}
                                     },
                         'image or contour':{'prob':{'image':1, 
                                                     'contour':1, 
                                                     'both':1}, # probability that this is just an image w/o contour lines (or both) 
                                            'both contours':{'prob gray': 0.95, # probability that you'll have just gray colors for contours
                                                            }
                                            }, 
                         'xmin':-10000,
                         'xmax':10000,
                         'ymin':-10000,
                         'ymax':10000,
                         'distribution': {
                             'random':{'prob':1},
                             'linear':{'prob':1, 
                                       'intersect':(-100,100), # range of "a" in mx + a
                                       'slope':(-5, 5), # range of "m" in mx + a
                                       'noise':(0, 0.25) # noise % range 
                                      },
                             'gmm':{ # contour gaussian mixture model
                                 'prob':1, 
                                 'nclusters':{'min':1, 'max':500},
                                 #'nsamples':{'min':10, 'max':500},
                                 'upsample factor log':{'min':0, 'max':0.5}, # upsample number of points, 10^X (was 2, 6)
                                 'cluster std':{'min':-1, 'max':1.5}, # in terms of factors of the x/y ranges
                                 'noise':{'min':0.05,'max':0.25}, # for noise in distribution and color when applicable
                                 'color noise prob': 0.5 # gmm relationship between x/y and color?
                             }, # end gmm for contour plots 
                            } # end distribution for contour plots 
                        } # end contour plots

                             
} # end specificiations


