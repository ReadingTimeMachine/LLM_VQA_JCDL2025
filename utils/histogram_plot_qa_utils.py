import numpy as np

from utils.plot_qa_utils import plot_index_to_words


# this version tries to give column and row numbers
def q_nbars_hist_plot_plotnums(data, qa_pairs, plot_num = 0, return_qa=True, verbose=True, use_words=True):
    big_tag = 'nbars'
    object = 'bars'
    adder = ''
    # how many plots
    nplots = 0
    for k,v in data.items():
        if 'plot' in k:
            nplots += 1
    if nplots == 1: # if single figure, as just about this figure
        q = 'How many '+object+' are there on the figure? '
    else: # many plots 
        # rows columns
        if not use_words:
            nrow = data['figure']['plot indexes'][plot_num][0]
            ncol = data['figure']['plot indexes'][plot_num][1]
            q = 'The following question refers to the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '. '
            q += 'If there are multiple plots the panels will be in row-major (C-style) order, with the numbering starting at (0,0) in the upper left panel. '
            q += 'If there is one plot, then this row and column refers to the single plot. '
            q += 'How many '+object+' are there for the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '? '
            adder += '(plot numbers)'
        else: # translate to words
            q = 'How many '+object+' are there for the plot in the ' + plot_index_to_words(data['figure']['plot indexes'][plot_num]) + ' panel? '     
            adder += '(words)'
        
    q += 'You are a helpful assistant, please format the output as a json as {"'+big_tag+'":""} for this figure panel, where the "'+big_tag+'" value should be an integer.'
    # get answer
    a = {big_tag + ' ' + adder: len(data['plot'+str(plot_num)]['data from plot']['data'][0])}
    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', a)
    if return_qa: 
        if big_tag + ' ' + adder not in qa_pairs['Level 1']['Plot-level questions']:
            qa_pairs['Level 1']['Plot-level questions'][big_tag + ' ' + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a}}
        else:
            qa_pairs['Level 1']['Plot-level questions'][big_tag + ' ' + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a}
        return qa_pairs


# this version tries to give column and row numbers
def q_colors_hist_plot_plotnums(data, qa_pairs, plot_num = 0, return_qa=True, verbose=True, use_words=True):
    big_tag = 'bar color'
    object = 'What is the color of the bars'
    value_format = 'should be in the form of a single RGBA tuple. '
    value_empty = '""' # tpyically an empty list or empty string
    answer = data['plot'+str(plot_num)]['data from plot']['plot params']['bar color']
    adder = ''
    # how many plots
    nplots = 0
    for k,v in data.items():
        if 'plot' in k:
            nplots += 1
    if nplots == 1: # if single figure, as just about this figure
        q = object + ' on the figure? '
    else: # many plots 
        # rows columns
        if not use_words:
            nrow = data['figure']['plot indexes'][plot_num][0]
            ncol = data['figure']['plot indexes'][plot_num][1]
            q = 'The following question refers to the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '. '
            q += 'If there are multiple plots the panels will be in row-major (C-style) order, with the numbering starting at (0,0) in the upper left panel. '
            q += 'If there is one plot, then this row and column refers to the single plot. '
            #q += 'How many '+object+' are there for the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '? '
            q += object + 'for the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '? '
            adder += '(plot numbers)'
        else: # translate to words
            q = object + 'for the plot in the ' + plot_index_to_words(data['figure']['plot indexes'][plot_num]) + ' panel? '     
            adder += '(words)'
        
    q += 'You are a helpful assistant, please format the output as a json as {"'+big_tag+'":'+value_empty+'} for this figure panel, where the "'+big_tag+'" value '
    q += value_format #'should be an integer.'
    # get answer
    a = {big_tag + ' ' + adder: answer}
    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', a)
    if return_qa: 
        if big_tag + ' ' + adder not in qa_pairs['Level 1']['Plot-level questions']:
            qa_pairs['Level 1']['Plot-level questions'][big_tag + ' ' + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a}}
        else:
            qa_pairs['Level 1']['Plot-level questions'][big_tag + ' ' + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a}
        return qa_pairs


# this version tries to give column and row numbers
def q_colorlines_hist_plot_plotnums(data, qa_pairs, plot_num = 0, return_qa=True, verbose=True, use_words=True):
    big_tag = 'bar edge color'
    object = 'What is the color of the bar edge-lines'
    value_format = 'should be in the form of a single RGBA tuple. '
    value_empty = '""' # tpyically an empty list or empty string
    answer = data['plot'+str(plot_num)]['data from plot']['plot params']['edge color']
    adder = ''
    # how many plots
    nplots = 0
    for k,v in data.items():
        if 'plot' in k:
            nplots += 1
    if nplots == 1: # if single figure, as just about this figure
        q = object + ' on the figure? '
    else: # many plots 
        # rows columns
        if not use_words:
            nrow = data['figure']['plot indexes'][plot_num][0]
            ncol = data['figure']['plot indexes'][plot_num][1]
            q = 'The following question refers to the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '. '
            q += 'If there are multiple plots the panels will be in row-major (C-style) order, with the numbering starting at (0,0) in the upper left panel. '
            q += 'If there is one plot, then this row and column refers to the single plot. '
            #q += 'How many '+object+' are there for the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '? '
            q += object + 'for the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '? '
            adder += '(plot numbers)'
        else: # translate to words
            q = object + 'for the plot in the ' + plot_index_to_words(data['figure']['plot indexes'][plot_num]) + ' panel? '     
            adder += '(words)'
        
    q += 'You are a helpful assistant, please format the output as a json as {"'+big_tag+'":'+value_empty+'} for this figure panel, where the "'+big_tag+'" value '
    q += value_format #'should be an integer.'
    # get answer
    a = {big_tag + ' ' + adder: answer}
    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', a)
    if return_qa: 
        if big_tag + ' ' + adder not in qa_pairs['Level 1']['Plot-level questions']:
            qa_pairs['Level 1']['Plot-level questions'][big_tag + ' ' + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a}}
        else:
            qa_pairs['Level 1']['Plot-level questions'][big_tag + ' ' + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a}
        return qa_pairs


def q_linethickness_hists(data, qa_pairs, plot_num = 0, return_qa=True, use_words=True, use_list=False,
                      line_list = [], verbose=True):
    """
    use_words : set to True to translate row, column to words; False will use C-ordering indexing
    use_list : use a list of "property" options for prompting
    """
    
    big_tag = 'line thicknesses'
    big_tag_single = 'line thickness'
    big_tag_short = 'thickness'
    plot_param_tag = 'linethick'
    object = 'What are the matplotlib line thicknesses for bar edge-lines'
    value_empty = '""'
    value_format = 'where the value is a matplotlib line thickness in the format of an integer. '
    answer = data['plot'+str(plot_num)]['data from plot']['plot params']['linethick']
    # how many plots
    adder = ''
    nplots = 0
    for k,v in data.items():
        if 'plot' in k:
            nplots += 1
    if nplots == 1: # if single figure, as just about this figure
        q = object + ' on the figure? '
    else: # many plots 
        if not use_words:
            adder = '(plot numbers)'
            # rows columns
            nrow = data['figure']['plot indexes'][plot_num][0]
            ncol = data['figure']['plot indexes'][plot_num][1]
            q = 'The following question refers to the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '. '
            q += 'If there are multiple plots the panels will be in row-major (C-style) order, with the numbering starting at (0,0) in the upper left panel. '
            q += 'If there is one plot, then this row and column refers to the single plot. '
            q += object + ' for the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '? '
        else: 
            adder = '(words)'
            q = object + ' for the plot in the ' + plot_index_to_words(data['figure']['plot indexes'][plot_num]) + ' panel? '     

    q += 'You are a helpful assistant, please format the output as a json as {"'+big_tag_single+'":'+value_empty+'}, '
    q += value_format
    #where each element of the list refers to the '
    #q += ' '+big_tag_short+' of each line in the figure panel in the form a matplotlib '+big_tag_short+' type. '
    if use_list:
        adder = adder.split(')')[0] + ' + list)'
        q += 'Please choose each '+big_tag_short+' from the following list: ['
        for pt in line_list:
            q += str(pt) + ', '
        q = q[:-2] # take off the last bit
        q += '].'

    # # get answer list
    # la = []
    # for k,v in data.items():
    #     if 'plot' + str(plot_num) == k: # is a plot
    #         for c in v['data from plot']['plot params'][plot_param_tag]:
    #             la.append(c) 

    k = big_tag + ' ' + adder
    k = k.strip()
    k = k.replace('  ', ' ')
    a = {k:answer} 
    ans = {k:{'plot'+str(plot_num):answer}} 
    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', a)
    if return_qa: 
        if big_tag + ' ' + adder not in qa_pairs['Level 1']['Plot-level questions']:
            qa_pairs['Level 1']['Plot-level questions'][big_tag + ' ' + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a}}
        else:
            qa_pairs['Level 1']['Plot-level questions'][big_tag + ' ' + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a}
        return qa_pairs


def q_barpoints_bars(data, qa_pairs, plot_num = 0, return_qa=True, use_words=True, verbose=True):
    """
    use_words : set to True to translate row, column to words; False will use C-ordering indexing
    use_list : use a list of "property" options for prompting
    """
    big_tag = 'bar locations'
    object = 'What are the locations of the bars'
    # how many plots
    nplots = 0
    for k,v in data.items(): # count number of plots
        if 'plot' in k:
            nplots += 1
    if not use_words:
        adder = '(plot numbers)'
        # rows columns
        nrow = data['figure']['plot indexes'][plot_num][0]
        ncol = data['figure']['plot indexes'][plot_num][1]
        q = 'The following question refers to the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '. '
        q += 'If there are multiple plots the panels will be in row-major (C-style) order, with the numbering starting at (0,0) in the upper left panel. '
        q += 'If there is one plot, then this row and column refers to the single plot. '
    else: 
        adder = '(words)'

    if nplots == 1: # single plot
        q = object+' in this figure? '
    else:
        if not use_words:
            q += object+' for the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '? '
        else:
            q = object+' for the plot in the ' + plot_index_to_words(data['figure']['plot indexes'][plot_num]) + ' panel? '     

    q += 'You are a helpful assistant, please format the output as a json as {"left":[], "right":[], "center",[], "height":[]} '
    q += 'where the values of "left" and "right" are the left and right edges of each bar, and "center" is the center of each bar, all in terms of the x-axis values. '
    q += 'Additionally, "height" is the location of the top of each bar and should be in terms of the y-axis values. ' 
    q += 'All values should in the format of a list of floats.  '

    # calculate answer
    heights = data['plot'+str(plot_num)]['data from plot']['data'][0]
    bin_edges = np.array(data['plot'+str(plot_num)]['data from plot']['data'][1])
    #print(bin_edges)
    centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()
    left = bin_edges[:-1].tolist()
    right = bin_edges[1:].tolist()

    k = big_tag + ' ' + adder
    k = k.strip()
    k = k.replace('  ', ' ')
    a = {k:{'left':left, 'right':right, 'center':centers, 'height':heights}}
    ans = {k:{'plot'+str(plot_num):{'left':left, 'right':right, 'center':centers, 'height':heights}}}

    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', ans)
    if return_qa: 
        if big_tag + ' ' + adder not in qa_pairs['Level 1']['Plot-level questions']:
            qa_pairs['Level 1']['Plot-level questions'][big_tag + ' ' + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a}}
        else:
            qa_pairs['Level 1']['Plot-level questions'][big_tag + ' ' + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a}
        return qa_pairs


def q_stats_hists(data, qa_pairs, stat = {'minimum':np.min}, plot_num = 0, 
                     return_qa=True, use_words=True, verbose=True):
    """
    stat: {'name':stat} which gives name of stat and function to calculate it, like {'minimum':np.min}
    use_words : set to True to translate row, column to words; False will use C-ordering indexing
    stat : dictionary of the name and function to use for each stat
    """
    big_tag = list(stat.keys())[0]
    # how many plots
    nplots = 0
    for k,v in data.items(): # count number of plots
        if 'plot' in k:
            nplots += 1
    if not use_words:
        adder = '(plot numbers)'
        # rows columns
        nrow = data['figure']['plot indexes'][plot_num][0]
        ncol = data['figure']['plot indexes'][plot_num][1]
        q = 'The following question refers to the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '. '
        q += 'If there are multiple plots the panels will be in row-major (C-style) order, with the numbering starting at (0,0) in the upper left panel. '
        q += 'If there is one plot, then this row and column refers to the single plot. '
    else: 
        adder = '(words)'

    if nplots == 1: # single plot
        q = 'What are the '+big_tag+' data values in this figure? '
    else:
        if not use_words:
            q += 'What are the '+big_tag+' data values for the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '? '
        else:
            q = 'What are the '+big_tag+' data values for the plot in the ' + plot_index_to_words(data['figure']['plot indexes'][plot_num]) + ' panel? '     

    q += 'You are a helpful assistant, please format the output as a json as {"'+big_tag+' x":""} where the '+big_tag+' value of "x" is calculated from  '
    q += 'the data values used to create the plot in the format of floats.  '
    
    # get answer list
    f = list(stat.values())[0] # what stastical function
    xs = data['plot'+str(plot_num)]['data']['xs']
    #ys = data['plot'+str(plot_num)]['data']['ys']
    la = {big_tag + " x":f(xs)}#, big_tag + " y":f(ys)}
    
    ans = {big_tag + ' ' + adder:{'plot'+str(plot_num):la}} 
    a = {big_tag + ' ' + adder:la}
    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', ans)
    #a = list(aout.values())[0]
    if return_qa: 
        if big_tag + ' ' + adder not in qa_pairs['Level 2']['Plot-level questions']:
            qa_pairs['Level 2']['Plot-level questions'][big_tag + ' ' + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a}}
        else:
            qa_pairs['Level 2']['Plot-level questions'][big_tag + ' ' + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a}
        return qa_pairs


def q_errorbars_existance_lines(data, qa_pairs, axis = 'x', plot_num = 0, return_qa=True, use_words=True, verbose=True, name_of_axis=None):
    """
    use_words : set to True to translate row, column to words; False will use C-ordering indexing
    """
    if name_of_axis == None:
        name_of_axis = axis
    # how many plots
    nplots = 0
    for k,v in data.items(): # count number of plots
        if 'plot' in k:
            nplots += 1
    if not use_words:
        adder = '(plot numbers)'
        # rows columns
        nrow = data['figure']['plot indexes'][plot_num][0]
        ncol = data['figure']['plot indexes'][plot_num][1]
        q = 'The following question refers to the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '. '
        q += 'If there are multiple plots the panels will be in row-major (C-style) order, with the numbering starting at (0,0) in the upper left panel. '
        q += 'If there is one plot, then this row and column refers to the single plot. '
    else: 
        adder = '(words)'

    if nplots == 1: # single plot
        q = 'Are there error bars on the data along the ' +name_of_axis+ '-axis in this figure? '
    else:
        if not use_words:
            q += 'Are there error bars on the data along the ' +name_of_axis+ '-axis for the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '? '
        else:
            q = 'Are there error bars on the data along the ' +name_of_axis+ '-axis for the plot in the ' + plot_index_to_words(data['figure']['plot indexes'][plot_num]) + ' panel? '     

    # q += 'You are a helpful assistant, please format the output as a json as {"'+axis+'-axis errors":[]} where the value for output is True '
    # q += '(if error bars exist) or False (if error bars do not exist). '
    # q += 'If there is one single array for '+axis+' data values for all lines on the plot, there will be a single True or False in the list, otherwise it will be list of True/False. '
    q += 'You are a helpful assistant, please format the output as a json as {"'+name_of_axis+'-axis errors":""} where the value for the output is True '
    q += '(if error bars exist) or False (if error bars do not exist). '

    # get answer
    if axis+'errs' in data['plot' + str(plot_num)]['data'].keys():
        a = {name_of_axis+'-axis errors' + ' ' + adder:True}
    else:
        a = {name_of_axis+'-axis errors' + ' ' + adder:False}

    #a = list(aout.values())[0]
    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', a)
    if return_qa: 
        if axis+'-axis errors' + ' ' + adder not in qa_pairs['Level 2']['Plot-level questions']:
            qa_pairs['Level 2']['Plot-level questions'][name_of_axis+'-axis errors' + ' ' + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a}}
        else:
            qa_pairs['Level 2']['Plot-level questions'][name_of_axis+'-axis errors' + ' ' + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a}
        return qa_pairs



def q_errorbars_size_histogram(data, qa_pairs, axis = 'x', plot_num = 0, return_qa=True, use_words=True, verbose=True, 
                             use_avg = False, name_of_axis='y'):
    """
    use_words : set to True to translate row, column to words; False will use C-ordering indexing
    use_avg : calculate the average size of the errorbars
    """
    if name_of_axis == None:
        name_of_axis = axis
    
    # how many plots
    nplots = 0
    for k,v in data.items(): # count number of plots
        if 'plot' in k:
            nplots += 1
    if not use_words:
        adder = '(plot numbers)'
        # rows columns
        nrow = data['figure']['plot indexes'][plot_num][0]
        ncol = data['figure']['plot indexes'][plot_num][1]
        q = 'The following question refers to the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '. '
        q += 'If there are multiple plots the panels will be in row-major (C-style) order, with the numbering starting at (0,0) in the upper left panel. '
        q += 'If there is one plot, then this row and column refers to the single plot. '
    else: 
        adder = '(words)'

    if nplots == 1: # single plot
        q = 'What is the approximate size of the error bars along the ' + name_of_axis + '-axis in this figure? '
    else:
        if not use_words:
            q += 'What is the approximate size of the error bars along the ' + name_of_axis + '-axis for the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '? '
        else:
            q = 'What is the approximate size of the error bars along the ' + name_of_axis + '-axis for the plot in the ' + plot_index_to_words(data['figure']['plot indexes'][plot_num]) + ' panel? '     

    if not use_avg:
        q += 'You are a helpful assistant, please format the output as a json as {"'+name_of_axis+'-axis error size":[]} where each element of the list corresponds to a single bar in the figure. '
    else:
        q += 'You are a helpful assistant, please format the output as a json as {"'+name_of_axis+'-axis error size":""} where the value is a float and the average size of the error bars along this axis. '
        q += 'The average size of the error bar can be calculated as the average length of the bar divided by the difference between the maximum and minimum data values along the '+name_of_axis+'-axis.'
        
    q += 'For reference, errorbars are shown with the matplotlib "ax.errorbar" function. '

    # get answer
    if axis+'errs' in data['plot' + str(plot_num)]['data'].keys(): # just a double check
        if use_avg:
            v = data['plot'+str(plot_num)]['data'][axis+'s']
            errs = np.mean(np.abs(data['plot'+str(plot_num)]['data'][axis+'errs']))/np.abs((np.max(v)-np.min(v)))
            a = {name_of_axis+'-axis errors size' + ' ' + adder:errs}
        else:
            # not-averaged tag
            adder = adder.split(')')[0] + ' + non-averaged)'
            a = {name_of_axis+'-axis errors size' + ' ' + adder:data['plot'+str(plot_num)]['data'][axis+'errs']}
            #print('a:', a)
            #print('data:', data['plot'+str(plot_num)]['data'][axis+'errs'])
    else:
        print('No axis error bars!!')
        import sys; sys.exit()

    #a = aout.values[0]

    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', {'plot'+str(plot_num):a})
    if return_qa: 
        if axis+'-axis error size' + ' ' + adder not in qa_pairs['Level 2']['Plot-level questions']:
            qa_pairs['Level 2']['Plot-level questions'][axis+'-axis error size' + ' ' + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a}}
        else:
            qa_pairs['Level 2']['Plot-level questions'][axis+'-axis error size' + ' ' + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a}
        return qa_pairs


def q_distribution_hists(data, qa_pairs, plot_num = 0, return_qa=True, use_words=True, use_list=True, 
                        line_list = ['random','linear','gaussian mixture model'], verbose=True):
    """
    use_words : set to True to translate row, column to words; False will use C-ordering indexing
    use_list : give a list of possible distributions
    use_nplots : give the number of lines in the prompt
    """
    
    # how many plots
    big_tag_short = 'distribution'
    object = 'What is the distribution used to create the histogram'
    nplots = 0
    for k,v in data.items(): # count number of plots
        if 'plot' in k:
            nplots += 1
    if not use_words:
        adder = '(plot numbers)'
        # rows columns
        nrow = data['figure']['plot indexes'][plot_num][0]
        ncol = data['figure']['plot indexes'][plot_num][1]
        q = 'The following question refers to the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '. '
        q += 'If there are multiple plots the panels will be in row-major (C-style) order, with the numbering starting at (0,0) in the upper left panel. '
        q += 'If there is one plot, then this row and column refers to the single plot. '
    else: 
        adder = '(words)'

    if nplots == 1: # single plot
        q = object + ' in this figure? '
    else:
        if not use_words:
            q += object + ' for the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '? '
        else:
            q = object + ' for the plot in the ' + plot_index_to_words(data['figure']['plot indexes'][plot_num]) + ' panel? '     

    q += 'You are a helpful assistant, please format the output as a json as {"'+big_tag_short+'":""} the value is a string. '

    if use_list:
        adder = adder.split(')')[0] + ' + list)'
        q += 'Please choose each '+big_tag_short+' from the following list: ['
        for pt in line_list:
            q += str(pt) + ', '
        q = q[:-2] # take off the last bit
        q += '].'

    # answer
    dist = data['plot'+str(plot_num)]['distribution']
    # to match
    if dist == 'gmm': dist = 'gaussian mixture model'
    #a = la
    a = {big_tag_short + ' ' + adder:dist}

    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', {'plot'+str(plot_num):a})
    if return_qa: 
        if big_tag_short + ' ' + adder not in qa_pairs['Level 3']['Plot-level questions']:
            #print('yes', big_tag_short + ' ' + adder)
            qa_pairs['Level 3']['Plot-level questions'][big_tag_short + ' ' + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a}}
        else:
            qa_pairs['Level 3']['Plot-level questions'][big_tag_short + ' ' + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a}
        return qa_pairs



def q_linear_equation_hists(data, qa_pairs, plot_num = 0, return_qa=True, use_words=True, verbose=True):
    """
    use_words : set to True to translate row, column to words; False will use C-ordering indexing
    use_nlines : give the number of lines in the prompt
    """

    big_tag_short = 'linear parameters'
    object = 'What are the parameters for the linear distribution used to create the histogram'
    # is this a linear relationship? if not this is an error
    hasLine = False
    if data['plot'+str(plot_num)]['distribution'] != 'linear': # not a linear relationship
        if verbose:
            print('Not a linear relationship!')
        if return_qa:
            return qa_pairs
        else:
            return None

    if data['plot'+str(plot_num)]['distribution'] == 'linear':
        hasLine = True

    if not hasLine:
        print("This is not a linear relationship, can't ask any questions!'")
        import sys; sys.exit()
        
    adder = ''
    # how many plots
    nplots = 0
    for k,v in data.items(): # count number of plots
        if 'plot' in k:
            nplots += 1
    if not use_words:
        adder += '(plot numbers)'
        # rows columns
        nrow = data['figure']['plot indexes'][plot_num][0]
        ncol = data['figure']['plot indexes'][plot_num][1]
        q = 'The following question refers to the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '. '
        q += 'If there are multiple plots the panels will be in row-major (C-style) order, with the numbering starting at (0,0) in the upper left panel. '
        q += 'If there is one plot, then this row and column refers to the single plot. '
    else: 
        adder += '(words)'

    if nplots == 1: # single plot
        q = object + ' in this figure? '
    else:
        if not use_words:
            q += object + ' for the figure panel on row number ' + \
              str(nrow) + ' and column number ' + str(ncol) + '? '
        else:
            q = object + ' for the plot in the ' + \
              plot_index_to_words(data['figure']['plot indexes'][plot_num]) + ' panel? '     


    q += 'You are a helpful assistant, please format the output as a json as {"'+big_tag_short+'":{"m":"","a":"","noise level":""}}. '
    q += 'Here, "m" is the slope of the relationship and "a" is the intercept and both should be floats. '
    # noise
    q += 'The "noise level" parameter should be the relative amount of noise added to the '
    q += 'linear function for the relationship and should be a float between 0 and 1. '
    
    # answer
    la = data['plot'+str(plot_num)]['data']['data params']

    a = {big_tag_short + ' ' + adder:la}

    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', {'plot'+str(plot_num):a})
    if return_qa: 
        if big_tag_short + ' ' + adder not in qa_pairs['Level 3']['Plot-level questions']:
            qa_pairs['Level 3']['Plot-level questions'][big_tag_short + ' ' + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a}}
        else:
            qa_pairs['Level 3']['Plot-level questions'][big_tag_short + ' ' + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a}
        return qa_pairs



def q_gmm_equation_hists(data, qa_pairs, plot_num = 0, return_qa=True, use_words=True, verbose=True):
    """
    use_words : set to True to translate row, column to words; False will use C-ordering indexing
    use_nlines : give the number of lines in the prompt
    """

    big_tag_short = 'gmm parameters'
    object = 'What are the parameters for the gaussian mixture model distribution used to create the histogram'

    # is this a linear relationship? if not this is an error
    hasLine = False
    if data['plot'+str(plot_num)]['distribution'] != 'gmm': # not a linear relationship
        if verbose:
            print('Not a gmm relationship!')
        if return_qa:
            return qa_pairs
        else:
            return None

    if data['plot'+str(plot_num)]['distribution'] == 'gmm':
        hasLine = True

    if not hasLine:
        print("This is not a gmm relationship, can't ask any questions!'")
        import sys; sys.exit()
        
    adder = ''
    # how many plots
    nplots = 0
    for k,v in data.items(): # count number of plots
        if 'plot' in k:
            nplots += 1
    if not use_words:
        adder += '(plot numbers)'
        # rows columns
        nrow = data['figure']['plot indexes'][plot_num][0]
        ncol = data['figure']['plot indexes'][plot_num][1]
        q = 'The following question refers to the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '. '
        q += 'If there are multiple plots the panels will be in row-major (C-style) order, with the numbering starting at (0,0) in the upper left panel. '
        q += 'If there is one plot, then this row and column refers to the single plot. '
    else: 
        adder += '(words)'

    if nplots == 1: # single plot
        q = object + ' in this figure? '
    else:
        if not use_words:
            q += object + ' for the figure panel on row number ' + \
              str(nrow) + ' and column number ' + str(ncol) + '? '
        else:
            q = object + ' for the plot in the ' + \
              plot_index_to_words(data['figure']['plot indexes'][plot_num]) + ' panel? '  

    # 'nsamples', 'nclusters', 'centers', 'cluster_std', 'noise level'
    q += 'You are a helpful assistant, please format the output as a json as {"'+big_tag_short+\
          '":{"nsamples":"","nclusters":"","centers":"","cluster_std":"","noise level":""}}. '
    q += 'Each "nsamples" is the number of samples in the distribution and should be an integer. '
    q += 'The "nclusters" parameter should be the number of clusters in the model and should be an integer. '
    q += 'The "centers" parameter should be the position of each cluster and should be a list of points which are floats. '
    q += 'The "cluster_std" parameter should be the standard deviation of each cluster and should be a list of points which are floats. '
    q += 'The "noise level" parameter should be the relative amount of noise added to the model and should be a float between 0 and 1. '
    
    # answer
    la = data['plot'+str(plot_num)]['data']['data params']

    a = {big_tag_short + ' ' + adder:la}

    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', {'plot'+str(plot_num):a})
    if return_qa: 
        if big_tag_short + ' ' + adder not in qa_pairs['Level 3']['Plot-level questions']:
            qa_pairs['Level 3']['Plot-level questions'][big_tag_short + ' ' + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a}}
        else:
            qa_pairs['Level 3']['Plot-level questions'][big_tag_short + ' ' + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a}
        return qa_pairs


def q_gmm_ngaussians_hists(data, qa_pairs, plot_num = 0, return_qa=True, use_words=True, verbose=True):
    """
    use_words : set to True to translate row, column to words; False will use C-ordering indexing
    use_nlines : give the number of lines in the prompt
    """

    big_tag_short = 'gmm ngaussians'
    #object = 'What are the parameters for the gaussian mixture model distribution used to create the histogram'
    object = 'How many gaussians have been used to generate the histogram'

    # is this a linear relationship? if not this is an error
    hasLine = False
    if data['plot'+str(plot_num)]['distribution'] != 'gmm': # not a linear relationship
        if verbose:
            print('Not a gmm relationship!')
        if return_qa:
            return qa_pairs
        else:
            return None

    if data['plot'+str(plot_num)]['distribution'] == 'gmm':
        hasLine = True

    if not hasLine:
        print("This is not a gmm relationship, can't ask any questions!'")
        import sys; sys.exit()
        
    adder = ''
    # how many plots
    nplots = 0
    for k,v in data.items(): # count number of plots
        if 'plot' in k:
            nplots += 1
    if not use_words:
        adder += '(plot numbers)'
        # rows columns
        nrow = data['figure']['plot indexes'][plot_num][0]
        ncol = data['figure']['plot indexes'][plot_num][1]
        q = 'The following question refers to the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '. '
        q += 'If there are multiple plots the panels will be in row-major (C-style) order, with the numbering starting at (0,0) in the upper left panel. '
        q += 'If there is one plot, then this row and column refers to the single plot. '
    else: 
        adder += '(words)'

    if nplots == 1: # single plot
        q = object + ' in this figure? '
    else:
        if not use_words:
            q += object + ' for the figure panel on row number ' + \
              str(nrow) + ' and column number ' + str(ncol) + '? '
        else:
            q = object + ' for the plot in the ' + \
              plot_index_to_words(data['figure']['plot indexes'][plot_num]) + ' panel? '  

    # 'nsamples', 'nclusters', 'centers', 'cluster_std', 'noise level'
    q += 'You are a helpful assistant, please format the output as a json as {"'+big_tag_short+\
          '":""}}. '
    q += 'Where the "ngaussians" parameter should be the number of gaussians used to generate the histogram and should be an integer. '

    # answer
    la = data['plot'+str(plot_num)]['data']['data params']['nclusters']

    a = {big_tag_short + ' ' + adder:la}

    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', {'plot'+str(plot_num):a})
    if return_qa: 
        if big_tag_short + ' ' + adder not in qa_pairs['Level 3']['Plot-level questions']:
            qa_pairs['Level 3']['Plot-level questions'][big_tag_short + ' ' + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a}}
        else:
            qa_pairs['Level 3']['Plot-level questions'][big_tag_short + ' ' + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a}
        return qa_pairs