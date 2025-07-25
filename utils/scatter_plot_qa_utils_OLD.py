import numpy as np

from utils.plot_qa_utils import plot_index_to_words


# this version tries to give column and row numbers
def q_npoints_scatter_plot_plotnums(data, qa_pairs, plot_num = 0, return_qa=True, verbose=True, use_words=True):
    big_tag = 'npoints'
    object = 'scatter points'
    adder = ''
    # how many plots
    nplots = 0
    for k,v in data.items():
        if 'plot' in k:
            nplots += 1
    if nplots == 1: # if single figure, as just about this figure
        q = 'How many scatter points are there for on the figure? '
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
    a = {big_tag + ' ' + adder: len(data['plot'+str(plot_num)]['data']['xs'])}
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
def q_colors_scatter_plot_plotnums(data, qa_pairs, plot_num = 0, return_qa=True, verbose=True, use_words=True):
    big_tag = 'point colors'
    object = 'What are the colors of each of the scatter plot points '
    value_format = 'should be in the form of an RGBA tuple, with one tuple for each scatter plot point. '
    value_empty = '[]' # tpyically an empty list or empty string
    answer = data['plot'+str(plot_num)]['data pixels']['colors'].tolist()
    adder = ''
    # how many plots
    nplots = 0
    for k,v in data.items():
        if 'plot' in k:
            nplots += 1
    if nplots == 1: # if single figure, as just about this figure
        q = 'How many scatter points are there for on the figure? '
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


def q_linemarkers_scatters(data, qa_pairs, plot_num = 0, return_qa=True, use_words=True, use_list=False,
                      line_list = [], verbose=True):
    """
    use_words : set to True to translate row, column to words; False will use C-ordering indexing
    use_list : use a list of "property" options for prompting
    """
    
    big_tag = 'scatter plot markers'
    big_tag_single = 'scatter plot marker'
    big_tag_short = 'marker'
    plot_param_tag = 'markers'
    extra_if_empty = '' # nothing extra for this
    value_format = 'where the value of the entry should be the matplotlib marker used for this scatter plot. '
    value_empty = '""' # tpyically an empty list or empty string
    answer = data['plot'+str(plot_num)]['data from plot']['marker'] # note: not sure this should be in "color bar params!"
    object = 'What are the ' +big_tag+ ' of the points'
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
        # how many plots
        nplots = 0
        for k,v in data.items():
            if 'plot' in k:
                nplots += 1
        if nplots == 1: # single plot
            q = 'What are the matplotlib '+big_tag+' in this figure? '
        else:
            q = 'What are the matplotlib '+big_tag+' for the plot in the ' + plot_index_to_words(data['figure']['plot indexes'][plot_num]) + ' panel? '     

    q += 'You are a helpful assistant, please format the output as a json as {"'+big_tag_single+'":'+value_empty+'}, '
    q += value_format
    #where each element of the list refers to the '
    #q += ' '+big_tag_short+' of each line in the figure panel in the form a matplotlib '+big_tag_short+' type. '
    q += extra_if_empty
    if use_list:
        adder = adder.split(')')[0] + ' + list)'
        q += 'Please choose each '+big_tag_short+' from the following list: ['
        for pt in line_list:
            q += str(pt) + ', '
        q = q[:-2] # take off the last bit
        q += '].'

    a = {big_tag + ' ' + adder:answer}
    ans = {'plot' + str(plot_num):a}
    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', ans)
    if return_qa: 
        if big_tag + ' ' + adder not in qa_pairs['Level 1']['Plot-level questions']:
            qa_pairs['Level 1']['Plot-level questions'][big_tag + ' ' + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a}}
        else:
            qa_pairs['Level 1']['Plot-level questions'][big_tag + ' ' + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a}
        return qa_pairs



def q_linemarkersize_scatters(data, qa_pairs, plot_num = 0, return_qa=True, use_words=True, use_list=False,
                      line_list = [], verbose=True):
    """
    use_words : set to True to translate row, column to words; False will use C-ordering indexing
    use_list : use a list of "property" options for prompting
    """
    
    big_tag = 'scatter point marker sizes'
    big_tag_single = 'marker size'
    big_tag_short = 'marker size'
    plot_param_tag = 'marker size' # tag in the plot params
    extra_if_empty = '' # nothing extra for this
    value_format = 'where the value of the entry should be the matplotlib marker size used for this scatter plot as an integer. '
    value_empty = '""' # tpyically an empty list or empty string
    answer = data['plot'+str(plot_num)]['data from plot']['marker size'] # note: not sure this should be in "color bar params!"
    object = 'What are the ' +big_tag+ ' of the points'
    
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
        # how many plots
        nplots = 0
        for k,v in data.items():
            if 'plot' in k:
                nplots += 1
        if nplots == 1: # single plot
            q = 'What are the matplotlib '+big_tag+' in this figure? '
        else:
            q = 'What are the matplotlib '+big_tag+' for the plot in the ' + plot_index_to_words(data['figure']['plot indexes'][plot_num]) + ' panel? '     

    q += 'You are a helpful assistant, please format the output as a json as {"'+big_tag_single+'":'+value_empty+'}, '#where each element of the list refers to the '
    #q += ' '+big_tag_short+' of each line in the figure panel in the form a matplotlib '+big_tag_short+' type. '
    q += value_format
    q += extra_if_empty
    if use_list:
        adder = adder.split(')')[0] + ' + list)'
        q += 'Please choose each '+big_tag_short+' from the following list: ['
        for pt in line_list:
            q += str(pt) + ', '
        q = q[:-2] # take off the last bit
        q += '].'

    ans = {big_tag + ' ' + adder:{'plot'+str(plot_num):answer}} 
    a = {big_tag + ' ' + adder:answer} 
    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', ans)
    if return_qa: 
        if big_tag + ' ' + adder not in qa_pairs['Level 1']['Plot-level questions']:
            qa_pairs['Level 1']['Plot-level questions'][big_tag + ' ' + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a}}
        else:
            qa_pairs['Level 1']['Plot-level questions'][big_tag + ' ' + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a}
        return qa_pairs


def q_datapoints_scatters(data, qa_pairs, plot_num = 0, return_qa=True, use_words=True, verbose=True, use_colors = False):
    """
    use_words : set to True to translate row, column to words; False will use C-ordering indexing
    use_list : use a list of "property" options for prompting
    """
    big_tag = 'data values'
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
        q = 'What are the data values in this figure? '
    else:
        if not use_words:
            q += 'What are the data values for the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '? '
        else:
            q = 'What are the data values for the plot in the ' + plot_index_to_words(data['figure']['plot indexes'][plot_num]) + ' panel? '     

    if not use_colors:
        q += 'You are a helpful assistant, please format the output as a json as {"x":[], "y":[]} where the values of "x" and "y" are the '
    else:
        q += 'You are a helpful assistant, please format the output as a json as {"x":[], "y":[], "colors":[]} where the values of "x", "y", and "colors" are the '
    q += 'data values used to create the plot and are in the format of a list of floats.  '
    
    # get answer list
    la = {"x":data['plot'+str(plot_num)]['data']['xs'], "y":data['plot'+str(plot_num)]['data']['ys']}
    if use_colors:
        la['colors'] = data['plot'+str(plot_num)]['data']['colors']
        
    ans = {big_tag + ' ' + adder:{'plot'+str(plot_num):la}} 
    a = {big_tag + ' ' + adder:la}
    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', ans)
    if return_qa: 
        if big_tag + ' ' + adder not in qa_pairs['Level 1']['Plot-level questions']:
            qa_pairs['Level 1']['Plot-level questions'][big_tag + ' ' + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a}}
        else:
            qa_pairs['Level 1']['Plot-level questions'][big_tag + ' ' + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a}
        return qa_pairs



def q_stats_scatters(data, qa_pairs, stat = {'minimum':np.min}, plot_num = 0, 
                     return_qa=True, use_words=True, verbose=True, use_colors=False):
    """
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

    if not use_colors:
        q += 'You are a helpful assistant, please format the output as a json as {"'+big_tag+' x":"", "'+big_tag+' y":""} where the '+big_tag+' values of "x" and "y" are the '
    else:
        q += 'You are a helpful assistant, please format the output as a json as {"'+big_tag+' x":"", "'+big_tag+' y":"", "'+big_tag+' colors":""} where the '+big_tag+' values of "x", "y" and "colors" are the '
    q += 'data values used to create the plot in the format of floats.  '
    #q += 'If there is one single array for x data values for all lines on the plot, "'+big_tag+' x" will be a single item list. '
    #q += 'If there is one single array for y data values for all lines on the plot, "'+big_tag+' y" will be a single item list. '
    #q += 'If there is one single array for y data values for all lines on the plot, "minimum y" will be a single list, otherwise it will be a list of lists. '
    
    # get answer list
    f = list(stat.values())[0] # what stastical function
    xs = data['plot'+str(plot_num)]['data']['xs']
    ys = data['plot'+str(plot_num)]['data']['ys']
    la = {big_tag + " x":f(xs), big_tag + " y":f(ys)}
    if use_colors:
        cs = data['plot'+str(plot_num)]['data']['colors']
        la[big_tag+' colors'] = f(cs)
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


def q_errorbars_size_scatters(data, qa_pairs, axis = 'x', plot_num = 0, return_qa=True, use_words=True, verbose=True, 
                             use_avg = False):
    """
    use_words : set to True to translate row, column to words; False will use C-ordering indexing
    use_avg : calculate the average size of the errorbars
    """
    
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
        q = 'What is the approximate size of the error bars along the ' + axis + '-axis in this figure? '
    else:
        if not use_words:
            q += 'What is the approximate size of the error bars along the ' + axis + '-axis for the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '? '
        else:
            q = 'What is the approximate size of the error bars along the ' + axis + '-axis for the plot in the ' + plot_index_to_words(data['figure']['plot indexes'][plot_num]) + ' panel? '     

    if not use_avg:
        q += 'You are a helpful assistant, please format the output as a json as {"'+axis+'-axis error size":[]} where each element of the list corresponds to a single point in the figure. '
    else:
        q += 'You are a helpful assistant, please format the output as a json as {"'+axis+'-axis error size":""} where the value is a float and the average size of the error bars along this axis. '
        q += 'The average size of the error bar can be calculated as the average length of the bar divided by the difference between the maximum and minimum data values along the '+axis+'-axis.'
        
    q += 'For reference, errorbars are shown with the matplotlib "ax.errorbar" function. '

    # get answer
    if axis+'errs' in data['plot' + str(plot_num)]['data'].keys(): # just a double check
        if use_avg:
            v = data['plot'+str(plot_num)]['data'][axis+'s']
            errs = np.mean(np.abs(data['plot'+str(plot_num)]['data'][axis+'errs']))/np.abs((np.max(v)-np.min(v)))
            a = {axis+'-axis errors size' + ' ' + adder:errs}
        else:
            # not-averaged tag
            adder = adder.split(')')[0] + ' + non-averaged)'
            a = {axis+'-axis errors size' + ' ' + adder:data['plot'+str(plot_num)]['data'][axis+'errs']}
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


def q_relationship_scatters(data, qa_pairs, plot_num = 0, return_qa=True, use_words=True, use_list=True, 
                        line_list = ['random','linear','gaussian mixture model'], verbose=True):
    """
    use_words : set to True to translate row, column to words; False will use C-ordering indexing
    use_list : give a list of possible distributions
    use_nplots : give the number of lines in the prompt
    """
    
    # how many plots
    big_tag_short = 'relationship'
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
        q = 'What is the functional relationship between the x and y values in this figure? '
    else:
        if not use_words:
            q += 'What is the functional relationship between the x and y values for the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '? '
        else:
            q = 'What is the functional relationship between the x and y values for the plot in the ' + plot_index_to_words(data['figure']['plot indexes'][plot_num]) + ' panel? '     

    q += 'You are a helpful assistant, please format the output as a json as {"relationship":""} the value is a string. '

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



def q_relationship_colors_scatters(data, qa_pairs, plot_num = 0, return_qa=True, use_words=True, use_list=True, 
                        line_list = ['no relationship', 'random','linear','gaussian mixture model'], 
                                   verbose=True):
    """
    use_words : set to True to translate row, column to words; False will use C-ordering indexing
    use_list : give a list of possible distributions
    use_nplots : give the number of lines in the prompt
    """
    
    # how many plots
    big_tag_short = 'color relationship'
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
        q = 'What is the functional relationship between the x and y values in this figure and the colors of the scatter plots? '
    else:
        if not use_words:
            q += 'What is the functional relationship between the x and y values for the figure panel on row number ' + \
              str(nrow) + ' and column number ' + str(ncol) + ' and the colors of the scatter plots? '
        else:
            q = 'What is the functional relationship between the x and y values for the plot in the ' + \
              plot_index_to_words(data['figure']['plot indexes'][plot_num]) + ' panel and the colors of the scatter plots? '     

    q += 'You are a helpful assistant, please format the output as a json as {"color relationship":""} the value is a string. '

    if use_list:
        adder = adder.split(')')[0] + ' + list)'
        q += 'Please choose each '+big_tag_short+' from the following list: ['
        for pt in line_list:
            q += str(pt) + ', '
        q = q[:-2] # take off the last bit
        q += '].'

    # answer
    dist = ''
    if 'data params' not in data['plot'+str(plot_num)]['data']:
        if verbose: print('no data params!')
        dist = 'no relationship'
        #import sys; sys.exit()
    elif 'colors' not in data['plot'+str(plot_num)]['data']['data params']:
        print('no color specific tag in data_params!')
        import sys; sys.exit()

    if dist == '':
        dist = data['plot'+str(plot_num)]['data']['data params']['colors']['type']
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


def q_linear_equation_scatters(data, qa_pairs, plot_num = 0, return_qa=True, use_words=True, verbose=True,
                              parameter_tag = 'points'):
    """
    use_words : set to True to translate row, column to words; False will use C-ordering indexing
    use_nlines : give the number of lines in the prompt
    paramter_tag : for x/y relationship = 'points', for color = 'colors'
    """

    big_tag_short = 'linear parameters'
    # is this a linear relationship? if not this is an error
    hasLine = False
    if parameter_tag == 'points':
        if data['plot'+str(plot_num)]['distribution'] != 'linear': # not a linear relationship
            if verbose:
                print('Not a linear relationship!')
            if return_qa:
                return qa_pairs
            else:
                return None
    elif parameter_tag == 'colors':
        if 'data params' in data['plot'+str(plot_num)]:
            if 'colors' in data['plot'+str(plot_num)]['data params']:
                if data['plot'+str(plot_num)]['data params']['colors']['type'] == 'linear':
                    hasLine = True
        if not hasLine:
            if verbose:
                print('Not a linear relationship in color!')
            if return_qa:
                return qa_pairs
            else:
                return None

    if data['plot'+str(plot_num)]['distribution'] == 'linear' and parameter_tag != 'colors':
        hasLine = True

    if not hasLine:
        print("This is not a linear relationship, can't ask any questions!'")
        import sys; sys.exit()
        
    adder = parameter_tag + ' '
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
        if parameter_tag == 'points':
            q = 'What are the parameters for the linear relationship between the x and y values in this figure? '
        elif parameter_tag == 'colors':
            q = 'What are the parameters for the linear relationship between the x and y values in this figure and the colors of the scatter points? '            
    else:
        if not use_words:
            q += 'What are the parameters for the linear relationship between for the figure panel on row number ' + \
              str(nrow) + ' and column number ' + str(ncol) + '? '
        else:
            q = 'What are the parameters for the linear relationship between the x and y values for the plot in the ' + \
              plot_index_to_words(data['figure']['plot indexes'][plot_num]) + ' panel? '     

    if parameter_tag == 'colors':
        q += 'You are a helpful assistant, please format the output as a json as {"'+big_tag_short+'":{"m1":"","a1":"","m2":"","a2":"","noise level":""}}. '
        q += 'Here, "m1" is the slope of the relationship and "a1" is the intercept in the "x" axis direction and both should be floats. '
        q += 'Additionally, "m2" is the slope of the relationship and "a2" is the intercept in the "y" axis direction and both should be floats. '
    else:
        q += 'You are a helpful assistant, please format the output as a json as {"'+big_tag_short+'":{"m":"","a":"","noise level":""}}. '
        q += 'Here, "m" is the slope of the relationship and "a" is the intercept and both should be floats. '
        
    q += 'The "noise level" parameter should be the relative amount of noise added to the '
    q += 'linear function for the relationship and should be a float between 0 and 1. '
    
    # answer
    # la = []
    # for k,v in data['plot'+str(plot_num)]['data']['data params']['points'].items():
    #     la.append(v)
    if parameter_tag == 'npoints':
        la = data['plot'+str(plot_num)]['data']['data params']['points']
    else:
        la = data['plot'+str(plot_num)]['data']['data params']['colors']

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



def q_gmm_equation_scatters(data, qa_pairs, plot_num = 0, return_qa=True, use_words=True, verbose=True,
                              parameter_tag = 'points'):
    """
    use_words : set to True to translate row, column to words; False will use C-ordering indexing
    use_nlines : give the number of lines in the prompt
    paramter_tag : for x/y relationship = 'points', for color = 'colors'
    """

    big_tag_short = 'gmm parameters'
    # is this a linear relationship? if not this is an error
    hasLine = False
    if parameter_tag == 'points':
        if data['plot'+str(plot_num)]['distribution'] != 'gmm': # not a linear relationship
            if verbose:
                print('Not a gmm relationship!')
            if return_qa:
                return qa_pairs
            else:
                return None
    elif parameter_tag == 'colors':
        if 'data params' in data['plot'+str(plot_num)]:
            if 'colors' in data['plot'+str(plot_num)]['data params']:
                if data['plot'+str(plot_num)]['data params']['colors']['type'] == 'gmm':
                    hasLine = True
        if not hasLine:
            if verbose:
                print('Not a gmm relationship in color!')
            if return_qa:
                return qa_pairs
            else:
                return None

    if data['plot'+str(plot_num)]['distribution'] == 'gmm' and parameter_tag != 'colors':
        hasLine = True

    if not hasLine:
        print("This is not a gmm relationship, can't ask any questions!'")
        import sys; sys.exit()
        
    adder = parameter_tag + ' '
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
        if parameter_tag == 'points':
            q = 'What are the parameters for the gaussian mixture model relationship between the x and y values in this figure? '
        elif parameter_tag == 'colors':
            q = 'What are the parameters for the gaussian mixture model relationship between the x and y values in this figure and the colors of the scatter points? '            
    else:
        if not use_words:
            q += 'What are the parameters for the gaussian mixture model relationship between for the figure panel on row number ' + \
              str(nrow) + ' and column number ' + str(ncol) + '? '
        else:
            q = 'What are the parameters for the gaussian mixture model relationship between the x and y values for the plot in the ' + \
              plot_index_to_words(data['figure']['plot indexes'][plot_num]) + ' panel? '  

    # 'nsamples', 'nclusters', 'centers', 'cluster_std', 'noise level'
    if parameter_tag == 'points':
        q += 'You are a helpful assistant, please format the output as a json as {"'+big_tag_short+\
          '":{"nsamples":"","nclusters":"","centers":"","cluster_std":"","noise level":""}}. '
        #q += 'where each dictionary element of the list corresponds to a single line in the figure. '
        q += 'Each "nsamples" is the number of samples in the distribution and should be an integer. '
        q += 'The "nclusters" parameter should be the number of clusters in the model and should be an integer. '
        q += 'The "centers" parameter should be the position of each cluster and should be a list of (x,y) points where "x" and "y" are both floats. '
        q += 'The "cluster_std" parameter should be the standard deviation of each cluster and should be a list of (x,y) points where "x" and "y" are both floats. '
        #q += 'The "cluster_std" parameter should be the standard deviation for each cluster on the x-axis and should be a float. '
        q += 'The "noise level" parameter should be the relative amount of noise added to the model and should be a float between 0 and 1. '
    else: # colors
        q += 'You are a helpful assistant, please format the output as a json as {"'+big_tag_short+'":""} '
        q += 'where the value is either "random" for randomly colored points or "by cluster" if the points ' \
           + 'from each gaussian mixure model cluster are colored by cluster membership. '
    
    # answer
    # la = []
    # for k,v in data['plot'+str(plot_num)]['data']['data params']['points'].items():
    #     la.append(v)
    if parameter_tag == 'npoints':
        la = data['plot'+str(plot_num)]['data']['data params']['points']
    else:
        la = data['plot'+str(plot_num)]['data']['data params']['colors']

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