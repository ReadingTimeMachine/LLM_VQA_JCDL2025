import numpy as np

from utils.plot_qa_utils import plot_index_to_words

# this version tries to give column and row numbers
def q_nlines_plot_plotnums(data, qa_pairs, plot_num = 0, return_qa=True, verbose=True):
    # rows columns
    nrow = data['figure']['plot indexes'][plot_num][0]
    ncol = data['figure']['plot indexes'][plot_num][1]
    q = 'The following question refers to the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '. '
    q += 'If there are multiple plots the panels will be in row-major (C-style) order, with the numbering starting at (0,0) in the upper left panel. '
    q += 'If there is one plot, then this row and column refers to the single plot. '
    q += 'How many different lines are there for the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '? '
    q += 'You are a helpful assistant, please format the output as a json as {"nlines":""} for this figure panel, where the "nlines" value should be an integer.'
    # get answer list
    la = []
    for k,v in data.items():
        if 'plot' + str(plot_num) in k: # is a plot
            #nlines = len(v['data']['data params'].keys())
            nlines = len(v['data']['ys'])
            la.append(nlines)
    a = {"nlines (plot numbers)":{'plot'+str(plot_num):la[0]}} # the 0 is a bit hacky
    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', a)
    if return_qa: 
        if 'nlines (plot numbers)' not in qa_pairs['Level 1']['Plot-level questions']:
            qa_pairs['Level 1']['Plot-level questions']['nlines (plot numbers)'] = {'plot'+str(plot_num):{'Q':q, 'A':a}}
        else:
            qa_pairs['Level 1']['Plot-level questions']['nlines (plot numbers)']['plot'+str(plot_num)] = {'Q':q, 'A':a}
        return qa_pairs


# this version tries to use words
def q_nlines_plot(data, qa_pairs, plot_num = 0, return_qa=True, verbose=True):
    # how many plots
    nplots = 0
    for k,v in data.items():
        if 'plot' in k:
            nplots += 1
    if nplots == 1: # single plot
        q = 'How many different lines are there on this figure? '
        q += 'You are a helpful assistant, please format the output as a json as {"nlines":""} for this figure panel, where the "nlines" value should be an integer.'
    else:
        q = 'How many different lines are there for the plot in the ' + plot_index_to_words(data['figure']['plot indexes'][plot_num]) + ' panel? '     
        q += 'You are a helpful assistant, please format the output as a json as {"nlines":""} for this figure panel, where the "nlines" value should be an integer.'
    # get answer list
    la = []
    for k,v in data.items():
        if 'plot' + str(plot_num) in k: # is a plot
            #nlines = len(v['data']['data params'].keys())
            nlines = len(v['data']['ys'])
            la.append(nlines)
    a = {"nlines (words)":{'plot'+str(plot_num):la[0]}} # the 0 is a bit hacky
    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', a)
    if return_qa: 
        if 'nlines (words)' not in qa_pairs['Level 1']['Plot-level questions']:
            qa_pairs['Level 1']['Plot-level questions']['nlines (words)'] = {'plot'+str(plot_num):{'Q':q, 'A':a}}
        else:
            qa_pairs['Level 1']['Plot-level questions']['nlines (words)']['plot'+str(plot_num)] = {'Q':q, 'A':a}
        return qa_pairs

def q_colors_lines(data, qa_pairs, plot_num = 0, return_qa=True, use_words=True, verbose=True):
    """
    use_words : set to True to translate row, column to words; False will use C-ordering indexing
    """
    big_tag = 'line colors'
    if not use_words:
        adder = '(plot numbers)'
        # rows columns
        nrow = data['figure']['plot indexes'][plot_num][0]
        ncol = data['figure']['plot indexes'][plot_num][1]
        q = 'The following question refers to the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '. '
        q += 'If there are multiple plots the panels will be in row-major (C-style) order, with the numbering starting at (0,0) in the upper left panel. '
        q += 'If there is one plot, then this row and column refers to the single plot. '
        q += 'What are the colors of the lines for the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '? '
    else: 
        adder = '(words)'
        # how many plots
        nplots = 0
        for k,v in data.items():
            if 'plot' in k:
                nplots += 1
        if nplots == 1: # single plot
            q = 'What are the colors of the lines in this figure? '
        else:
            q = 'What are the colors of the lines for the plot in the ' + plot_index_to_words(data['figure']['plot indexes'][plot_num]) + ' panel? '     

    q += 'You are a helpful assistant, please format the output as a json as {"line colors":[]}, where each element of the list refers to the '
    q += ' color of each line in the figure panel in the form of an RGBA list with values between 0 and 1. '
    # get answer list
    la = []
    for k,v in data.items():
        if 'plot' + str(plot_num) == k: # is a plot
            #if 'plot params' in v['data from plot']:
            for c in v['data from plot']['plot params']['colors']:
                la.append(c[0]) # the "0" here is because, in theory, the line could be made up of more than one color
            #else: # no colors
            #    la.append("")

    a = {big_tag + ' ' + adder:{'plot'+str(plot_num):la}} 
    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', a)
    if return_qa: 
        if big_tag + ' ' + adder not in qa_pairs['Level 1']['Plot-level questions']:
            qa_pairs['Level 1']['Plot-level questions'][big_tag + ' ' + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a}}
        else:
            qa_pairs['Level 1']['Plot-level questions'][big_tag + ' ' + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a}
        return qa_pairs


def q_linestyles_lines(data, qa_pairs, plot_num = 0, return_qa=True, use_words=True, use_list=False,
                      linestyle_list = [], verbose=True):
    """
    use_words : set to True to translate row, column to words; False will use C-ordering indexing
    use_list : use a list of linestyle options for prompting
    """
    
    big_tag = 'line styles'
    if not use_words:
        adder = '(plot numbers)'
        # rows columns
        nrow = data['figure']['plot indexes'][plot_num][0]
        ncol = data['figure']['plot indexes'][plot_num][1]
        q = 'The following question refers to the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '. '
        q += 'If there are multiple plots the panels will be in row-major (C-style) order, with the numbering starting at (0,0) in the upper left panel. '
        q += 'If there is one plot, then this row and column refers to the single plot. '
        q += 'What are the '+big_tag+' of the lines for the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '? '
    else: 
        adder = '(words)'
        # how many plots
        nplots = 0
        for k,v in data.items():
            if 'plot' in k:
                nplots += 1
        if nplots == 1: # single plot
            q = 'What are the matplotlib linestyles in this figure? '
        else:
            q = 'What are the matplotlib linestyles for the plot in the ' + plot_index_to_words(data['figure']['plot indexes'][plot_num]) + ' panel? '     

    q += 'You are a helpful assistant, please format the output as a json as {"line styles":[]}, where each element of the list refers to the '
    q += ' linestyle of each line in the figure panel in the form a matplotlib linestyle type. '
    if use_list:
        adder = adder.split(')')[0] + ' + list)'
        q += 'Please choose each linestyle from the following list: ['
        for pt in linestyle_list:
            q += pt + ', '
        q = q[:-2] # take off the last bit
        q += '].'

    # get answer list
    la = []
    for k,v in data.items():
        if 'plot' + str(plot_num) == k: # is a plot
            for c in v['data from plot']['plot params']['linestyles']:
                la.append(c) 

    a = {big_tag + ' ' + adder:{'plot'+str(plot_num):la}} 
    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', a)
    if return_qa: 
        if big_tag + ' ' + adder not in qa_pairs['Level 1']['Plot-level questions']:
            qa_pairs['Level 1']['Plot-level questions'][big_tag + ' ' + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a}}
        else:
            qa_pairs['Level 1']['Plot-level questions'][big_tag + ' ' + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a}
        return qa_pairs


def q_linethickness_lines(data, qa_pairs, plot_num = 0, return_qa=True, use_words=True, use_list=False,
                      line_list = [], verbose=True):
    """
    use_words : set to True to translate row, column to words; False will use C-ordering indexing
    use_list : use a list of "property" options for prompting
    """
    
    big_tag = 'line thicknesses'
    big_tag_single = 'line thickness'
    big_tag_short = 'thickness'
    plot_param_tag = 'linethick'
    if not use_words:
        adder = '(plot numbers)'
        # rows columns
        nrow = data['figure']['plot indexes'][plot_num][0]
        ncol = data['figure']['plot indexes'][plot_num][1]
        q = 'The following question refers to the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '. '
        q += 'If there are multiple plots the panels will be in row-major (C-style) order, with the numbering starting at (0,0) in the upper left panel. '
        q += 'If there is one plot, then this row and column refers to the single plot. '
        q += 'What are the '+big_tag+' of the lines for the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '? '
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

    q += 'You are a helpful assistant, please format the output as a json as {"'+big_tag_single+'":[]}, where each element of the list refers to the '
    q += ' '+big_tag_short+' of each line in the figure panel in the form a matplotlib '+big_tag_short+' type. '
    if use_list:
        adder = adder.split(')')[0] + ' + list)'
        q += 'Please choose each '+big_tag_short+' from the following list: ['
        for pt in line_list:
            q += pt + ', '
        q = q[:-2] # take off the last bit
        q += '].'

    # get answer list
    la = []
    for k,v in data.items():
        if 'plot' + str(plot_num) == k: # is a plot
            for c in v['data from plot']['plot params'][plot_param_tag]:
                la.append(c) 

    a = {big_tag + ' ' + adder:{'plot'+str(plot_num):la}} 
    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', a)
    if return_qa: 
        if big_tag + ' ' + adder not in qa_pairs['Level 1']['Plot-level questions']:
            qa_pairs['Level 1']['Plot-level questions'][big_tag + ' ' + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a}}
        else:
            qa_pairs['Level 1']['Plot-level questions'][big_tag + ' ' + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a}
        return qa_pairs


def q_linemarkers_lines(data, qa_pairs, plot_num = 0, return_qa=True, use_words=True, use_list=False,
                      line_list = [], verbose=True):
    """
    use_words : set to True to translate row, column to words; False will use C-ordering indexing
    use_list : use a list of "property" options for prompting
    """
    
    big_tag = 'line markers'
    big_tag_single = 'line marker'
    big_tag_short = 'marker'
    plot_param_tag = 'markers'
    extra_if_empty = 'If there are no markers for a specific line, please indicate this with an empty "" for this element of the list.'
    if not use_words:
        adder = '(plot numbers)'
        # rows columns
        nrow = data['figure']['plot indexes'][plot_num][0]
        ncol = data['figure']['plot indexes'][plot_num][1]
        q = 'The following question refers to the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '. '
        q += 'If there are multiple plots the panels will be in row-major (C-style) order, with the numbering starting at (0,0) in the upper left panel. '
        q += 'If there is one plot, then this row and column refers to the single plot. '
        q += 'What are the '+big_tag+' of the lines for the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '? '
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

    q += 'You are a helpful assistant, please format the output as a json as {"'+big_tag_single+'":[]}, where each element of the list refers to the '
    q += ' '+big_tag_short+' of each line in the figure panel in the form a matplotlib '+big_tag_short+' type. '
    q += extra_if_empty
    if use_list:
        adder = adder.split(')')[0] + ' + list)'
        q += 'Please choose each '+big_tag_short+' from the following list: ['
        for pt in line_list:
            q += pt + ', '
        q = q[:-2] # take off the last bit
        q += '].'

    # get answer list
    la = []
    for k,v in data.items():
        if 'plot' + str(plot_num) == k: # is a plot
            for c in v['data from plot']['plot params'][plot_param_tag]:
                la.append(c) 

    a = {big_tag + ' ' + adder:{'plot'+str(plot_num):la}} 
    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', a)
    if return_qa: 
        if big_tag + ' ' + adder not in qa_pairs['Level 1']['Plot-level questions']:
            qa_pairs['Level 1']['Plot-level questions'][big_tag + ' ' + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a}}
        else:
            qa_pairs['Level 1']['Plot-level questions'][big_tag + ' ' + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a}
        return qa_pairs


def q_linemarkersize_lines(data, qa_pairs, plot_num = 0, return_qa=True, use_words=True, use_list=False,
                      line_list = [], verbose=True):
    """
    use_words : set to True to translate row, column to words; False will use C-ordering indexing
    use_list : use a list of "property" options for prompting
    """
    
    big_tag = 'line marker sizes'
    big_tag_single = 'line marker size'
    big_tag_short = 'marker size'
    plot_param_tag = 'marker size' # tag in the plot params
    extra_if_empty = 'If there are no markers for a specific line, please indicate this with an empty "" for this element of the list.'
    if not use_words:
        adder = '(plot numbers)'
        # rows columns
        nrow = data['figure']['plot indexes'][plot_num][0]
        ncol = data['figure']['plot indexes'][plot_num][1]
        q = 'The following question refers to the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '. '
        q += 'If there are multiple plots the panels will be in row-major (C-style) order, with the numbering starting at (0,0) in the upper left panel. '
        q += 'If there is one plot, then this row and column refers to the single plot. '
        q += 'What are the '+big_tag+' of the lines for the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '? '
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

    q += 'You are a helpful assistant, please format the output as a json as {"'+big_tag_single+'":[]}, where each element of the list refers to the '
    q += ' '+big_tag_short+' of each line in the figure panel in the form a matplotlib '+big_tag_short+' type. '
    q += extra_if_empty
    if use_list:
        adder = adder.split(')')[0] + ' + list)'
        q += 'Please choose each '+big_tag_short+' from the following list: ['
        for pt in line_list:
            q += pt + ', '
        q = q[:-2] # take off the last bit
        q += '].'

    # get answer list
    la = []
    for k,v in data.items():
        if 'plot' + str(plot_num) == k: # is a plot
            for c in v['data from plot']['plot params'][plot_param_tag]:
                la.append(c) 

    a = {big_tag + ' ' + adder:{'plot'+str(plot_num):la}} 
    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', a)
    if return_qa: 
        if big_tag + ' ' + adder not in qa_pairs['Level 1']['Plot-level questions']:
            qa_pairs['Level 1']['Plot-level questions'][big_tag + ' ' + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a}}
        else:
            qa_pairs['Level 1']['Plot-level questions'][big_tag + ' ' + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a}
        return qa_pairs



def q_datapoints_lines(data, qa_pairs, plot_num = 0, return_qa=True, use_words=True, 
                      line_list = [], verbose=True):
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

    q += 'You are a helpful assistant, please format the output as a json as {"x":[], "y":[]} where the values of "x" and "y" are the '
    q += 'data values used to create the plot.  '
    q += 'If there is one single array for x data values for all lines on the plot, "x" will be a single list, otherwise it will be a list of lists. '
    q += 'If there is one single array for y data values for all lines on the plot, "y" will be a single list, otherwise it will be a list of lists. '
    
    # get answer list
    la = {"x":data['plot'+str(plot_num)]['data']['xs'], "y":data['plot'+str(plot_num)]['data']['ys']}
    a = {big_tag + ' ' + adder:{'plot'+str(plot_num):la}} 
    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', a)
    if return_qa: 
        if big_tag + ' ' + adder not in qa_pairs['Level 1']['Plot-level questions']:
            qa_pairs['Level 1']['Plot-level questions'][big_tag + ' ' + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a}}
        else:
            qa_pairs['Level 1']['Plot-level questions'][big_tag + ' ' + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a}
        return qa_pairs



def q_stats_lines(data, qa_pairs, stat = {'minimum':np.min}, plot_num = 0, return_qa=True, use_words=True, verbose=True):
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

    q += 'You are a helpful assistant, please format the output as a json as {"'+big_tag+' x":[], "'+big_tag+' y":[]} where the '+big_tag+' values of "x" and "y" are the '
    q += 'data values used to create the plot.  '
    q += 'If there is one single array for x data values for all lines on the plot, "'+big_tag+' x" will be a single item list. '
    q += 'If there is one single array for y data values for all lines on the plot, "'+big_tag+' y" will be a single item list. '
    #q += 'If there is one single array for y data values for all lines on the plot, "minimum y" will be a single list, otherwise it will be a list of lists. '
    
    # get answer list
    f = list(stat.values())[0]
    xs = []; ys = []
    for x in data['plot'+str(plot_num)]['data']['xs']:
        xs.append(f(x))
    for y in data['plot'+str(plot_num)]['data']['ys']:
        ys.append(f(y))
    la = {big_tag + " x":xs, big_tag + " y":ys}
    a = {big_tag + ' ' + adder:{'plot'+str(plot_num):la}} 
    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', a)
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


def q_errorbars_size_lines(data, qa_pairs, axis = 'x', plot_num = 0, return_qa=True, use_words=True, verbose=True):
    """
    use_words : set to True to translate row, column to words; False will use C-ordering indexing
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

    q += 'You are a helpful assistant, please format the output as a json as {"'+axis+'-axis error size":[]} where each element of the list corresponds to a single line in the figure. '
    q += 'The size of the error bar can be calculated as the average length of the bar divided by the difference between the maximum and minimum data values along the '+axis+'-axis.'

    # get answer
    if axis+'errs' in data['plot' + str(plot_num)]['data'].keys(): # just a double check
        errs = []
        for ev,v in zip(data['plot'+str(plot_num)]['data'][axis+'errs'],data['plot'+str(plot_num)]['data'][axis+'s']): # all lines
            errs.append(np.mean(np.abs(ev))/(np.max(v)-np.min(v)))
        a = {axis+'-axis errors size' + ' ' + adder:errs}
    else:
        print('No axis error bars!!')
        import sys; sys.exit()

    #a = aout.values[0]

    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', a)
    if return_qa: 
        if axis+'-axis error size' + ' ' + adder not in qa_pairs['Level 2']['Plot-level questions']:
            qa_pairs['Level 2']['Plot-level questions'][axis+'-axis error size' + ' ' + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a}}
        else:
            qa_pairs['Level 2']['Plot-level questions'][axis+'-axis error size' + ' ' + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a}
        return qa_pairs


########## L2/L3 #############

def q_relationship_lines(data, qa_pairs, plot_num = 0, return_qa=True, use_words=True, use_list=True, use_nlines = True,
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

    q += 'You are a helpful assistant, please format the output as a json as {"relationship":[]} where each element of the list corresponds to a single line in the figure. '
    if use_nlines:
        adder = adder.split(')')[0] + ' + nlines)'
        q += 'Please note that there are a total of ' + str(int(len(data['plot' + str(plot_num)]['data']['ys']))) + ' lines in this plot, so the list should have a '
        q += 'total of ' +str(int(len(data['plot' + str(plot_num)]['data']['ys'])))+ ' entries. '

    if use_list:
        adder = adder.split(')')[0] + ' + list)'
        q += 'Please choose each '+big_tag_short+' for each line from the following list for each line: ['
        for pt in line_list:
            q += pt + ', '
        q = q[:-2] # take off the last bit
        q += '].'

    # answer
    la = []
    for i in range(len(data['plot' + str(plot_num)]['data']['ys'])):
        dist = data['plot'+str(plot_num)]['distribution']
        # to match
        if dist == 'gmm': dist = 'gaussian mixture model'
        la.append(dist) # note: assumes same for all!
    #a = la
    a = {big_tag_short + ' ' + adder:la}

    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', a)
    if return_qa: 
        if big_tag_short + ' ' + adder not in qa_pairs['Level 3']['Plot-level questions']:
            #print('yes', big_tag_short + ' ' + adder)
            qa_pairs['Level 3']['Plot-level questions'][big_tag_short + ' ' + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a, 
                                                                                                              'note':'this currently assumes all elements on a single plot have the same relationship type'}}
        else:
            qa_pairs['Level 3']['Plot-level questions'][big_tag_short + ' ' + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a, 
                                                                                                              'note':'this currently assumes all elements on a single plot have the same relationship type'}
        return qa_pairs



def q_linear_equation_lines(data, qa_pairs, plot_num = 0, return_qa=True, use_words=True, use_nlines = True, verbose=True):
    """
    use_words : set to True to translate row, column to words; False will use C-ordering indexing
    use_nlines : give the number of lines in the prompt
    """

    big_tag_short = 'linear parameters'
    # is this a linear relationship? if not this is an error
    hasLine = False
    #if 'data params' not in data['plot'+str(plot_num)]['data']: # not a linear relationship
    if data['plot'+str(plot_num)]['distribution'] != 'linear': # not a linear relationship
        if verbose:
            print('Not a linear relationship!')
            if return_qa:
                return qa_pairs
            else:
                return None
    #elif data['distribution'] 

    #for k,v in data['plot'+str(plot_num)]['data']['data params']['points'].items():
    for k,v in data['plot'+str(plot_num)]['data']['data params'].items():
        if 'line' in k:
            hasLine = True
    if not hasLine:
        print("This is not a linear relationship, can't ask any questions!'")
        import sys; sys.exit()
    
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
        q = 'What are the parameters for the linear relationship between the x and y values in this figure? '
    else:
        if not use_words:
            q += 'What are the parameters for the linear relationship between for the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '? '
        else:
            q = 'What are the parameters for the linear relationship between the x and y values for the plot in the ' + plot_index_to_words(data['figure']['plot indexes'][plot_num]) + ' panel? '     

    q += 'You are a helpful assistant, please format the output as a json as {"'+big_tag_short+'":[{"m":"","a":"","noise level":""}]} '
    q += 'where each dictionary element of the list corresponds to a single line in the figure. '
    q += 'Each "m" is the slope for each line and "a" is the intercept for each line and should be a float. '
    q += 'The "noise level" parameter should be the relative amount of noise added to each linear function for each line and should be a float between 0 and 1. '
    
    if use_nlines:
        adder = adder.split(')')[0] + ' + nlines)'
        q += 'Please note that there are a total of ' + str(int(len(data['plot' + str(plot_num)]['data']['ys']))) + ' lines in this plot, so the list should have a '
        q += 'total of ' +str(int(len(data['plot' + str(plot_num)]['data']['ys'])))+ ' entries. '

    # answer
    la = []
    #for k,v in data['plot'+str(plot_num)]['data']['data params']['points'].items():
    for k,v in data['plot'+str(plot_num)]['data']['data params'].items():
        la.append(v)

    a = {big_tag_short + ' ' + adder:la}

    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', a)
    if return_qa: 
        if big_tag_short + ' ' + adder not in qa_pairs['Level 3']['Plot-level questions']:
            qa_pairs['Level 3']['Plot-level questions'][big_tag_short + ' ' + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a, 
                                                                                                              'note':'this currently assumes all elements on a single plot have the same relationship type'}}
        else:
            qa_pairs['Level 3']['Plot-level questions'][big_tag_short + ' ' + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a, 
                                                                                                              'note':'this currently assumes all elements on a single plot have the same relationship type'}
        return qa_pairs




def q_linear_gmm_lines(data, qa_pairs, plot_num = 0, return_qa=True, use_words=True, use_nlines = True, verbose=True):
    """
    use_words : set to True to translate row, column to words; False will use C-ordering indexing
    use_nlines : give the number of lines in the prompt
    """

    big_tag_short = 'gmm parameters'
    # is this a linear relationship? if not this is an error
    hasLine = False
    if data['plot'+str(plot_num)]['distribution'] != 'gmm': # not a right relationship
        if verbose:
            print('Not a gmm relationship!')
            if return_qa:
                return qa_pairs
            else:
                return None
                
    for k,v in data['plot'+str(plot_num)]['data']['data params'].items():
        if 'line' in k:
            hasLine = True
    if not hasLine:
        print("This is not a gmm relationship, can't ask any questions!'")
        import sys; sys.exit()
    
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
        q = 'What are the parameters for the gaussian mixture model relationship between the x and y values in this figure? '
    else:
        if not use_words:
            q += 'What are the parameters for the gaussian mixture model relationship between for the figure panel on row number ' \
               + str(nrow) + ' and column number ' + str(ncol) + '? '
        else:
            q = 'What are the parameters for the gaussian mixture model relationship between the x and y values for the plot in the ' \
               + plot_index_to_words(data['figure']['plot indexes'][plot_num]) + ' panel? '     

    # 'nsamples', 'nclusters', 'centers', 'cluster_std', 'noise level'
    q += 'You are a helpful assistant, please format the output as a json as {"'+big_tag_short+\
      '":[{"nsamples":"","nclusters":"","centers":"","cluster_std":"","noise level":""}]} '
    q += 'where each dictionary element of the list corresponds to a single line in the figure. '
    q += 'Each "nsamples" is the number of samples in the distribution and should be an integer. '
    q += 'The "nclusters" parameter should be the number of clusters in the model and should be an integer. '
    q += 'The "centers" parameter should be the position of each cluster on the x-axis and should be a float. '
    q += 'The "cluster_std" parameter should be the standard deviation for each cluster on the x-axis and should be a float. '
    q += 'The "noise level" parameter should be the relative amount of noise added to the model for each line and should be a float between 0 and 1. '
    
    if use_nlines:
        adder = adder.split(')')[0] + ' + nlines)'
        q += 'Please note that there are a total of ' + str(int(len(data['plot' + str(plot_num)]['data']['ys']))) + ' lines in this plot, so the list should have a '
        q += 'total of ' +str(int(len(data['plot' + str(plot_num)]['data']['ys'])))+ ' entries. '

    # answer
    la = []
    for k,v in data['plot'+str(plot_num)]['data']['data params'].items():
        la.append(v)

    a = {big_tag_short + ' ' + adder:la}

    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', a)
    if return_qa: 
        if big_tag_short + ' ' + adder not in qa_pairs['Level 3']['Plot-level questions']:
            qa_pairs['Level 3']['Plot-level questions'][big_tag_short + ' ' + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a, 
                                                                                                              'note':'this currently assumes all elements on a single plot have the same relationship type'}}
        else:
            qa_pairs['Level 3']['Plot-level questions'][big_tag_short + ' ' + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a, 
                                                                                                              'note':'this currently assumes all elements on a single plot have the same relationship type'}
        return qa_pairs
