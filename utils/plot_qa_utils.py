import numpy as np

# stats = {'minimum':np.min, 'maximum':np.max, 'median':np.median, 'mean':np.mean}
# #stats.keys()
# {'minimum':np.min}.values()

############ FIGURES IN GENERAL ##############

# plot index to words
n_to_word = {0:'first', 1:'second', 2:'third', 3:'fourth', 4:'fifth', 5:'sixth', 6:'seventh', 7:'eigth', 8:'ninth', 9:'tenth'}
# ... for the plot in the [] panel...
def plot_index_to_words(pind):
    y = pind[0] # row
    x = pind[1] # column
    if y == 0 and x == 0:
        p = 'top-left' 
    elif y == 0:
        p = 'top row and ' + n_to_word[x] + ' from the left'
    elif x == 0:
        p = n_to_word[y] + ' row and left-most'
    else:
        p = n_to_word[x] + 'row and ' + n_to_word[y] + ' column'
    return p


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



# How many panels in the figure?
def q1(data, qa_pairs, return_qa=True, verbose=True):
    q1 = 'How many panels are in this figure?'
    q1 += ' You are a helpful assistant, please format the output as a json as {"nrows":"", "ncols":""} to store the number of rows and columns.'
    a1 = {"nrows":data['figure']['nrows'], "ncols":data['figure']['ncols']}
    if verbose:
        print('QUESTION:', q1)
        print('ANSWER:', a1)
        print('')
    # add to pairs
    if return_qa: 
        qa_pairs['Level 1']['Figure-level questions']['rows/columns'] = {'Q':q1, 'A':a1}
        return qa_pairs


# Plotting style?
def q2(data, qa_pairs, return_qa=True, verbose=True):
    q2 = 'Assuming this is a figure made with matplotlib in Python, what is the plot style used?  Examples of plotting styles are "classic" or "ggplot".'
    q2 += ' You are a helpful assistant, please format the output as a json as {"plot style":""} to store the matplotlib plotting style used in the figure.'
    a2 = {"plot style":data['figure']['plot style']}
    if verbose:
        print('QUESTION:', q2)
        print('ANSWER:', a2)
        print('')
    # add to pairs
    if return_qa: 
        qa_pairs['Level 1']['Figure-level questions']['plot style'] = {'Q':q2, 'A':a2}
        return qa_pairs


# Colormaps?
def q3(data, qa_pairs, return_qa=True, verbose=True):
    q2 = 'Assuming this is a figure made with matplotlib in Python, what is the colormap that was used?  Examples of matplotlib colormaps are "rainbow" or "Reds".'
    q2 += ' You are a helpful assistant, please format the output as a json as {"colormap":""} to store the matplotlib colormap used in the figure.'
    a2 = {"colormap":data['figure']['color map']}
    if verbose:
        print('QUESTION:', q2)
        print('ANSWER:', a2)
        print('')
    # add to pairs
    if return_qa: 
        qa_pairs['Level 1']['Figure-level questions']['colormap'] = {'Q':q2, 'A':a2, 
                                                                 'notes':"Some of the plot styles don't allow for updates to the colormap for REASONS, so just keep that in mind."}
        return qa_pairs


# aspect ratio
def q4(data, qa_pairs, return_qa=True, verbose=True):
    q1 = 'What is the aspect ratio of this figure?'
    q1 += ' You are a helpful assistant, please format the output as a json as {"aspect ratio":""} to store the aspect ratio of the plot.'
    a1 = {"aspect ratio":data['figure']['aspect ratio']}
    if verbose:
        print('QUESTION:', q1)
        print('ANSWER:', a1)
        print('')
    # add to pairs
    if return_qa: 
        qa_pairs['Level 1']['Figure-level questions']['aspect ratio'] = {'Q':q1, 'A':a1}
        return qa_pairs


# dpi
def dpi(data, qa_pairs, return_qa=True, verbose=True):
    q1 = 'What is the DPI (dots per square inch) of this figure?'
    q1 += ' You are a helpful assistant, please format the output as a json as {"dpi":""} to store the DPI of the plot.'
    a1 = {"dpi":data['figure']['dpi']}
    if verbose:
        print('QUESTION:', q1)
        print('ANSWER:', a1)
        print('')
    # add to pairs
    if return_qa: 
        qa_pairs['Level 1']['Figure-level questions']['dpi'] = {'Q':q1, 'A':a1}
        return qa_pairs



############ INDIVIDUAL PLOTS IN GENERAL ##############

def q_plot_titles(data, qa_pairs, return_qa=True, verbose=True):
    q = 'What are the titles for each figure panel?'
    q += ' You are a helpful assistant, please format the output as a json as {"titles":[]}, where the list is a list of strings of all of the titles.  '
    q += 'If there is a single plot, this should be one element in this list, and if there are multiple plots the list should be in row-major (C-style) order.'
    q += " If a plot does not have a title, then denote this by an empty string in the list.  Please format any formulas in the title in a Python LaTeX string (for example 'Light $\\\\alpha$'). "
    la = []
    for k,v in data.items():
        if 'plot' in k: # is a plot
            if 'title' not in v:
                la.append("")
            else:
                la.append(v['title']['words'])
    a = {"title":la}
    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', a)
    if return_qa: 
        qa_pairs['Level 1']['Plot-level questions']['titles'] = {'Q':q, 'A':a}
        return qa_pairs


def q_plot_xlabels(data, qa_pairs, return_qa=True, verbose=True):
    q = 'What are the x-axis titles for each figure panel?'
    q += ' You are a helpful assistant, please format the output as a json as {"xlabels":[]}, where the list is a list of strings of all of the x-axis titles.  '
    q += 'If there is a single plot, this should be one element in this list, and if there are multiple plots the list should be in row-major (C-style) order.'
    q += " If a plot does not have an x-axis title, then denote this by an empty string in the list.  "
    q += "Please format any formulas in the title in a Python LaTeX string (for example 'Light $\\\\alpha$'). "
    la = []
    for k,v in data.items():
        if 'plot' in k: # is a plot
            if 'xlabel' not in v:
                la.append("")
            else:
                la.append(v['xlabel']['words'])
    a = {"xlabels":la}
    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', a)
    if return_qa: 
        qa_pairs['Level 1']['Plot-level questions']['xlabels'] = {'Q':q, 'A':a}
        return qa_pairs


def q_plot_ylabels(data, qa_pairs, return_qa=True, verbose=True):
    q = 'What are the y-axis titles for each figure panel?'
    q += ' You are a helpful assistant, please format the output as a json as {"ylabels":[]}, where the list is a list of strings of all of the y-axis titles.  '
    q += 'If there is a single plot, this should be one element in this list, and if there are multiple plots the list should be in row-major (C-style) order.'
    q += " If a plot does not have an y-axis title, then denote this by an empty string in the list.  "
    q += "Please format any formulas in the title in a Python LaTeX string (for example 'Light $\\\\alpha$'). "
    la = []
    for k,v in data.items():
        if 'plot' in k: # is a plot
            if 'ylabel' not in v:
                la.append("")
            else:
                la.append(v['ylabel']['words'])
    a = {"ylabels":la}
    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', a)
    if return_qa: 
        qa_pairs['Level 1']['Plot-level questions']['ylabels'] = {'Q':q, 'A':a}
        return qa_pairs



def ticklabels(data, qa_pairs, tick_type='x', return_qa=True, verbose=True):
    q = 'What are the values for each of the tick marks on the '+tick_type+'-axis?'
    q += ' You are a helpful assistant, please format the output as a json as {"'+tick_type+'tick values":[[]]}, where each element of the outer list refers to a single panel, '
    q += 'and each inner list is a list of the '+tick_type+'-axis tick mark values. '
    q += 'If there is a single plot, this should be one element in the outer list, and if there are multiple plots the outer list should be in row-major (C-style) order.'
    q += " If a plot does not have any "+tick_type+"-axis tick values, then denote this by an empty string in the inner list.  "
    q += "Please format any formulas in the "+tick_type+"-axis tick values in a Python LaTeX string (for example 'Light $\\\\alpha$'). "
    la = []
    for k,v in data.items():
        if 'plot' in k: # is a plot
            if tick_type+'ticks' not in v:
                la.append("")
            else:
                t = []
                for l in v[tick_type+'ticks']:
                    t.append(l['data'])
                la.append(t)
    a = {tick_type+"tick values":la}
    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', a)
    if return_qa: 
        qa_pairs['Level 1']['Plot-level questions'][tick_type+'tick values'] = {'Q':q, 'A':a}
        return qa_pairs



def q_plot_types(data, qa_pairs, plot_types, return_qa=True, use_list=True, verbose=True):
    q = 'What are the plot types for each panel in the figure? '
    q += 'You are a helpful assistant, please format the output as a json as {"plot types":[]}, where each element of the list refers to the plot type of a single panel. '
    q += 'If there is a single plot, this should be one element in this list, and if there are multiple plots the list should be in row-major (C-style) order. '
    if use_list:
        q += 'Please choose each plot type from the following list: ['
        for pt in plot_types:
            q += pt + ', '
        q = q[:-2] # take off the last bit
        q += '].'
    la = []
    for k,v in data.items():
        if 'plot' in k: # is a plot
            if 'type' not in v:
                la.append("")
            else:
                la.append(v['type'])
    a = {"plot types":la}
    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', a)
    if return_qa: 
        qa_pairs['Level 1']['Plot-level questions']['plot types'] = {'Q':q, 'A':a}
        return qa_pairs
