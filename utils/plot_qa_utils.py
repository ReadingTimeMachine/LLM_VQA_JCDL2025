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


############### ROUND 2 ---- BETTER QA ###############

def get_nplots(data):
    # how many plots
    nplots = 0
    for k,v in data.items():
        if 'plot' in k:
            nplots += 1
    return nplots

def persona(text=None):
    """
    Craft a persona for the LLM to add into each question.

    text : if set to None, a default will be created.
    """
    if text is None:
        text = 'You are a helpful assistant that can analyze images.'
    return text

def context(nrow, ncol, plot_index = [0,0], 
            use_words=True, single_figure_flag=True):
    """
    This sets the context of the question by flagging what panel of 
     the figure for the LLM to look at.

    use_words : if True will replace plot indices to words 
      (e.g. upper left plot)
    single_figure_flag : if True, will not use plot numbers for 
      single-panel figures
    """

    if not use_words:
        #nrow = data['figure']['plot indexes'][plot_num][0]
        #ncol = data['figure']['plot indexes'][plot_num][1]
        q = 'The following question refers to the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '. '
        q += 'If there are multiple plots the panels will be in row-major (C-style) order, with the numbering starting at (0,0) in the upper left panel. '
        q += 'If there is one plot, then this row and column refers to the single plot. '
        #q += 'How many '+object+' are there for the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '? '
        #adder += '(plot numbers)'
    else: # translate to words
        #q = 'How many '+object+' are there for the plot in the ' + \
        #    plot_index_to_words(plot_index) + ' panel? '   
        q = 'The following question refers to the plot in the ' + \
        plot_index_to_words(plot_index) + ' panel.'
        
    if nrow == ncol and nrow == 0 and single_figure_flag: # just use one plot
        q = ''
    return q


##### FEEDER FUNCTIONS ######

def how_many(object, big_tag, val_type = 'an integer', nplots = 1, 
             use_words=True, to_generate=False):
    """
    to_generate : flag for wording
    """
    if not to_generate:
        q = 'How many '+object+' are there in the specified figure panel?'
    else:
        q = 'How many ' + object + ' were used to generate the data for the plot in the figure panel?'
    if use_words and nplots > 1:
        adder = '(words)'
    elif not use_words and nplots > 1:
        adder = '(plot numbers)'
    elif nplots == 1:
        adder = ''
    # formatting for output
    format = 'Please format the output as a json as {"'+big_tag+'":""} for this figure panel, where the "'+big_tag+'" value should be '+val_type+'.'
    return q, adder, format

def how_much_data_values(big_tag, nplots=1, axis='x', val_type='a float', use_words=True):
    q = 'What are the '+big_tag+' data values in this figure panel? '
    if use_words and nplots > 1:
        adder = '(words)'
    elif not use_words and nplots > 1:
        adder = '(plot numbers)'
    elif nplots == 1:
        adder = ''
    # formatting for output
    format = 'Please format the output as a json as {"'+big_tag+' '+axis + '":""} for this figure panel, where the "'+big_tag+'" value should be '+val_type+', calculated from the '
    format += 'data values used to create the plot.'
    return q, adder, format


def check_relationship(data, plot_num, qa_pairs, rel = 'gmm',
                       return_qa = True, verbose=False):
    # is this the correct relationship? if not this is an error
    hasRel= False
    #rel = 'gmm'
    if data['plot'+str(plot_num)]['distribution'] != rel: # not a linear relationship
        if verbose:
            print('Not a '+rel+' relationship!')
        if return_qa:
            return qa_pairs, hasRel
        else:
            return None, hasRel

    if data['plot'+str(plot_num)]['distribution'] == rel:
        hasRel = True

    if not hasRel:
        print("This is not a '"+rel+"' relationship, can't ask any questions!'")
        import sys; sys.exit()
    return hasRel, hasRel


def init_qa_pairs():
    """
    Set format for qa pairs part of JSON
    """
    # create qa pairs
    qa_pairs = {}
    # question levels    
    qa_pairs['Level 1'] = {}
    qa_pairs['Level 2'] = {}
    qa_pairs['Level 3'] = {}
    qa_pairs['Level 1']['Figure-level questions'] = {} # Figure-level questions
    qa_pairs['Level 1']['Plot-level questions'] = {}
    qa_pairs['Level 2']['Plot-level questions'] = {}
    qa_pairs['Level 3']['Plot-level questions'] = {}
    return qa_pairs