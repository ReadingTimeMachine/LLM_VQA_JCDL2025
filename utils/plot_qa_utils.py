import numpy as np

# stats = {'minimum':np.min, 'maximum':np.max, 'median':np.median, 'mean':np.mean}
# #stats.keys()
# {'minimum':np.min}.values()

############ FIGURES IN GENERAL ##############

# plot index to words
n_to_word = {0:'first', 1:'second', 2:'third', 3:'fourth', 4:'fifth', 5:'sixth', 6:'seventh', 7:'eighth', 8:'ninth', 9:'tenth'}
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
        p = n_to_word[x] + ' row and ' + n_to_word[y] + ' column'
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




# # dpi
# def dpi(data, qa_pairs, return_qa=True, verbose=True):
#     q1 = 'What is the DPI (dots per square inch) of this figure?'
#     q1 += ' You are a helpful assistant, please format the output as a json as {"dpi":""} to store the DPI of the plot.'
#     a1 = {"dpi":data['figure']['dpi']}
#     if verbose:
#         print('QUESTION:', q1)
#         print('ANSWER:', a1)
#         print('')
#     # add to pairs
#     if return_qa: 
#         qa_pairs['Level 1']['Figure-level questions']['dpi'] = {'Q':q1, 'A':a1}
#         return qa_pairs



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


def context_single_multi(data, nplots, plot_num, use_words, single_figure_flag):
    if nplots == 1 and single_figure_flag:
        text_context = context(0, 0, use_words=use_words,
                                single_figure_flag=single_figure_flag)
    else:
        nrow = data['figure']['plot indexes'][plot_num][0]
        ncol = data['figure']['plot indexes'][plot_num][1]
        # ncols = data['figure']['ncols']
        # nrows = data['figure']['nrows']
        pindex = data['figure']['plot indexes'][plot_num]
        #print('nrow, ncol, pindex', nrow,ncol,pindex)
        text_context = context(nrow,ncol,plot_index=pindex, 
                               use_words=use_words, 
                               single_figure_flag=False)
    return text_context


##### FEEDER FUNCTIONS (NEW) ######

def get_adder(nplots, use_words, use_list=False, use_nlines=False):
    if use_words and nplots > 1:
        adder = ' (words)'
    elif not use_words and nplots > 1:
        adder = ' (plot numbers)'
    elif nplots == 1:
        adder = ''
    if use_list:
        adder += ' (list)'
    if use_nlines:
        adder += ' (nlines)'
    return adder


def how_many(object, big_tag, val_type = 'an integer', nplots = 1, 
             use_words=True, to_generate=False):
    """
    to_generate : flag for wording
    """
    if not to_generate:
        q = 'How many '+object+' are there in the specified figure panel?'
    else:
        q = 'How many ' + object + ' were used to generate the data for the plot in the figure panel?'
    adder = get_adder(nplots, use_words)
    # formatting for output
    format = 'Please format the output as a json as {"'+big_tag+'":""} for this figure panel, where the "'+big_tag+'" value should be '+val_type+'.'
    return q, adder, format


def how_much_data_values(big_tag, nplots=1, axis='x', val_type='a float', 
                         use_words=True, along_an_axis=False, 
                         for_each=''):
    axis_words = ''
    if along_an_axis:
        axis_words = 'along the ' + axis + '-axis'
    #q = 'What are the '+big_tag+' data values '+axis_words+' in this figure panel? '
    #q = 'What is the '+big_tag+' data value '+axis_words+' in this figure panel? ' # What is the median data value in this figure panel?
    q = 'What is the '+big_tag+' value of the data '+axis_words+' in this figure panel? ' # What is the median data value in this figure panel?
    adder = get_adder(nplots, use_words)
    # list or not?
    if 'list' in val_type:
        outputf = '"[]"'
    else:
        outputf = '""'
    # formatting for output
    format = 'Please format the output as a json as {"'+big_tag+' '+axis + '":'+outputf+'} for this figure panel, where the "'+big_tag+' '+axis +'" value should be '+val_type+', calculated from the '
    format += 'data values used to create the plot'+for_each+'.'
    # check formatting in case of any double spaces
    q = q.replace('  ', ' ')
    format = format.replace('  ', ' ')
    return q, adder, format


def what_is_relationship(big_tag, nplots=1, axis='x', val_type='a float', 
                         use_words=True, along_an_axis=False, 
                         for_each='', use_list=False):
    """
    axis : 'x', 'y', 'color' or 'x/y'/'x-y' -- if '/' or '-' assume along 2 axes.
    """
    axis_words = ''
    axis = ' ' + axis
    if along_an_axis:
        if '/' not in axis and '-' not in axis:
            axis_words = ' along the' + axis + '-axis'
        else:
            axis_words = ' in the' + axis + '-plane'
        pass
    else:
        axis = ''
    #q = 'What are the '+big_tag+' data values '+axis_words+' in this figure panel? '
    q = 'What is the underlying '+big_tag+' used to create the data in this figure panel'+axis_words+'?'
    adder = get_adder(nplots, use_words, use_list=use_list)
    # list or not?
    if 'list' in val_type:
        outputf = '"[]"'
    else:
        outputf = '""'
    # formatting for output
    format = 'Please format the output as a json as {"'+big_tag+axis + '":'+outputf+'} for this figure panel, where the "'+big_tag+axis +'" value should be '+val_type+', calculated from the '
    format += 'data values used to create the plot'+for_each+'.'
    return q, adder, format


def what_is_relationship_plot(big_tag, nplots=1, val_type='a float', 
                         use_words=True):
    q = 'What is the '+big_tag+' used to create the plot in this figure panel?'
    adder = get_adder(nplots, use_words)
    # list or not?
    if 'list' in val_type:
        outputf = '"[]"'
    else:
        outputf = '""'
    # formatting for output
    #print('***')
    #print(big_tag)
    #print(outputf)
    #print(val_type)
    format = 'Please format the output json as '
    #format += '{"'+big_tag+ + '":'+outputf+'} '
    format += '{"' + big_tag + '":'+outputf+'} '
    #format += 'for this figure panel, where the "'+big_tag+ +'" value should be '+val_type+' for this plot.'
    format += 'for this figure panel, where the "'+big_tag+'" value shoudl be '+val_type+' for this plot.'
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