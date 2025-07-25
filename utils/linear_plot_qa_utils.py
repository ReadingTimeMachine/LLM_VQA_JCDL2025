import numpy as np

#from utils.plot_qa_utils import plot_index_to_words

#q_gmm_ngaussians_hists



from .plot_qa_utils import get_nplots, persona, context, how_many, how_much_data_values, check_relationship




####### CONSTRUCT QUESTIONS ########

# this version tries to give column and row numbers
def q_nlines_plot_plotnums(data, qa_pairs, plot_num = [0,0], 
                               return_qa=True, verbose=True, use_words=True, 
                               single_figure_flag=True, 
                               text_persona = None):
    """
    Construct Q/A for how many lines are in the plot.
    """
    big_tag = 'nlines'
    object = 'lines'
    # get answer
    ans = len(data['plot'+str(plot_num)]['data']['ys'])

    #--- typically don't need to change below ----

    nplots = get_nplots(data)

    ### persona of assistant
    text_persona = persona(text=text_persona)
    ## context for question
    if nplots == 1 and single_figure_flag:
        text_context = context(0, 0, use_words=use_words,
                                single_figure_flag=single_figure_flag)
    else:
        nrow = data['figure']['plot indexes'][plot_num][0]
        ncol = data['figure']['plot indexes'][plot_num][1]
        pindex = data['figure']['plot indexes'][plot_num]
        text_context = context(nrow,ncol,plot_index=pindex, use_words=use_words)

    ### question, format of output
    text_question, adder, text_format = how_many(object, big_tag, 
                                                       val_type = 'an integer', 
                                                       nplots = nplots, 
                                                       use_words=use_words)
    # check adder
    # if adder != '':
    #     adder = adder
    # construct question:
    q = text_persona + " " + text_context + " " + text_question + " " + text_format
    # get answer, formatted
    a = {big_tag + adder: ans}
    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', a)
    if return_qa: 
        if big_tag + adder not in qa_pairs['Level 1']['Plot-level questions']:
            qa_pairs['Level 1']['Plot-level questions'][big_tag +  adder] = {'plot'+str(plot_num):{'Q':q, 'A':a, 
                                                                                                        'persona':text_persona, 
                                                                                                        'context':text_context,
                                                                                                        'question':text_question, 
                                                                                                        'format':text_format}}
        else:
            qa_pairs['Level 1']['Plot-level questions'][big_tag + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a, 
                                                                                                        'persona':text_persona, 
                                                                                                        'context':text_context,
                                                                                                        'question':text_question, 
                                                                                                        'format':text_format}
        return qa_pairs
    

def q_stats_lines(data, qa_pairs, stat = {'minimum':np.min}, axis = 'x',
                  plot_num = 0, 
                     return_qa=True, use_words=True, verbose=True, 
                     single_figure_flag=True, 
                               text_persona = None):
    """
    stat: {'name':stat} which gives name of stat and function to calculate it, like {'minimum':np.min}
    use_words : set to True to translate row, column to words; False will use C-ordering indexing
    stat : dictionary of the name and function to use for each stat
    """
    # output type
    val_type = 'a list of floats'

    # check
    if axis.lower() == 'x': 
        axis = 'x'
    elif axis.lower() == 'y':
        axis = 'y'
    else:
        print('Axis not chosen correctly:', axis)
        import sys; sys.exit()

    #### Answer
    f = list(stat.values())[0] # what statical function
    zs = data['plot'+str(plot_num)]['data'][axis+'s']
    list_stat = []
    for z in zs:
        list_stat.append(f(z))

    #---- Don't have to change much below ----
    big_tag = list(stat.keys())[0]
    # get nplots    
    nplots = get_nplots(data)

    ### persona of assistant
    text_persona = persona(text=text_persona)
    ## context for question
    if nplots == 1 and single_figure_flag:
        text_context = context(0, 0, use_words=use_words,
                                single_figure_flag=single_figure_flag)
    else:
        nrow = data['figure']['plot indexes'][plot_num][0]
        ncol = data['figure']['plot indexes'][plot_num][1]
        pindex = data['figure']['plot indexes'][plot_num]
        text_context = context(nrow,ncol,plot_index=pindex, use_words=use_words)

    text_question, adder, text_format  = how_much_data_values(big_tag, nplots=1, 
                                                              axis=axis, 
                                                              val_type=val_type, 
                                                              use_words=use_words, 
                                                              along_an_axis=True)
    # big tag update
    big_tag += ' ' + axis
    # format answer
    #la = {big_tag + " "+axis:list_stat}
    la = {big_tag:list_stat}
    ans = {big_tag +  adder:{'plot'+str(plot_num):la}} 
    a = {big_tag +  adder:la}
    # construct question:
    q = text_persona + " " + text_context + " " + text_question + " " + text_format

    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', ans)
    if return_qa: 
        if big_tag + adder not in qa_pairs['Level 2']['Plot-level questions']:
            qa_pairs['Level 2']['Plot-level questions'][big_tag + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a, 
                                                                                                        'persona':text_persona, 
                                                                                                        'context':text_context,
                                                                                                        'question':text_question, 
                                                                                                        'format':text_format}}
        else:
            qa_pairs['Level 2']['Plot-level questions'][big_tag + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a, 
                                                                                                        'persona':text_persona, 
                                                                                                        'context':text_context,
                                                                                                        'question':text_question, 
                                                                                                        'format':text_format}
        return qa_pairs
    



########## L2/L3 #############

def q_relationship_lines(data, qa_pairs, plot_num = 0, return_qa=True, use_words=True, use_list=True, use_nlines = True,
                        line_list = ['random','linear','gaussian mixture model'], 
                        single_figure_flag=True,
                        verbose=True):
    """
    use_words : set to True to translate row, column to words; False will use C-ordering indexing
    use_list : give a list of possible distributions
    use_nplots : give the number of lines in the prompt
    """
    
    # how many plots
    big_tag_short = 'relationship'
    # get nplots    
    nplots = get_nplots(data)

    ### persona of assistant
    text_persona = persona(text=text_persona)
    ## context for question
    if nplots == 1 and single_figure_flag:
        text_context = context(0, 0, use_words=use_words,
                                single_figure_flag=single_figure_flag)
    else:
        nrow = data['figure']['plot indexes'][plot_num][0]
        ncol = data['figure']['plot indexes'][plot_num][1]
        pindex = data['figure']['plot indexes'][plot_num]
        text_context = context(nrow,ncol,plot_index=pindex, use_words=use_words)

    # **HERE** get text persona

    # # construct question:
    # q = text_persona + " " + text_context + " " + text_question + " " + text_format
    # # get answer, formatted
    # a = {big_tag + adder: ans}

    # *****
    # nplots = 0
    # for k,v in data.items(): # count number of plots
    #     if 'plot' in k:
    #         nplots += 1
    # if not use_words:
    #     adder = '(plot numbers)'
    #     # rows columns
    #     nrow = data['figure']['plot indexes'][plot_num][0]
    #     ncol = data['figure']['plot indexes'][plot_num][1]
    #     q = 'The following question refers to the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '. '
    #     q += 'If there are multiple plots the panels will be in row-major (C-style) order, with the numbering starting at (0,0) in the upper left panel. '
    #     q += 'If there is one plot, then this row and column refers to the single plot. '
    # else: 
    #     adder = '(words)'

    # if nplots == 1: # single plot
    #     q = 'What is the functional relationship between the x and y values in this figure? '
    # else:
    #     if not use_words:
    #         q += 'What is the functional relationship between the x and y values for the figure panel on row number ' + str(nrow) + ' and column number ' + str(ncol) + '? '
    #     else:
    #         q = 'What is the functional relationship between the x and y values for the plot in the ' + plot_index_to_words(data['figure']['plot indexes'][plot_num]) + ' panel? '     

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