import numpy as np

from utils.plot_qa_utils import plot_index_to_words

#q_stats_hists
#q_gmm_ngaussians_hists

def persona(text=None):
    """
    Craft a persona for the LLM to add into each question.

    text : if set to None, a default will be created.
    """
    if text is None:
        text = 'You are a helpful assistant that can analyze images.'
    return text


def context(nrow, ncol, plot_index = 0, 
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


def how_many(object, big_tag, val_type = 'an integer', nplots = 1, use_words=True):
    q = 'How many '+object+' are there in the specified figure panel?'
    if use_words and nplots > 1:
        adder = '(words)'
    elif not use_words and nplots > 1:
        adder = '(plot numbers)'
    elif nplots == 1:
        adder = ''
    # formatting for output
    format = 'Please format the output as a json as {"'+big_tag+'":""} for this figure panel, where the "'+big_tag+'" value should be '+val_type+'.'
    return q, adder, format



# this version tries to give column and row numbers
def q_nbars_hist_plot_plotnums(data, qa_pairs, plot_num = 0, 
                               return_qa=True, verbose=True, use_words=True, 
                               single_figure_flag=True, 
                               text_persona = None):
    big_tag = 'nbars'
    object = 'bars'
    #adder = ''
    # how many plots
    nplots = 0
    for k,v in data.items():
        if 'plot' in k:
            nplots += 1

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
    # get answer
    a = {big_tag + ' ' + adder: len(data['plot'+str(plot_num)]['data from plot']['data'][0])}
    # construct question:
    q = text_persona + " " + text_context + " " + text_question + " " + text_format
    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', a)
    if return_qa: 
        if big_tag + ' ' + adder not in qa_pairs['Level 1']['Plot-level questions']:
            qa_pairs['Level 1']['Plot-level questions'][big_tag + ' ' + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a, 
                                                                                                        'persona':text_persona, 
                                                                                                        'context':text_context,
                                                                                                        'question':text_question, 
                                                                                                        'format':text_format}}
        else:
            qa_pairs['Level 1']['Plot-level questions'][big_tag + ' ' + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a, 
                                                                                                        'persona':text_persona, 
                                                                                                        'context':text_context,
                                                                                                        'question':text_question, 
                                                                                                        'format':text_format}
        return qa_pairs