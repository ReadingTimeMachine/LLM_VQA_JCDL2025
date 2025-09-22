import numpy as np


####### CONSTRUCT QUESTIONS ########

from .plot_qa_utils import get_nplots, persona, context_single_multi, how_many, how_much_data_values, check_relationship

# this version tries to give column and row numbers
def q_nbars_hist_plot_plotnums(data, qa_pairs, plot_num = [0,0], 
                               return_qa=True, verbose=True, use_words=True, 
                               single_figure_flag=True, 
                               text_persona = None):
    big_tag = 'nbars'
    object = 'bars'
    nplots = get_nplots(data)

    ### persona of assistant
    text_persona = persona(text=text_persona)
    ## context for question
    text_context = context_single_multi(data, nplots, plot_num, use_words, single_figure_flag)

    ### question, format of output
    text_question, adder, text_format = how_many(object, big_tag, 
                                                       val_type = 'an integer', 
                                                       nplots = nplots, 
                                                       use_words=use_words)
    # get answer
    a = {big_tag + adder: len(data['plot'+str(plot_num)]['data from plot']['data'][0])}
    # construct question:
    q = text_persona + " " + text_context + " " + text_question + " " + text_format
    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', a)
    if return_qa: 
        if big_tag + adder not in qa_pairs['Level 1']['Plot-level questions']:
            qa_pairs['Level 1']['Plot-level questions'][big_tag + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a, 
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
    

def q_stats_hists(data, qa_pairs, stat = {'minimum':np.min}, plot_num = 0, 
                     return_qa=True, use_words=True, verbose=True, 
                     single_figure_flag=True, 
                               text_persona = None):
    """
    stat: {'name':stat} which gives name of stat and function to calculate it, like {'minimum':np.min}
    use_words : set to True to translate row, column to words; False will use C-ordering indexing
    stat : dictionary of the name and function to use for each stat
    """
    big_tag = list(stat.keys())[0]
    # get nplots    
    nplots = get_nplots(data)

    ### persona of assistant
    text_persona = persona(text=text_persona)
    ## context for question
    text_context = context_single_multi(data, nplots, plot_num, use_words, single_figure_flag)

    text_question, adder, text_format  = how_much_data_values(big_tag, nplots=1, 
                                                              axis='x', 
                                                              val_type='a float', 
                                                              use_words=use_words)
    
    #### Answer
    f = list(stat.values())[0] # what stastical function
    xs = data['plot'+str(plot_num)]['data']['xs']
    la = {big_tag + " x":f(xs)}#, big_tag + " y":f(ys)}
    
    ans = {big_tag + adder:{'plot'+str(plot_num):la}} 
    a = {big_tag + adder:la}
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
from .plot_qa_utils import what_is_relationship
def q_relationship_histograms(data, qa_pairs, plot_num = 0, 
                         return_qa=True, use_words=True, 
                         use_list=True, 
                        line_list = ['random','linear','gaussian mixture model'], 
                        single_figure_flag=True,
                        verbose=True, text_persona = None):
    """
    use_words : set to True to translate row, column to words; False will use C-ordering indexing
    use_list : give a list of possible distributions
    use_nplots : give the number of lines in the prompt

    """
    
    # how many plots
    big_tag = 'distribution'
    val_type = 'a string'
    for_each = ''
    along_an_axis = False
    mark = 'scatter'
    #big_tag += '-' + axis

    ### answer
    dist = data['plot'+str(plot_num)]['distribution']
    # to match
    if dist == 'gmm': dist = 'gaussian mixture model'
    la = dist

    # ---- don't have to change much below this -----
    # get nplots    
    nplots = get_nplots(data)
    #adder = get_adder(nplots, use_words)
    text_question, adder, text_format = what_is_relationship(big_tag, nplots=nplots, 
                                                              val_type=val_type, 
                                                              use_words=use_words, 
                                                              along_an_axis=along_an_axis,
                                                              for_each=for_each)
    
    ### persona of assistant
    text_persona = persona(text=text_persona)
    ## context for question
    text_context = context_single_multi(data, nplots, plot_num, use_words, single_figure_flag)

    if use_list:
        adder = adder.split(')')[0] + ' + list)'
        if for_each != '':
            fe = ' for each '+mark
            eot = 'each'
        else:
            fe = ''
            eot = 'the'
        text_context += ' Please choose '+eot+' '+big_tag+fe+' from the following list: ['
        for pt in line_list:
            text_context += pt + ', '
        text_context = text_context[:-2] # take off the last bit
        text_context += '].'

    q = text_persona + " " + text_context + " " + text_question + " " + text_format

    a = {big_tag + adder:la}

    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', a)
    if return_qa: 
        if big_tag + adder not in qa_pairs['Level 3']['Plot-level questions']:
            qa_pairs['Level 3']['Plot-level questions'][big_tag + '-' + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a, 
                                                                                                              'note':'this currently assumes all elements on a single plot have the same relationship type', 
                                                                                                        'persona':text_persona, 
                                                                                                        'context':text_context,
                                                                                                        'question':text_question, 
                                                                                                        'format':text_format}}
        else:
            qa_pairs['Level 3']['Plot-level questions'][big_tag + '-' + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a, 
                                                                                                              'note':'this currently assumes all elements on a single plot have the same relationship type', 
                                                                                                        'persona':text_persona, 
                                                                                                        'context':text_context,
                                                                                                        'question':text_question, 
                                                                                                        'format':text_format}
        return qa_pairs





def q_gmm_ngaussians_hists(data, qa_pairs, plot_num = 0, 
                           return_qa=True, use_words=True, verbose=True, 
                           single_figure_flag = True, 
                               text_persona = None, 
                               additional_context=''):
    """
    use_words : set to True to translate row, column to words; False will use C-ordering indexing
    use_nlines : give the number of lines in the prompt
    additional_context : if you want to for example give a range of number of hists
    """

    # check correct relationship
    hasRel, qatmp = check_relationship(data, plot_num, qa_pairs, rel = 'gmm',
                       return_qa = return_qa, verbose=verbose)
    if not hasRel:
        return qatmp

    big_tag = 'ngaussians'
    object = 'gaussians'
    nplots = get_nplots(data)

    ### persona of assistant
    text_persona = persona(text=text_persona)
    ## context for question
    text_context = context_single_multi(data, nplots, plot_num, use_words, single_figure_flag)

    ### question, format of output
    text_question, adder, text_format = how_many(object, big_tag, 
                                                       val_type = 'an integer', 
                                                       nplots = nplots, 
                                                       use_words=use_words, 
                                                       to_generate=True)
    text_format += additional_context

    ### answer
    la = data['plot'+str(plot_num)]['data']['data params']['nclusters']

    a = {big_tag + adder:la}
    # construct question:
    q = text_persona + " " + text_context + " " + text_question + " " + text_format

    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', {'plot'+str(plot_num):a})
    if return_qa: 
        if big_tag + adder not in qa_pairs['Level 3']['Plot-level questions']:
            qa_pairs['Level 3']['Plot-level questions'][big_tag + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a, 
                                                                                                        'persona':text_persona, 
                                                                                                        'context':text_context,
                                                                                                        'question':text_question, 
                                                                                                        'format':text_format}}
        else:
            qa_pairs['Level 3']['Plot-level questions'][big_tag + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a, 
                                                                                                        'persona':text_persona, 
                                                                                                        'context':text_context,
                                                                                                        'question':text_question, 
                                                                                                        'format':text_format}
        return qa_pairs