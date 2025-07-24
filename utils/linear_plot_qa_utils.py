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
    

# def q_gmm_ngaussians_hists(data, qa_pairs, plot_num = 0, 
#                            return_qa=True, use_words=True, verbose=True, 
#                            single_figure_flag = True, 
#                                text_persona = None):
#     """
#     use_words : set to True to translate row, column to words; False will use C-ordering indexing
#     use_nlines : give the number of lines in the prompt
#     """

#     # check correct relationship
#     hasRel, qatmp = check_relationship(data, plot_num, qa_pairs, rel = 'gmm',
#                        return_qa = return_qa, verbose=verbose)
#     if not hasRel:
#         return qatmp

#     big_tag = 'ngaussians'
#     object = 'gaussians'
#     nplots = get_nplots(data)

#     ### persona of assistant
#     text_persona = persona(text=text_persona)
#     ## context for question
#     if nplots == 1 and single_figure_flag:
#         text_context = context(0, 0, use_words=use_words,
#                                 single_figure_flag=single_figure_flag)
#     else:
#         nrow = data['figure']['plot indexes'][plot_num][0]
#         ncol = data['figure']['plot indexes'][plot_num][1]
#         pindex = data['figure']['plot indexes'][plot_num]
#         text_context = context(nrow,ncol,plot_index=pindex, use_words=use_words)

#     ### question, format of output
#     text_question, adder, text_format = how_many(object, big_tag, 
#                                                        val_type = 'an integer', 
#                                                        nplots = nplots, 
#                                                        use_words=use_words, 
#                                                        to_generate=True)

#     ### answer
#     la = data['plot'+str(plot_num)]['data']['data params']['nclusters']

#     a = {big_tag + adder:la}
#     # construct question:
#     q = text_persona + " " + text_context + " " + text_question + " " + text_format

#     if verbose:
#         print('QUESTION:', q)
#         print('ANSWER:', {'plot'+str(plot_num):a})
#     if return_qa: 
#         if big_tag + adder not in qa_pairs['Level 3']['Plot-level questions']:
#             qa_pairs['Level 3']['Plot-level questions'][big_tag + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a, 
#                                                                                                         'persona':text_persona, 
#                                                                                                         'context':text_context,
#                                                                                                         'question':text_question, 
#                                                                                                         'format':text_format}}
#         else:
#             qa_pairs['Level 3']['Plot-level questions'][big_tag + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a, 
#                                                                                                         'persona':text_persona, 
#                                                                                                         'context':text_context,
#                                                                                                         'question':text_question, 
#                                                                                                         'format':text_format}
#         return qa_pairs