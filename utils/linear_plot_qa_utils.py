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
    big_tag = 'nlines'
    object = 'lines'
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
    

# def q_stats_hists(data, qa_pairs, stat = {'minimum':np.min}, plot_num = 0, 
#                      return_qa=True, use_words=True, verbose=True, 
#                      single_figure_flag=True, 
#                                text_persona = None):
#     """
#     stat: {'name':stat} which gives name of stat and function to calculate it, like {'minimum':np.min}
#     use_words : set to True to translate row, column to words; False will use C-ordering indexing
#     stat : dictionary of the name and function to use for each stat
#     """
#     big_tag = list(stat.keys())[0]
#     # get nplots    
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

#     text_question, adder, text_format  = how_much_data_values(big_tag, nplots=1, 
#                                                               axis='x', 
#                                                               val_type='a float', 
#                                                               use_words=use_words)
    
#     #### Answer
#     f = list(stat.values())[0] # what stastical function
#     xs = data['plot'+str(plot_num)]['data']['xs']
#     la = {big_tag + " x":f(xs)}#, big_tag + " y":f(ys)}
    
#     ans = {big_tag + ' ' + adder:{'plot'+str(plot_num):la}} 
#     a = {big_tag + ' ' + adder:la}
#     # construct question:
#     q = text_persona + " " + text_context + " " + text_question + " " + text_format

#     if verbose:
#         print('QUESTION:', q)
#         print('ANSWER:', ans)
#     if return_qa: 
#         if big_tag + ' ' + adder not in qa_pairs['Level 2']['Plot-level questions']:
#             qa_pairs['Level 2']['Plot-level questions'][big_tag + ' ' + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a, 
#                                                                                                         'persona':text_persona, 
#                                                                                                         'context':text_context,
#                                                                                                         'question':text_question, 
#                                                                                                         'format':text_format}}
#         else:
#             qa_pairs['Level 2']['Plot-level questions'][big_tag + ' ' + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a, 
#                                                                                                         'persona':text_persona, 
#                                                                                                         'context':text_context,
#                                                                                                         'question':text_question, 
#                                                                                                         'format':text_format}
#         return qa_pairs
    

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

#     a = {big_tag + ' ' + adder:la}
#     # construct question:
#     q = text_persona + " " + text_context + " " + text_question + " " + text_format

#     if verbose:
#         print('QUESTION:', q)
#         print('ANSWER:', {'plot'+str(plot_num):a})
#     if return_qa: 
#         if big_tag + ' ' + adder not in qa_pairs['Level 3']['Plot-level questions']:
#             qa_pairs['Level 3']['Plot-level questions'][big_tag + ' ' + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a, 
#                                                                                                         'persona':text_persona, 
#                                                                                                         'context':text_context,
#                                                                                                         'question':text_question, 
#                                                                                                         'format':text_format}}
#         else:
#             qa_pairs['Level 3']['Plot-level questions'][big_tag + ' ' + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a, 
#                                                                                                         'persona':text_persona, 
#                                                                                                         'context':text_context,
#                                                                                                         'question':text_question, 
#                                                                                                         'format':text_format}
#         return qa_pairs