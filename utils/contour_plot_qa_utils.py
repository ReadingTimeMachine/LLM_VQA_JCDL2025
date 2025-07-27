import numpy as np

#from utils.plot_qa_utils import plot_index_to_words

#q_gmm_ngaussians_hists



from .plot_qa_utils import get_nplots, persona, context, how_many, how_much_data_values, what_is_relationship, what_is_relationship_plot#, check_relationship




# ####### CONSTRUCT QUESTIONS ########


########## L1 ##########
def q_plottype(data, plot_type, qa_pairs, plot_num = [0,0], 
                         return_qa=True, use_words=True, 
                         use_list=True, 
                        line_list = ['line','scatter','contour', 'histogram'], 
                        single_figure_flag=True,
                        verbose=True, text_persona = None):
    """
    use_words : set to True to translate row, column to words; False will use C-ordering indexing
    use_list : give a list of possible distributions
    use_nplots : give the number of lines in the prompt
    """
    
    # what kind of plot
    big_tag = 'plot type'
    val_type = 'a string'

    ### answer
    #dist = data['plot'+str(plot_num)]['distribution']
    # # to match
    # if dist == 'gmm': dist = 'gaussian mixture model'
    # la = dist
    la = str(plot_type)

    # ---- don't have to change much below this -----
    # get nplots    
    nplots = get_nplots(data)
    #adder = get_adder(nplots, use_words)
    text_question, adder, text_format = what_is_relationship_plot(big_tag, nplots=nplots, 
                                                              val_type=val_type)
    
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

    if use_list:
        adder = adder.split(')')[0] + ' + list)'
        text_format += ' Please choose from the following list: ['
        for pt in line_list:
            text_format += pt + ', '
        text_format = text_format[:-2] # take off the last bit
        text_format += '].'

    q = text_persona + " " + text_context + " " + text_question + " " + text_format

    a = {big_tag + ' ' + adder:la}

    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', a)
    if return_qa: 
        if big_tag + ' ' + adder not in qa_pairs['Level 3']['Plot-level questions']:
            #print('yes', big_tag_short + ' ' + adder)
            qa_pairs['Level 3']['Plot-level questions'][big_tag + ' ' + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a, 
                                                                                                        'persona':text_persona, 
                                                                                                        'context':text_context,
                                                                                                        'question':text_question, 
                                                                                                        'format':text_format}}
        else:
            qa_pairs['Level 3']['Plot-level questions'][big_tag + ' ' + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a, 
                                                                                                        'persona':text_persona, 
                                                                                                        'context':text_context,
                                                                                                        'question':text_question, 
                                                                                                        'format':text_format}
        return qa_pairs



# # this version tries to give column and row numbers
# def q_npoints_scatter_plot_plotnums(data, qa_pairs, plot_num = [0,0], 
#                                return_qa=True, verbose=True, use_words=True, 
#                                single_figure_flag=True, 
#                                text_persona = None):
#     """
#     Construct Q/A for how many lines are in the plot.
#     """
#     big_tag = 'npoints'
#     object = 'scatter points'
#     # get answer
#     ans = len(data['plot'+str(plot_num)]['data']['xs']) # only 1 x-value

#     #--- typically don't need to change below ----

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
#                                                        use_words=use_words)
#     # check adder
#     # if adder != '':
#     #     adder = adder
#     # construct question:
#     q = text_persona + " " + text_context + " " + text_question + " " + text_format
#     # get answer, formatted
#     a = {big_tag + adder: ans}
#     if verbose:
#         print('QUESTION:', q)
#         print('ANSWER:', a)
#     if return_qa: 
#         if big_tag + adder not in qa_pairs['Level 1']['Plot-level questions']:
#             qa_pairs['Level 1']['Plot-level questions'][big_tag +  adder] = {'plot'+str(plot_num):{'Q':q, 'A':a, 
#                                                                                                         'persona':text_persona, 
#                                                                                                         'context':text_context,
#                                                                                                         'question':text_question, 
#                                                                                                         'format':text_format}}
#         else:
#             qa_pairs['Level 1']['Plot-level questions'][big_tag + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a, 
#                                                                                                         'persona':text_persona, 
#                                                                                                         'context':text_context,
#                                                                                                         'question':text_question, 
#                                                                                                         'format':text_format}
#         return qa_pairs
    

# def q_stats_scatters(data, qa_pairs, stat = {'minimum':np.min}, axis = 'x',
#                   plot_num = [0,0], 
#                      return_qa=True, use_words=True, verbose=True, 
#                      single_figure_flag=True, 
#                                text_persona = None):
#     """
#     stat: {'name':stat} which gives name of stat and function to calculate it, like {'minimum':np.min}
#     use_words : set to True to translate row, column to words; False will use C-ordering indexing
#     stat : dictionary of the name and function to use for each stat
#     """
#     # output type
#     val_type = 'a floats'

#     # check
#     if axis.lower() == 'x': 
#         axis = 'x'
#     elif axis.lower() == 'y':
#         axis = 'y'
#     elif axis.lower() == 'color':
#         axis = 'color'
#     else:
#         print('Axis not chosen correctly:', axis)
#         import sys; sys.exit()

#     #### Answer
#     f = list(stat.values())[0] # what statical function
#     zs = data['plot'+str(plot_num)]['data'][axis+'s']
#     # list_stat = []
#     # for z in zs:
#     #     list_stat.append(f(z))

#     # for_each = ', where each element of the list corresponds to one line in the plot'
#     for_each = ''
#     list_stat = f(zs)

#     #---- Don't have to change much below ----
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

#     text_question, adder, text_format  = how_much_data_values(big_tag, nplots=nplots, 
#                                                               axis=axis, 
#                                                               val_type=val_type, 
#                                                               use_words=use_words, 
#                                                               along_an_axis=True,
#                                                               for_each=for_each)
#     # big tag update
#     big_tag += ' ' + axis
#     # format answer
#     #la = {big_tag + " "+axis:list_stat}
#     la = {big_tag:list_stat}
#     ans = {big_tag +  adder:{'plot'+str(plot_num):la}} 
#     a = {big_tag +  adder:la}
#     # construct question:
#     q = text_persona + " " + text_context + " " + text_question + " " + text_format

#     if verbose:
#         print('QUESTION:', q)
#         print('ANSWER:', ans)
#     if return_qa: 
#         if big_tag + adder not in qa_pairs['Level 2']['Plot-level questions']:
#             qa_pairs['Level 2']['Plot-level questions'][big_tag + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a, 
#                                                                                                         'persona':text_persona, 
#                                                                                                         'context':text_context,
#                                                                                                         'question':text_question, 
#                                                                                                         'format':text_format}}
#         else:
#             qa_pairs['Level 2']['Plot-level questions'][big_tag + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a, 
#                                                                                                         'persona':text_persona, 
#                                                                                                         'context':text_context,
#                                                                                                         'question':text_question, 
#                                                                                                         'format':text_format}
#         return qa_pairs
    



# ########## L2/L3 #############
# from .plot_qa_utils import what_is_relationship
# def q_relationship_scatters(data, qa_pairs, plot_num = [0,0], 
#                          return_qa=True, use_words=True, 
#                          use_list=True, 
#                         line_list = ['random','linear','gaussian mixture model'], 
#                         single_figure_flag=True,
#                         verbose=True, text_persona = None):
#     """
#     use_words : set to True to translate row, column to words; False will use C-ordering indexing
#     use_list : give a list of possible distributions
#     use_nplots : give the number of lines in the prompt
#     """
    
#     # how many plots
#     big_tag = 'distribution'
#     val_type = 'a string'
#     for_each = ''
#     along_an_axis = True
#     axis = 'color'
#     mark = 'scatter'

#     ### answer
#     dist = data['plot'+str(plot_num)]['distribution']
#     # to match
#     if dist == 'gmm': dist = 'gaussian mixture model'
#     la = dist

#     # ---- don't have to change much below this -----
#     # get nplots    
#     nplots = get_nplots(data)
#     #adder = get_adder(nplots, use_words)
#     text_question, adder, text_format = what_is_relationship(big_tag, nplots=nplots, 
#                                                               val_type=val_type, 
#                                                               use_words=use_words, 
#                                                               along_an_axis=along_an_axis,
#                                                               axis=axis,
#                                                               for_each=for_each)
    
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

#     if use_list:
#         adder = adder.split(')')[0] + ' + list)'
#         if for_each != '':
#             fe = ' for each '+mark
#             eot = 'each'
#         else:
#             fe = ''
#             eot = 'the'
#         text_format += ' Please choose '+eot+' '+big_tag+fe+' from the following list: ['
#         for pt in line_list:
#             text_format += pt + ', '
#         text_format = text_format[:-2] # take off the last bit
#         text_format += '].'

#     q = text_persona + " " + text_context + " " + text_question + " " + text_format

#     a = {big_tag + ' ' + adder:la}

#     if verbose:
#         print('QUESTION:', q)
#         print('ANSWER:', a)
#     if return_qa: 
#         if big_tag + ' ' + adder not in qa_pairs['Level 3']['Plot-level questions']:
#             #print('yes', big_tag_short + ' ' + adder)
#             qa_pairs['Level 3']['Plot-level questions'][big_tag + ' ' + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a, 
#                                                                                                               'note':'this currently assumes all elements on a single plot have the same relationship type', 
#                                                                                                         'persona':text_persona, 
#                                                                                                         'context':text_context,
#                                                                                                         'question':text_question, 
#                                                                                                         'format':text_format}}
#         else:
#             qa_pairs['Level 3']['Plot-level questions'][big_tag + ' ' + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a, 
#                                                                                                               'note':'this currently assumes all elements on a single plot have the same relationship type', 
#                                                                                                         'persona':text_persona, 
#                                                                                                         'context':text_context,
#                                                                                                         'question':text_question, 
#                                                                                                         'format':text_format}
#         return qa_pairs
    
   

