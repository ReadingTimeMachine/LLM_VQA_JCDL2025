# this does general plot-level questions
from .plot_qa_utils import get_nplots, persona, get_adder, context_single_multi

def q_errorbars_existance_lines(data, qa_pairs, axis = 'x', plot_num = 0, return_qa=True, 
                                use_words=True, verbose=True, single_figure_flag=True, 
                                text_persona=None):
    """
    use_words : set to True to translate row, column to words; False will use C-ordering indexing
    """
    
    big_tag = axis + '-errorbars'
    ### question
    text_question = 'Are there error bars on the data along the ' +axis+ '-axis in this figure panel?'
    ### format
    text_format = 'Please format the output as a json as {"'+axis+'-axis errors":""} where the value for the output is True '
    text_format += '(if error bars exist) or False (if error bars do not exist).'

    nplots = get_nplots(data)
    adder = get_adder(nplots, use_words)
    # get answer
    if axis+'errs' in data['plot' + str(plot_num)]['data'].keys():
        a = {axis+'-axis errors' + adder:True}
    else:
        a = {axis+'-axis errors' + adder:False}

    ### persona of assistant
    text_persona = persona(text=text_persona)
    ## context for question
    text_context = context_single_multi(data, nplots, plot_num, use_words, single_figure_flag)
    q = text_persona + " " + text_context + " " + text_question + " " + text_format

    #a = list(aout.values())[0]
    if verbose:
        print('QUESTION:', q)
        print('ANSWER:', a)
    if return_qa: 
        # JPN updates -- leads to "xx-error bars" and "yy-error bars"
        if axis+'-axis errors' + adder not in qa_pairs['Level 2']['Plot-level questions']:
            # qa_pairs['Level 2']['Plot-level questions'][axis+big_tag + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a, 
            #                                                                                             'persona':text_persona, 
            #                                                                                             'context':text_context,
            #                                                                                             'question':text_question, 
            #                                                                                             'format':text_format}}
            qa_pairs['Level 2']['Plot-level questions'][big_tag + adder] = {'plot'+str(plot_num):{'Q':q, 'A':a, 
                                                                                                        'persona':text_persona, 
                                                                                                        'context':text_context,
                                                                                                        'question':text_question, 
                                                                                                        'format':text_format}}
        else:
            # qa_pairs['Level 2']['Plot-level questions'][axis+big_tag + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a, 
            #                                                                                             'persona':text_persona, 
            #                                                                                             'context':text_context,
            #                                                                                             'question':text_question, 
            #                                                                                             'format':text_format}
            qa_pairs['Level 2']['Plot-level questions'][big_tag + adder]['plot'+str(plot_num)] = {'Q':q, 'A':a, 
                                                                                                        'persona':text_persona, 
                                                                                                        'context':text_context,
                                                                                                        'question':text_question, 
                                                                                                        'format':text_format}
        return qa_pairs




