import pandas as pd
import pickle
import numpy as np

from nltk.corpus import stopwords
import string

import spacy

nlp = spacy.load("en_core_web_sm")


#from importlib import reload # debug
#import utils.synthetic_fig_utils # debug
#reload(synthetic_fig_utils) # debug

from utils.synthetic_fig_utils import subset_by_percent, \
 get_nrows_and_ncols, normalize_params_prob, get_ticks, get_titles_or_labels, \
 get_font_info

# this is hacktacular from: https://github.com/ReadingTimeMachine/TexSoup
# and needs to be updated
#tex_lib = '/Users/jnaiman/TexSoup/'
# tex_lib = '/Users/jnaiman/scienceDigitization/arxiv_mining_synthetic_data/utils/'
# from sys import path
# path.append(tex_lib)
from .TexSoupUtils import preprocessing, postprocess
# import TexSoup
# from TexSoup import preprocessing, postprocess



def get_popular_nouns(data_dir, 
                      ntop = 5000, # top total words --> only nouns are taken
                      ntop_nouns = 1000, # now many top nouns
                      ignore_words = ['l', 'sect', 'cent'] # words to ignore
                     ):

    # all clean words
    with open(data_dir+'words_cleaned.pickle','rb') as f:
        words_cleaned = pickle.load(f)
    
    # also cleaned words
    normalized=True
    pdf_letters = []; ocr_letters = []; counts = []; counts_unnormed = []
    for pl,ols in words_cleaned.items():
        if normalized:
            cdiv=0.0
            for ol,c in ols.items(): # % in OCR
                cdiv += c
        else:
            cdiv = 1.0
            
        for ol,c in ols.items():
            pdf_letters.append(pl)
            ocr_letters.append(ol)
            counts.append(c/cdiv*100)
            counts_unnormed.append(c)
            
    df_words_clean = pd.DataFrame({'pdf_letters':pdf_letters,
                            'ocr_letters':ocr_letters,
                            'counts':counts, 'counts unnormalized':counts_unnormed})
    # stop words
    # lets look for stopwords
    stop_words_all = stopwords.words('English')
    
    # get upper case alpha characters
    alphas = list(string.ascii_lowercase)
    alphas_lower = alphas.copy()
    # add larger ones
    for a in alphas_lower:
        alphas.append(a.upper())
    
    digits = np.arange(0,10,1).astype('int').astype('str').tolist()
    
    punctuation = list(string.punctuation)
    # pop out our markers
    for p in ['^','@']:
        try:
            i = punctuation.index(p)
            punctuation.pop(i)
        except:
            pass
            
    not_freq_words = []
    not_freq_words.extend(stop_words_all)
    not_freq_words.extend(digits)
    not_freq_words.extend(alphas)
    not_freq_words.extend(punctuation)
    
    # other randoms to take out
    words_randoms = []
    words_randoms.extend(['-','fig','et'])
    
    df_freq = df_words_clean.loc[~((df_words_clean['pdf_letters'].isin(not_freq_words)) & (df_words_clean['ocr_letters'].isin(not_freq_words)))]
    df_freq_large = df_words_clean.loc[~(df_words_clean['pdf_letters'].isin(not_freq_words))]
    
    # how many top words?
    pdf_words = df_freq_large.sort_values('counts unnormalized',ascending=False).iloc[:ntop]['pdf_letters'].values
    
    df_freq_top = df_words_clean.loc[(df_words_clean['pdf_letters'].isin(pdf_words))&(df_words_clean['ocr_letters'].isin(pdf_words))]
    df_freq_top_large = df_words_clean.loc[df_words_clean['pdf_letters'].isin(pdf_words)]
    df_freq_top_large = df_freq_top_large.loc[~df_freq_top_large['pdf_letters'].isin(words_randoms)]
    print(len(df_freq_top),len(df_freq_top_large))
    
    # cut off lower-counted things
    tol_count = 5
    df_freq_top = subset_by_percent(df_freq_top.copy(), tol_count = tol_count) # formatting
    df_freq_top_large = subset_by_percent(df_freq_top_large.copy(), tol_count = tol_count) # formatting

    # list by most frequent
    words_list = np.array(df_freq_top_large.groupby('pdf_letters')['% of all OCR tokens'].sum().sort_values(ascending=False).index)
    
    #words = " ".join(df_freq_top_large['pdf_letters'].unique().tolist())
    words = " ".join(words_list.tolist())
    doc = nlp(words)
    
    popular_nouns = []
    # loop and get just the nouns
    for token in doc:
        if len(popular_nouns) > ntop_nouns-1:
            break
        if (str(token.pos_) == 'NOUN') and (token.text not in words_randoms):
            if token.text.lower() not in ignore_words:
                popular_nouns.append(token.text)

    return popular_nouns


def get_inline_math(fullproc_r,
                    base_dir_arxiv = None,
                    dirs_latex = None,
                    recreate_inlines = False, # if true, you need access to data!
                    verbose = True,
                    use_uniques=True

                   ):
    # get a random inline math formula
    recreate_inlines = False # if true, you need access to data!
    
    if recreate_inlines:
        inlines = []
        
        texfiles = glob(base_dir_arxiv + dirs_latex[0] + '/*/*.tex')
        for i in range(1,len(dirs_latex)):
            texfiles.extend(glob(base_dir_arxiv + dirs_latex[i] + '/*/*.tex'))
        #len(texfiles)
    
        for ii,f in enumerate(texfiles):
            if ii%100 == 0 and verbose:
                print('on', ii+1, 'of', len(texfiles), 'so far have', len(inlines), 'inline formulas')
                
        # inline_math = ''
        # while inline_math == '':
            
            #f = np.random.choice(texfiles)
            try:
                with open(f, 'r') as ff:
                    tex = ff.read()
            except Exception as e:
                if verbose:
                    print('issue opening file:', f)
                    print(e)
                continue
        
            try:
                soup = TexSoup.TexSoup(r''+tex, tolerance=0)
            except Exception as e:
                if verbose:
                    print('issue makign soup for:', f)
                    print(e)
                continue
                
            # pre-process with our TexSoup
            err = False
            tex_doc = preprocessing.process_begin_end(tex) # cleaning begin/end
            tex_doc, error_accents = preprocessing.clean_accents(tex_doc) # parse accents
            if error_accents:
                continue
            # get new commands/envs
            newcommands, newenvironments = preprocessing.get_newcommands_and_newenvs(tex_doc,
                                                                                 verbose=False)
            args_newcommands,err = preprocessing.find_args_newcommands(newcommands,error_out=False, 
                                                                   verbose=False)
            if err:
                continue 
            args_newenvironments, err = preprocessing.find_args_newenvironments(newenvironments,
                                                                       error_out=False, verbose=False)        
            if err:
                continue
        
              
            error = False
            try:
                comments, find_replace, error = preprocessing.generate_find_replace_newcommands(args_newcommands, 
                                                                       verbose=False,arg_type = 'newcommand')  
                error = error[0]
            except:
                error = True
                
            if error:
                continue
            
            warnings = []
            try:
                tex_doc_nc, error, warnings = preprocessing.replace_newcommands_and_newenvironments(tex_doc, 
                                                                                      args_newcommands,
                                                                                      args_newenvironments,
                                                                                                    verbose=True)
            except:
                error = [True]
                
            if error[0]:
                continue
                
            # try to make soup
            error = False
            try:
                soup = TexSoup.TexSoup(r''+tex_doc_nc, tolerance=0)
            except:
                error = True
            if error:
                continue
        
            # clean up commands with slashes
            error = False
            try:
                soup_clean = postprocess.clean_slash_commands(soup)
            except:
                error = True
            if error:
                continue
                
            error = False
            try:
                texout_arr = postprocess.parse_soup_to_tags(soup,tex_doc_nc, verbose=False)
            except:
                error = True
            if error:
                continue
    
            for t,tt in texout_arr:
                if tt == 'inline':
                    inlines.append(t)
            #inlines = np.unique(inlines).tolist()
    
        #inlines = np.unique(inlines).tolist()
        with open(fullproc_r + 'inlines.csv','w') as f:
            for l in inlines:
                print(l.replace('\n', ' '),file=f)
        # and uniques
        inlines_u = np.unique(inlines)
        with open(fullproc_r + 'inlines_unique.csv','w') as f:
            for l in inlines_u:
                print(l.replace('\n', ' '),file=f)
    
        inlines_ignore = []
        for il,l in enumerate(inlines):
            try:
                plt.close('all')
                plt.plot([1,2],[1,2])
                plt.xlabel(r""+l)
                plt.savefig(fake_figs_dir + 'p1.png')
            except Exception as e:
                #print(e)
                inlines_ignore.append(il)
    
        with open(fullproc_r + 'inlines_uniques_ignore.csv','w') as f:
            for l in inlines_ignore:
                print(l,file=f)
    else:
        if not use_uniques:
            if verbose: print('not using unique inlines, not sure if this is a good idea!')
            with open(fullproc_r + 'inlines.csv', 'r') as f:
                inlines = f.readlines()
        else:
            with open(fullproc_r + 'inlines_unique.csv', 'r') as f:
                inlines = f.readlines()
            with open(fullproc_r + 'inlines_uniques_ignore.csv', 'r') as f:
                inds_ignore = f.readlines()
            inds_ignore = np.array(inds_ignore).astype('int')
            inlines = np.delete(inlines, inds_ignore)
            
    
    if verbose: print('number of inlines = ', len(inlines))
    return inlines