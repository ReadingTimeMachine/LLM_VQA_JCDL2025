# post processing utils
from TexSoup.data import TexNamedEnv, TexCmd, TexText, TexMathModeEnv, BraceGroup, TexDisplayMathModeEnv, TexDisplayMathEnv

#from TexSoup.preprocessing import accents, accents_alone
#from TexSoup.preprocessing import accents, accents_alone
# why does this not work???
accents = ['\\`',"\\'",'\\^','\\"','\\H', '\\~', '\\c', '\\k', 
           '\\l', '\\=', '\\b', '\\.', '\\d', '\\r', '\\u', 
           '\\v', '\\t']
accents_alone = ['\\o', '{\\i}', '\\oe', '\\ae', '\\ss', '\\aa', '\\AA', 
                '\\O', '\\AE', '\\OE']



from string import whitespace



def clean_slash_commands(soup, only_clean_document = True):
    if only_clean_document:
        # combine \command, \ as just \command\
        for iss,s in enumerate(soup.all):
            #if type(s.expr) == TexSoup.data.TexNamedEnv: # parts of the text
            if type(s.expr) == TexNamedEnv: # parts of the text
                if 'begin' in s.expr.begin and 'document' in s.expr.begin:
                    for isss,ss in enumerate(s.all):
                        if str(ss) == '\\ ': # just a lone bracket
                            #if type(s.all[isss-1].expr) == TexSoup.data.TexCmd: # is a command, update it
                            if type(s.all[isss-1].expr) == TexCmd: # is a command, update it
                                soup.all[iss].all[isss-1].name += '\\ ' # add 
                                soup.all[iss].all[isss].delete() # delete this node
    else:
        # combine \command, \ as just \command\
        for iss,s in enumerate(soup.all):
            #if type(s.expr) == TexSoup.data.TexNamedEnv: # parts of the text
            if type(s.expr) == TexNamedEnv: # parts of the text
                for isss,ss in enumerate(s.all):
                    if str(ss) == '\\ ': # just a lone bracket
                        #if type(s.all[isss-1].expr) == TexSoup.data.TexCmd: # is a command, update it
                        if type(s.all[isss-1].expr) == TexCmd: # is a command, update it
                            soup.all[iss].all[isss-1].name += '\\ ' # add 
                            soup.all[iss].all[isss].delete() # delete this node
    return soup


#phrase = ' The key practical difference between both methods'

def get_replacement_tex(tex_doc, soup, isss, s, verbose=False, strip=True, verbose_level = 1):
    """
    tex_doc : this is the *original* tex document, NOT the one that generated from the soup
    soup : the soup from TexSoup
    """
    ind1 = s.position
    err = False
    if ind1 == -1: ind1 = 0 # newline
    if str(s) not in tex_doc[ind1:]: # has been reformated
        if isss < len(soup.all)-1:
            next_el = soup.all[isss+1]
            #if phrase in str(next_el):
            #    print('**** get_replacement_tex:', 'IN next_el 1')
            ind2 = next_el.position
            icount = 1
            #icount = 0 # should this be 0 or 1???
            # what happens if next position is -1? 
            while (ind2 == -1 or ind2 is None) and icount < len(soup.all)-isss: # for new-lines
                next_el = soup.all[isss+icount]
                ind2 = next_el.position # this should be starting character of next element
                if len(str(next_el).strip()) > 0: # not just whitespace
                    if (str(next_el)[0] == '\n') or (str(next_el)[0] == '\t') or (str(next_el)[0] == '\r'): # starts with a newline
                        # update the count for extra
                        icount += 1
                        ind2 = soup.all[isss+icount].position - len(str(next_el))
                icount+=1
            if icount == len(soup.all)-isss: # hit the end
                ind2 = len(tex_doc)
        else: # last one
            ind2 = len(tex_doc)
        strout = tex_doc[ind1:ind2]
        if strip:
            strout = strout.rstrip()
    else:
        #if phrase in str(s):
        #    print('**** get_replacement_tex:', 'IN tex_doc')
        #    print(str(s))
            
        strout = str(s)

    #if phrase in str(strout):
    #    print('')
    #    print('**** get_replacement_tex:', 'in strout')
    #    print(str(s))
    #    print('-------------')
    #    print(str(strout))
    #    print('')

        
    if verbose and verbose_level>=1:
        print(str(s))
        print('')
        print(strout)
        print('----------------------')
    if err:
        print('HAS ERROR')
    return strout, err


def parse_soup(soup, tex_doc_accent, verbose=False):
    icount = 0
    texout_arr = []
    errorAll = False
    err = False
    for isss, s in enumerate(soup.all):
        #if phrase in str(s):
        #    print('parse_soup:', 'in s', str(s).count(phrase))
            
        if type(s.expr) == TexNamedEnv: # parts of the text
            if 'begin' in s.expr.begin and 'document' in s.expr.begin: # beginning and end of document
                # add preamble
                #texout_arr.append((preamble,'preamble'))
                texout_arr.append(('\\begin{document}\n','beginDoc'))
                try: 
                    iterator = s.all
                except:
                    errorAll = True
                if not errorAll:
                    for is3,ss in enumerate(s.all):  # go through document
                        #if phrase in str(ss):
                        #    print('parse_soup:', 'in ss', str(ss).count(phrase))
                        if type(ss.expr) == TexText: # mark text words
                            #if phrase in str(ss):
                            #    print('parse_soup: IS TEXT', 'in ss', str(ss).count(phrase))
                            strout,err = get_replacement_tex(tex_doc_accent, s, is3,ss, verbose=verbose)
                            if err:
                                return '',True
                            #if phrase in str(ss):
                            #    print('parse_soup: strout 1', 'in ss', strout.count(phrase))
                            if len(strout.strip()) > 0: # not just white-space
                                if strout.lstrip()[0] != '%': # not a comment:
                                    if str(ss.expr) in accents_alone: # accent alone
                                        #print('here 1')
                                        strout,err = get_replacement_tex(tex_doc_accent, s, is3,ss, verbose=verbose)
                                        if err:
                                            return '', err
                                        texout_arr.append((strout, 'accent'))
                                    elif texout_arr[-1][0] in accents:
                                        #print('here 2')
                                        strout,err = get_replacement_tex(tex_doc_accent, s, is3,ss, verbose=verbose)
                                        if err:
                                            return '', err
                                        texout_arr[-1] = ((texout_arr[-1][0] + strout, 'accent')) # put together
                                    else:
                                        #if phrase in str(ss):
                                        #    print('parse_soup: strout 2', strout.count(phrase))
                                        #    print('what is strout', strout)

                                        texout_arr.append((strout, 'text'))
                                else: # for all comments
                                    texout_arr.append((strout, 'comment'))
                            else: # just whitespace
                                texout_arr.append((strout,'whitespace'))
                        elif 'cite' in ss.name: # citations
                            strout,err = get_replacement_tex(tex_doc_accent, s, is3,ss, verbose=verbose)
                            if err:
                                return '',err
                            texout_arr.append((strout, 'citation'))
                        elif 'ref' in ss.name: # references
                            strout,err = get_replacement_tex(tex_doc_accent, s, is3,ss, verbose=verbose)
                            if err:
                                return '',err
                            texout_arr.append((strout, 'reference'))
                        elif type(ss.expr) == TexMathModeEnv: # inline math
                            #if isss == 197:
                            #    print('yes math mode')
                            strout,err = get_replacement_tex(tex_doc_accent, s, is3,ss, verbose=verbose)
                            if err:
                                return '',err
                            texout_arr.append((strout,'inline'))
                        elif type(ss.expr) == BraceGroup:
                            strout,err = get_replacement_tex(tex_doc_accent, s, is3,ss, verbose=verbose)
                            if err:
                                return '',err
                            isAccent = False
                            for a in accents:
                                if a in strout:
                                    isAccent = True
                            if not isAccent:
                                for a in accents_alone:
                                    if a in strout:
                                        isAccent = True
                            if isAccent:
                                texout_arr.append((strout,'accent'))
                            else:
                                # possible that last one is an accent
                                if texout_arr[-1][0] in accents:
                                    texout_arr[-1] = ((texout_arr[-1][0] + strout, 'accent')) # put together
                                else: # just a bracket
                                    texout_arr.append((strout,'bracket'))
                        elif type(ss.expr) == TexDisplayMathModeEnv:
                            strout,err = get_replacement_tex(tex_doc_accent, s, is3,ss, verbose=verbose)
                            if err:
                                return '',err
                            texout_arr.append((strout,'displayMath')) 
                        elif type(ss.expr) == TexDisplayMathEnv:
                            strout,err = get_replacement_tex(tex_doc_accent, s, is3,ss, verbose=verbose)
                            if err:
                                return '',err
                            texout_arr.append((strout,'displayMath')) 
                        else: # a command or named environment
                            # special ones
                            if str(ss.expr).strip() == '\\S':
                                strout,err = get_replacement_tex(tex_doc_accent, s, is3,ss, verbose=verbose)
                                if err:
                                    return '',err
                                texout_arr.append((strout, 'S-command'))
                            # JPN added 20250622
                            # elif 'section' in str(ss.name):
                            #     strout = str(ss)
                            #     isAccent = False
                            #     for a in accents:
                            #         if a in strout:
                            #             isAccent = True
                            #     if not isAccent:
                            #         for a in accents_alone:
                            #             if a in strout:
                            #                 isAccent = True
                            #     if isAccent:
                            #         texout_arr.append((strout,'accent'))
                            #     else:
                            #         texout_arr.append((str(ss), 'section')) 
                            elif type(ss.expr) == TexNamedEnv:
                                #if phrase in str(ss):
                                #    print('here now in named env')
                                strout,err = get_replacement_tex(tex_doc_accent, s, is3,ss, verbose=verbose)
                                if err:
                                    return '',err
                                #if phrase in strout:
                                #    print('isss=', isss, 'is3=', is3)
                                #    print('here now in named env -- but after')
                                #    print(str(ss))
                                #    print('-----------')
                                #    print(strout)
                                #    print('')
                                #if 'eqnarray' in str(ss): import sys; sys.exit()
                                texout_arr.append((strout, 'namedEnv'))
                            elif str(ss.expr) in accents_alone: # accent alone
                                strout,err = get_replacement_tex(tex_doc_accent, s, is3,ss, verbose=verbose)
                                if err:
                                    return '',err
                                texout_arr.append((strout, 'accent'))
                            elif texout_arr[-1][0] in accents:
                                strout,err = get_replacement_tex(tex_doc_accent, s, is3,ss, verbose=verbose)
                                if err:
                                    return '',err
                                texout_arr[-1] = ((texout_arr[-1][0] + strout, 'accent')) # put together
                                #import sys; sys.exit()
                            else:
                                strout,err = get_replacement_tex(tex_doc_accent, s, is3,ss, verbose=verbose)
                                if err:
                                    return '',err
                                texout_arr.append((strout,'commandOrBracket'))

                else: # some error has happend in getting s-all
                    if verbose: print('soup error encountered')
                    texout_arr.append((str(ss),'soup error'))
                    continue
                texout_arr.append(('\\end{document}\n','endDoc'))
            else: # something else -- outside of document
                if verbose: print('not in begin/end!', s)
                #import sys; sys.exit()
                ##errorAll = True
                strout,err = get_replacement_tex(tex_doc_accent, soup, isss,s, verbose=verbose)
                if err:
                    return '',err
                texout_arr.append((strout,'outside'))
        else: # typically \usepackage commands, \documentclass, etc
            strout,err = get_replacement_tex(tex_doc_accent, soup, isss,s, verbose=verbose)
            if err:
                return '', err
            texout_arr.append((strout, 'others'))         
            
    return texout_arr,err



def parse_soup_after_accents(texout_arr):
    # combine accents
    texout_arr2 = []
    text = ''
    for it,(t,ttype) in enumerate(texout_arr):
        if ttype == 'accent' and len(texout_arr2) > 0:
            # what is the ender of the last one?
            if texout_arr2[-1][0][-1] not in whitespace: # yup, connect the two together
                texout_arr2[-1] = ((texout_arr2[-1][0]+t,'textWithAccent'))
            else: # just leave as is
                texout_arr2.append((t,'textWithAccent'))
        else:
            texout_arr2.append((t,ttype))

    # also combine before... 
    # should probably do this in one loop for efficiency but... you know how it goes
    texout_arr3 = []
    skips = 0
    for it,(t,ttype) in enumerate(texout_arr2):
        if skips==0:
            #if it>66: import sys; sys.exit()
            if ttype == 'textWithAccent': #check after
                ik = 1
                if it+ik < len(texout_arr2)-1: # not at end
                    c = texout_arr2[it+ik][0]
                    typeNext = texout_arr2[it+ik][1]
                    cout = ''
                    if c[0] not in whitespace and (typeNext=='text' or typeNext=='textWithAccent'):
                        while c[0] not in whitespace and it+ik < len(texout_arr2)-1 \
                          and (typeNext=='text' or typeNext=='textWithAccent'):
                            c = texout_arr2[it+ik][0]
                            cout += c
                            ik += 1
                        texout_arr3.append((t+cout, 'textWithAccent'))
                        skips = ik-1
                    else: # have whitespace, should be it's own thing
                        texout_arr3.append((t,ttype))
                else:
                    texout_arr3.append((t,ttype))
            else:
                texout_arr3.append((t,ttype))
        else:
            skips -= 1
            
    return texout_arr3


# combine together, etc
def parse_soup_to_tags(soup, tex_doc_nc, verbose=False):
    #print('parse_soup_to_tags:', 'soup --', str(soup).count(' The key practical difference between both methods'), 'times')
    #print('parse_soup_to_tags:', 'tex --', tex_doc_nc.count(' The key practical difference between both methods'), 'times')

    # # return before for debugging
    # print("CHANGE BACK HERE!!")
    # return tex_doc_nc, False

    err = False
    texout_arr,err = parse_soup(soup,tex_doc_nc,verbose=verbose)
    if err:
        if verbose:
            print('ERROR: parse_soup -- erroring out')
            return '',True
    # print('ALSO CHANGE HERE!')
    # return texout_arr,err
    #count = 0
    #for t,tt in texout_arr:
    #    if ' The key practical difference between both methods' in t:
    #        count += 1
    #print('after parse soup:', count, 'times')
    texout_arr_out = parse_soup_after_accents(texout_arr)
    count = 0
    #for t,tt in texout_arr_out:
    #    if ' The key practical difference between both methods' in t:
    #        count += 1
    #print('after parse soup after accents:', count, 'times')
    return texout_arr_out, err
