# Pre-processing for any fixes that need to be applied before generation of soup
import re
import numpy as np

#https://en.wikibooks.org/wiki/LaTeX/Special_Characters
# LaTeX command	Sample	Description
# \`{o}	ò	grave accent
# \'{o}	ó	acute accent
# \^{o}	ô	circumflex
# \"{o}	ö	umlaut, trema or dieresis
# \H{o}	ő	long Hungarian umlaut (double acute)
# \~{o}	õ	tilde
# \c{c}	ç	cedilla
# \k{a}	ą	ogonek
# \l{}	ł	barred l (l with stroke)
# \={o}	ō	macron accent (a bar over the letter)
# \b{o}	o	bar under the letter
# \.{o}	ȯ	dot over the letter
# \d{u}	ụ	dot under the letter
# \r{a}	å	ring over the letter (for å there is also the special command \aa)
# \u{o}	ŏ	breve over the letter
# \v{s}	š	caron/háček ("v") over the letter
# \t{oo}	o͡o	"tie" (inverted u) over the two letters

# "alone" accents
# \o{}	ø	slashed o (o with stroke)
# {\i}	ı	dotless i (i without tittle)
accents = ['\\`',"\\'",'\\^','\\"','\\H', '\\~', '\\c', '\\k', 
           '\\l', '\\=', '\\b', '\\.', '\\d', '\\r', '\\u', 
           '\\v', '\\t']
accents_alone = ['\\o', '{\\i}', '\\oe', '\\ae', '\\ss', '\\aa', '\\AA', 
                '\\O', '\\AE', '\\OE']


def process_begin_end(tex_doc,
                     search_weirdos_begin = r'\\begin(\s*){(\s*)[A-Za-z]*(\s*)}',
                     search_weirdos_end = r'\\end(\s*){(\s*)[A-Za-z]*(\s*)}'):
    """
    tex_doc : raw text input
    search_weirdos_begin : things like \begin {equation} or \end { document}
    search_weirdos_end : same deal, but for \end statements
    """
    ind = 0
    tex_doc_out = tex_doc
    while ind < len(tex_doc):
        if re.search(search_weirdos_begin, tex_doc[ind:]):
            istart,iend = re.search(search_weirdos_begin, tex_doc[ind:]).span()
            text = tex_doc[ind:][istart:iend]
            text_replace = "".join(text.split())
            #print(text, text_replace)
            tex_doc_out = tex_doc_out.replace(text,text_replace)
            ind += iend
        else:
            ind += len(tex_doc[ind:])

    tex_doc = tex_doc_out
    ind = 0
    while ind < len(tex_doc):
        if re.search(search_weirdos_end, tex_doc[ind:]):
            istart,iend = re.search(search_weirdos_end, tex_doc[ind:]).span()
            text = tex_doc[ind:][istart:iend]
            text_replace = "".join(text.split())
            #print(text, text_replace)
            tex_doc_out = tex_doc_out.replace(text,text_replace)
            ind += iend
        else:
            ind += len(tex_doc[ind:])
    tex_doc = tex_doc_out
    return tex_doc

# utils
def error_and_quit(message,ignore_quit=False,warn=True):
    if warn: print(message)
    if not ignore_quit: sys.exit()

#https://stackoverflow.com/questions/29991917/indices-of-matching-parentheses-in-python
def find_closing(text,dopen='(',dclose=')', debug=True,
                 remove_newline=False, check_closing=True):
    istart = []  # stack of indices of opening parentheses
    d = {}
    error = False
    #print('------text------')
    #print(text)
    #print('----------------')
    if remove_newline: text = text.strip('\n')
    text = text + '     '
    for i, c in enumerate(text):
        if c == dopen:
            istart.append(i)
        if c == dclose:
            try:
                d[istart.pop()] = i
            except IndexError:
                if debug: print('Too many closing parentheses')
                error=True
    if len(istart)!=0:  # check if stack is empty afterwards or left over openers?
        if debug: print('Too many opening parentheses, check 2')
        error=True
    if error and not check_closing: error = False
    return d, error

def split_function_with_delimiters(l,function='\\footnote',
                                   dopen='{',dclose='}',
                                   debug=False, 
                                  remove_newline=False,
                                  check_closing=True,
                                  start_after_function=False,
                                  verbose=False,
                                  return_bracket_index=False):
    if function not in l: # no function
        if debug: print('function not in text:', function)
        return -1, -1
    ind1 = l.index(function)
    if start_after_function:
        ind1 = ind1+len(function)
    tt = l[ind1:]
    # find first bracket
    if dopen not in l[ind1:]: # no first bracket
        if debug: print('no opening bracket in text')
        return -1,-1
    ind2 = l[ind1:].index(dopen)
    tt2 = l[ind1+ind2:]
    d,error = find_closing(tt2,dopen=dopen,dclose=dclose,
                           debug=debug,
                           remove_newline=remove_newline,
                          check_closing=check_closing)
    if error:
        if debug: print('error in find_closing')
        return ind1,-1
    try:
        ind3 = ind1+ind2+d[0]+1
    except:
        ind3 = -1; ind1 = -1
        if verbose: print(d)
        if verbose: print('in split_function_with_delimiters: no d[0]')
    if start_after_function: ind1 = ind1-len(function)
    if not return_bracket_index:
        return ind1,ind3 # ind1 starts at the start of the function, ind3 at its closing bracket
    else:
        return ind2,ind3 # ind2 start of braket, ind3 is end of bracket


def spc(l,function='\\footnote',
                                   dopen='{',dclose='}', error_tag='',
                                              start_after_function=False,
                                              error_out = True,
                                              return_bracket_index=False,
                                              verbose=False):
    error=False
    ind1,ind2 = split_function_with_delimiters(l,
                                                   function=function,
                                                   dopen=dopen,
                                                   dclose=dclose,
                                                   debug=False,
                                                  check_closing=False,
                                              start_after_function=start_after_function,
                                              return_bracket_index=return_bracket_index,
                                              verbose=verbose)
    if ind1 == -1 or ind2 == -1: # re-run w/o check closing
        if verbose: print(error_tag+' :not found, trying again w/o checking closing...')
        ind1,ind2 = split_function_with_delimiters(l,
                                                   function=function,
                                                   dopen=dopen,
                                                   dclose=dclose,
                                                   debug=False,
                                                  check_closing=False,
                                                  start_after_function=start_after_function,
                                                  return_bracket_index=return_bracket_index,
                                                  verbose=verbose)
        if ind1 == -1 or ind2 == -1: # re-run w/o check closing
            ind1,ind2 = split_function_with_delimiters(l,
                                                       function=function,
                                                       dopen=dopen,
                                                       dclose=dclose,
                                                       debug=verbose,
                                                      check_closing=False,
                                                      start_after_function=start_after_function,
                                                      return_bracket_index=return_bracket_index,
                                                      verbose=verbose)
            if error_out:
                print(function)
                print('---------')
                print(l)
                error_and_quit(error_tag+' : couldnt figure it out!')
            else:
                if verbose: print(error_tag+' : couldnt figure it out!')
                error=True
    if error_out:
        return ind1,ind2 # ind1 starts at the start of the function, ind3 at its closing bracket
    else:
        return ind1,ind2,error
    
    
def get_newcommands_and_newenvs(text, 
                                search = r'(\\newcommand)|(\\renewcommand)|(\\environment)|(\\newenvironment)|(\\renewenvironment)|(\\def)',
                                verbose = False):

    # -------------- search for new commands/environments ---------
    ind = 0
    news = []
    #error = []
    while ind<len(text):
        s = re.search(search, text[ind:])
        if s:
            news.append((s.group(0),ind+s.span()[0],ind+s.span()[1]))
            ind += s.span()[1]
        else: # nothing left!
            ind += len(text[ind:])


    # NOTE: all of this below is in no way elegant and should be dealt with
    newcommands = []
    for c,start,end in news:
        if ('\\newcommand' in c) or ('\\renewcommand' in c):
            ind1,ind2,err = spc(text[start:],function=c,
                                dopen='{',dclose='}',
                               error_out=False)
            if not err:
                if ind1==-1 or ind2==-1:
                    err = True
                    #error.append( ('error') )
                    newcommands.append( ('error','','',''))
                    #print('hihi1')
                    continue
                cc = "".join(text[start:][ind1:ind2].split())
                #cc = text[start:][ind1:ind2].replace(' ','')
                if '\\newcommand' in c:
                    ic = cc.index('\\newcommand')+len('\\newcommand')
                elif '\\renewcommand' in c:
                    ic = cc.index('\\renewcommand')+len('\\renewcommand')
                else:
                    if verbose: print('something has gone super wrong!')
                    err=True
                if not err:
                    char = cc[ic]
                    if char == '*': # stared version, \newcommand*{\cmd}[nargs]{defn}
                        ic += 1
                        char = cc[ic]
                    if char == '{': # "typical"
                        # get out interior
                        nc = text[start:start+ind2]
                        ind11 = nc.index('{')
                        # get actual command
                        ncc = nc[ind11+1:-1]
                        # get rest of command
                        ind11,ind22,err = spc(text[start+ind2:],function='',
                                          dopen='{',dclose='}', error_out=False)
                        if not err:
                            nc2 = text[start:start+ind22+ind2]
                            newcommands.append( (ncc,nc2,start,start+ind2+ind22) )
                        else:
                            newcommands.append( ('error','','',''))
                            #print('hihi2')
                            continue
                    elif char == '\\': # weirder - \newcommand\foo[]{}
                        # get out interior
                        nc = text[start:start+ind2]
                        i2 = nc.index('{') # assumes \def\foo{defn}
                        # is there a '[' ? like \newcommand\foo[nargs]{defn}
                        i4 = i2 + 1
                        try:
                            i4 = nc.index('[')
                        except:
                            pass
                        # also need to check for commandds like \newcommand\[{\begin{equation}}
                        if i4 < i2: # have a [
                            if ']' in nc[:i2]: # have closed! otherwise \[ is the command
                                i2 = i4
                        nc2 = nc[:i2]
                        i3 = nc2[::-1].index('\\')
                        ncc = '\\'+nc2[len(nc2)-i3:]                        
                        newcommands.append( (ncc,nc,start,start+ind2) )
                else:
                    newcommands.append(('error','','',''))
                    #print('hihi33')
                    continue

        elif '\\def' in c:
            # check one thing -- for the whole start to spell out define
            # want to search for \def NOT \define as the start
            if text[start:start+len('\\define')] == '\\define':
                err = True
            else:
                ind1,ind2,err = spc(text[start:],function=c,
                                    dopen='{',dclose='}',
                                   error_out=False)
            if not err:
                # get out interior
                nc = text[start:start+ind2]
                # find actual command
                i2 = nc.index('{')
                nc2 = nc[:i2]
                i3 = nc2[::-1].index('\\')
                ncc = '\\'+nc2[len(nc2)-i3:]
                newcommands.append( (ncc,nc,start,start+ind2) )
            else:
                newcommands.append(('error','','',''))
                #print('hihi44')
                continue
        # see if newcommands
        #print("newcommands here:", newcommands)    


    # also grab possible environments
    # check no blank \environment
    errEnv = False
    for c in news:
        if '\\environment' in c:
            errEnv = True

    if errEnv:
        newcommands.append(('error'))
        print('hihi444')
        # HERE continue        

    newenvironments = []
    #print('here')
    for c,start,end in news:
        if ('\\newenvironment' in c) or ('\\renewenvironment' in c):
            #\newenvironment{nam}[args]{begdef}{enddef} --> "nam"
            ind1,ind2,err = spc(text[start:],function=c,
                                dopen='{',dclose='}',
                               error_out=False)
            if not err:
                # get environment name
                i1 = text[start:][ind1:ind2].index('{')+1
                i2 = text[start:][ind1:ind2].index('}')
                envname = text[start:][ind1:ind2][i1:i2]

                # should be two other {}'s -- one of two
                ind11,ind22,err = spc(text[start:][ind2:],
                                      function='',
                                      dopen='{',dclose='}',
                                     error_out=False)
                if not err:
                    ind22 += ind2
                    # should be two other {}'s -- two of two
                    ind111,ind222,err = spc(text[start:][ind22:],
                                            function='',dopen='{',
                                            dclose='}',
                                           error_out=False)
                    if not err:
                        ind222 += ind22
                        newenvironments.append( (envname, text[start:][ind1:ind222], 
                                                 start, start+ind222) )
                    else:
                        newenvironments.append( ('error'))
                        #print('err here1')
                        continue
                else:
                    newenvironments.append(('error'))
                    #print('err here2')
                    continue
            else:
                newenvironments.append(('error'))
                #print('err here3')
                continue
                
    return newcommands, newenvironments


def find_args_newcommands(newcommands, error_out = False, verbose=False):
    # counting arguments
    args = []
    err = False
    for i,nn in enumerate(newcommands):
        nnc = nn[1]
        ind = 0
        nums = []
        while ind < len(nnc): # loop through the whole thing
            if '#' in nnc[ind:]: # have some inputs!
                ind = nnc[ind:].index('#')+ind # find next, keep index of whole
                if ind==0 or (nnc[ind-1]!='\\' and ind<len(nnc)-1): # zero index, not escaped, >1 left
                    if nnc[ind+1].isdigit(): # #1
                        nums.append(int(nnc[ind+1]))
                        if ind<len(nnc)-2: # like #11, should only be #1-#9 if there is extra 
                                           # number, we could have an issue
                            if nnc[ind+2].isdigit():
                                if verbose: 
                                    print('more dig','||', nnc, 
                                                  '||', nnc[ind+2], '||', 
                                                  nnc[ind+2:])
                                    print('')
                        ind+=2 # add 2 because #(DIGIT)
                    else: # not digit, moving on... could be ## !
                        ind += 1
                else:
                    ind += 1
            else:
                ind += len(nnc[ind:])
        if len(nums)>0:
            nums = np.max(nums)
        else:
            nums=0
        #print(nums)
        nout = list(nn).copy()
        nout.extend([nums])
        args.append(nout)

    errArgs = [False]
    for ia,a in enumerate(args):
        if a[-1]>0:
            # check for [NUM] and not from something like \def\command#1
            if '['+str(a[-1]) + ']' not in a[1] and '#' not in a[0] and '@' not in a[1]:
                if '\\def' not in a[1] and '{#' + str(a[-1]) + '}' not in a[1]: # sometimes \def\command{#(DIGIT)}
                    # is it just an add one?
                    if '[' + str(a[-1]+1) +']' not in a[1]:
                        if verbose:
                            print(a)
                            #print(f)
                        # just add and hope its all OK
                        i2 = a[1].split('[')[-1].split(']')[0]
                        if verbose: print(i2)
                        try:
                            args[ia][-1] = int(i2) 
                        except:
                            if verbose:
                                print('error in trying to get args:', a)#, ff)
                            errArgs = [True,a]
                    else:
                        args[ia][-1] = a[-1]+1

    if errArgs[0]:
        if error_out:
            print('error in args for newcommands!')
            import sys; sys.exit()
        err = True
        
    if error_out:
        return args
    else:
        return args, err
    
    
def find_args_newenvironments(newenvironments, error_out = False, verbose =False):
    # arguments of new environments not in commands
    args_env = []
    err = False
    for i,nn in enumerate(newenvironments):
        nnc = nn[1]
        ind = 0
        nums = []
        while ind < len(nnc): # loop through the whole thing
            if '#' in nnc[ind:]: # have some inputs!
                ind = nnc[ind:].index('#')+ind # find next, keep index of whole
                if ind==0 or (nnc[ind-1]!='\\' and ind<len(nnc)-1): # zero index, not escaped, >1 left
                    if nnc[ind+1].isdigit(): # #1
                        nums.append(int(nnc[ind+1]))
                        if ind<len(nnc)-2: # like #11, should only be #1-#9 if there is extra 
                                           # number, we could have an issue
                            if nnc[ind+2].isdigit():
                                if verbose:
                                    print('more dig','||', nnc, '||', nnc[ind+2], '||', nnc[ind+2:])
                                    print('')
                        ind+=2 # add 2 because #(DIGIT)
                    else: # not digit, moving on... could be ## !
                        ind += 1
                else:
                    ind += 1
            else:
                ind += len(nnc[ind:])
        if len(nums)>0:
            nums = np.max(nums)
        else:
            nums=0
        #print(nums)
        nout = list(nn).copy()
        nout.extend([nums])
        args_env.append(nout)
        
    for ia,a in enumerate(args_env):
        if a[-1]>0:
            # check for [NUM] and not from something like \def\command#1
            if '['+str(a[-1]) + ']' not in a[1] and '#' not in a[0] and '@' not in a[1]:
                if '\\def' not in a[1] and '{#' + str(a[-1]) + '}' not in a[1]: # sometimes \def\command{#(DIGIT)}
                    # is it just an add one?
                    if '[' + str(a[-1]+1) +']' not in a[1]:
                        if verbose:
                            print(a)
                            print(f)
                        # just add and hope its all OK
                        i2 = a[1].split('[')[-1].split(']')[0]
                        if verbose: print(i2)
                        args_env[ia][-1] = int(i2)        
                    else:
                        args_env[ia][-1] = a[-1]+1
                        
    # no real errors now, but could change
    if error_out:
        return args_env
    else:
        return args_env, err
    

def generate_find_replace_newcommands_old(args_new, arg_type = 'newcommand', verbose=False):
    """
    The purpose of this function (I think) is to loop and replace newcommands that 
     have a \begin or \end command with the actual text.  This helps with TexSoup 
     processing into environments.
    """
    # go through and find if there is begin/end in there
    # note: at this point all beginnings/endings should be fixed
    find_replace = []
    comments = []
    error = [False]
    #print('hi1')
    # print('args new at top:')
    # print(args_new)
    for ic,nc in enumerate(args_new):
        #print('nc=', nc)
        if nc[0] == 'error':
            continue
        n,fn,i1,i2,nArgs = nc
        #print('hi2')
        if nArgs == 0: # no arguements
            if '\\begin' in fn and not ('\\end' in fn):
                i = fn.index(n+'}') + len(n+'}')
                ind1,ind2,err = spc(fn[i:],
                                function='',dopen='{',
                                dclose='}',
                               error_out=False)
                if not err:
                    cmd = fn[i:][ind1+1:ind2-1]
                    find_replace.append((n,cmd,nArgs))
                else:
                    error = [True, 'error finding closing brackets for ' + arg_type]
                comments.append(fn)

            elif '\\end' in fn and not ('\\begin' in fn):
                i = fn.index(n+'}') + len(n+'}')
                ind1,ind2,err = spc(fn[i:],
                                function='',dopen='{',
                                dclose='}',
                               error_out=False)
                if not err:
                    cmd = fn[i:][ind1+1:ind2-1]
                    find_replace.append((n,cmd,nArgs))
                else:
                    error = [True, 'error finding closing brackets for '+arg_type]

                comments.append(fn)
            elif '\\def' in fn: # def statement
                pass
            elif '\\begin' in fn and '\\end' in fn: # have both
                # here

                if verbose: 
                    print('have both begin/end in a '+arg_type+', this is not supported')
                    print('(1) thingy is:', fn)
                    print('')
                #import sys; sys.exit()
                error = [True,'have both begin/end in a '+arg_type]
            else: # nothing to worry about
                pass
            #print('hi3')
        else: # has arguments -- only parse if have begin/end
            if '\\begin' in fn and not ('\\end' in fn):
                find_replace.append((n,fn,nArgs))
                comments.append(fn)  
            elif '\\end' in fn and not ('\\begin' in fn):
                find_replace.append((n,fn,nArgs))
                comments.append(fn)  
            elif '\\begin' in fn and '\\end' in fn: # have both
                if verbose: 
                    print('have both begin/end in a '+arg_type+', this is not supported')
                    print('(2) thingy is:', fn)
                    print('')
                #import sys; sys.exit()
                error = [True,'have both begin/end in a '+arg_type]
            else: # ignore
                pass

    # print('')
    # print("comments:")
    # print(comments)
    # print('find_replace:')
    # print(find_replace)
    # print('')
    return comments, find_replace, error



#**HERE** issue with begin/end ones
def generate_find_replace_newcommands(args_new, arg_type = 'newcommand', verbose=False):
    """
    The purpose of this function (I think) is to loop and replace newcommands that 
     have a \begin or \end command with the actual text.  This helps with TexSoup 
     processing into environments.
    """
    # go through and find if there is begin/end in there
    # note: at this point all beginnings/endings should be fixed
    find_replace = []
    comments = []
    error = [False]
    err = False
    #print('hi1')
    # print('args new at top:')
    # print(args_new)
    for ic,nc in enumerate(args_new):
        #print('nc=', nc)
        if nc[0] == 'error':
            continue
        # n = name of command
        # fn = LaTeX of new command
        # i1,i2 = where new command is defined
        # nArgs = number of arguments
        try:
            n,fn,i1,i2,nArgs = nc
        except Exception as e:
            if verbose:
                print('Met error in unpacking of args:', str(e))
                print('nc = ', nc)
                err = True
        if err:
            continue

        #print('hi2')
        if nArgs == 0: # no arguements
            if ('\\begin' in fn and not ('\\end' in fn)) or \
                 ('\\end' in fn and not ('\\begin' in fn)) or \
                    ('\\begin' in fn and '\\end' in fn):
                try:
                    i = fn.index(n+'}') + len(n+'}') # end of new command definition
                    err = False
                except Exception as e:
                    if verbose:
                        print('cant find brackets')
                        print(str(e))
                        error = [True, 'bracket finding for fn ' + arg_type]
                    err = True

                if not err:
                    # find replacement indecies
                    ind1,ind2,err = spc(fn[i:],
                                    function='',dopen='{',
                                    dclose='}',
                                error_out=False)
                    if not err:
                        cmd = fn[i:][ind1+1:ind2-1]
                        find_replace.append((n,cmd,nArgs))
                        # print("all ok: n, cmd")
                        # print(n)
                        # print(cmd)
                        # print('')
                    else:
                        error = [True, 'error finding closing brackets for ' + arg_type]
                comments.append(fn)
            # elif  ('\\begin' in fn and '\\end' in fn): # has both beginnings and endings
            #     # outputs are:
            #     # n = newcommand text (already found)
            #     # cmd = full input of new command (from fn)
            #     # nArgs = how many arguments there are (already found)
            #     print('1')
            #     print('n:', n)
            #     print('fn:', fn)
            #     print('nArgs:', nArgs)
            #     import sys; sys.exit()
            elif '\\def' in fn: # def statement
                pass
            else: # nothing to worry about
                pass
            #print('hi3')
        else: # has arguments -- only parse if have begin/end
            if ('\\begin' in fn and not ('\\end' in fn)) or \
                ('\\end' in fn and not ('\\begin' in fn)) or \
                    ('\\begin' in fn and '\\end' in fn):
                find_replace.append((n,fn,nArgs))
                comments.append(fn)  
                # print("all ok 2: n, cmd")
                # print(n)
                # print(fn)
                # print('')
            # elif ('\\begin' in fn and '\\end' in fn):
            #     # outputs are:
            #     # n = newcommand text (already found)
            #     # cmd = full input of new command (from fn)
            #     # nArgs = how many arguments there are (already found)
            #     print('2')
            #     print('n:', n)
            #     print('fn:', fn)
            #     print('nArgs:', nArgs)
            #     import sys; sys.exit()
            else: # ignore
                pass

    # print('')
    # print("comments:")
    # print(comments)
    # print('find_replace:')
    # print(find_replace)
    # print('')
    return comments, find_replace, error

    
    
def replace_newcommands_and_newenvironments(text, args_newcommands, args_newenvironments, 
                                            verbose = False, replace_comments = True):
    
    error = [False]
    warnings = []
    # for new commands, actually error out
    comments, find_replace, error = generate_find_replace_newcommands(args_newcommands, 
                                                               verbose=verbose,arg_type = 'newcommand')
    # for environments, just add to warnings
    comments_ne, find_replace_ne, error_ne = generate_find_replace_newcommands(args_newenvironments, 
                                                               verbose=verbose,arg_type = 'newenvironments')
    if error_ne[0]:
        warnings.append('NEW ENV: ' + error_ne[1])
        
    try:
        # check for inputs in environments, not supported
        for instr,outstr in find_replace_ne:
            if '#' in outstr:
                if verbose:
                    print('input to new environment, not supported')
                error = [True, 'input to new environment, not supported']
    except:
        if verbose:
            print('could not parse find/replace for new environment')
        error = [True, 'could not parse find/replace for new environment']
    
    
    if error[0]: # have an error, just return
        return '', error, warnings

    # find/replace -- require a whitespace after
    for instr, outstr,nArgs in find_replace: # newcommand to replace, replace with, # args
        if nArgs == 0: # now arguments
            ind = 0
            text_out = []
            search_cmd = re.escape(instr) + '(\\s{1,})'

            while ind < len(text):
                if re.search(search_cmd, text[ind:]):
                    istart,iend1 = re.search(search_cmd, text[ind:]).span()
                    # get the command specifically
                    _,iend = re.search(re.escape(instr),text[ind:][istart:iend1]).span()
                    #iend += iend1
                    i1 = istart
                    i2 = istart+iend
                    # make sure we are not in a command
                    inCommand = False
                    for n,fn,i1c,i2c,ncc in args_newcommands:
                        if i1+ind >= i1c and i2+ind <= i2c: # inside
                            inCommand = True

                    if not inCommand: # not in a command
                        text_out.append(text[ind:][:istart])
                        text_out.append(outstr)
                        ind += i2
                    else: # in command, move on
                        text_out.append(text[ind:][:i2])
                        ind += i2
                else: # not in there anymore
                    text_out.append(text[ind:])
                    ind += len(text[ind:])

            text = "".join(text_out)
        else: # has arguments -- need to parse these as well for replacement
            ind = 0
            text_out = []
            search_cmd = re.escape(instr) + r'(\s*){' # include bracket in search
            while ind < len(text):
                if re.search(search_cmd, text[ind:]): # start of command
                    istart,iend = re.search(search_cmd, text[ind:]).span()
                    args = {} # get all arguments and store their values
                    err = False
                    #if icount > 0: import sys; sys.exit()
                    for ia in range(nArgs): # loop through all expected arguments
                        # search for matching brakets, starting at starting {
                        ind1,ind2,err = spc(text[ind+iend-1:],
                                        function='',dopen='{',
                                        dclose='}',
                                       error_out=False)
                        if not err: # found it!
                            args[ia+1] = text[ind:][iend-1:][ind1+1:ind2-1]
                            # now we have to update everything for the next look at args
                            iend += ind2
                        else: # have an issue, carry on
                            if verbose: print('have issue finding {} for replacement args')
                            error = [True, 
                                     'have issue finding {} for replacement args in newcommands']
                            iend += 1
                            break
                        #import sys; sys.exit()
                        #if ia>0: import sys; sys.exit()
                    #import sys; sys.exit()
                    # replace
                    outstr_mod = outstr
                    for k,v in args.items():
                        outstr_mod = outstr_mod.replace('#'+str(k), v)
                    # now get just inside part
                    ind1,ind2,err = spc(outstr_mod[::-1],
                        function='',dopen='}',
                        dclose='{',
                       error_out=False)
                    if not err:
                        outstr_mod = outstr_mod[::-1][ind1+1:ind2-1][::-1]
                    else:
                        error = [True, 'error in parsing outstr in newcommand replacement']
                    inCommand = False
                    i1 = istart
                    i2 = iend-1
                    for n,fn,i1c,i2c,ncc in args_newcommands:
                        if i1+ind >= i1c and i2+ind <= i2c: # inside
                            inCommand = True

                    if not inCommand:      
                        text_out.append(text[ind:][:istart])
                        text_out.append(outstr_mod)
                    else:
                        text_out.append(text[ind:iend-1])
                    ind += iend-1
                    if verbose: 
                        print('args for:', outstr)
                        print(args)
                else:
                    text_out.append(text[ind:])
                    ind += len(text[ind:])
            text = "".join(text_out)
            
    if error[0]:
        return '', error, warnings
            
    # finally, replace comments
    if verbose: print('')
    if replace_comments:
        for c in comments:
            ind1 = text.index(c)
            ind2 = ind1 + len(c)
            text_before = text[:ind1]
            text_middle = text[ind1:ind2]
            text_after = text[ind2:]
            # update middle
            text_middle = '%' + text_middle
            # if any newlines, make sure all lines are commented
            if '\n' in text_middle:
                text_middle = text_middle.replace('\n', '\n%')
                # add new line at end?
                #text_middle += '\n'
            text = text_before + text_middle + text_after
            if verbose:
                print('----- before commenting ----')
                print(c)
                print('----- after commenting -----')
                print(text_middle)
                print('-----------------------------')
                print('')
            # text = text.replace(c, '%'+c)
            # # if newlines, make sure next lines are commented as well
            # if '\n' in text: 
            #     text.replace('\n', '\n%')
            #     # add newline at end
            #     text += '\n'
            if verbose:
                print(c, 'gets commented')

    return text, error, warnings



# for cleaning accents
# for cleaning up accents -- these will only work in NON-math mode!
def clean_accents_splits(main_body_here, verbose=True, return_is_accent_flag = False,
                 error_out = False, addone=1):
    """
    return_is_accent_flag : if set to true, just flags the start/stop 
                            index of the accent, does not fix
    """
    # all of these do the same:
    # Naz\H { h }
    # Naz\H{ h }
    # Naz\H{h}
    # Naz{ \H { h }} adds space at the beginning before the \H --> move this space outside
    # Naz{ \ H { h }} does *not* apply the accent
    # Naz\H{$\omega$} places accent *before* *whatever* is on the $$
    # Naz\H h works
    # Naz\Hh is an error
    # Naz\H 8 (i.e. with a number) works
    # Naz\H \n 8 also works (i.e with a newline)
    
    errOut = False

    # need to split into accents that are denoted by punctuation or not
    accentsNP = []; accentsP= []
    for a in accents:
        s = a.split('\\')[-1]
        if re.search('[A-Za-z]',s): # non-punct
            accentsNP.append(a)
        else:
            accentsP.append(a)
            
    if return_is_accent_flag:
        accent_inds_out = []

    # note \s is for ALL white spaces (space, tab, newline, \r, \t, \n)
    main_body_here2 = []
    imainbody = 0 # index for tracking throughout text
    for m,mt in main_body_here:
        if mt == '': # not labeled
            ind=0
            mout = ''
            while ind<len(m):
                found = False
                istart_min = 1e50; iEnd=-1; replace_min=''
                # --- not a letter accents -----
                for r in accentsP: # not a letter
                    # either: {[SPACES][ACCENT THAT IS NOT A LETTER][SPACES][LETTER/NUMBER][SPACES]} - { \' e }
                    ss = '{'+ '(\s*)'+re.escape(r) + '(\s*)[A-Za-z0-9](\s*)'+'}' 
                    if re.search(ss,m[ind:]):
                        istart,iend = re.search(ss,m[ind:]).span()
                        if istart < istart_min:
                            # any space after { and before \ --> move outside
                            replace = m[ind:][istart:iend]
                            ib1 = replace.index('{')
                            ib2 = replace.index('\\')
                            replace_min = replace[ib1+1:ib2] + ''.join(replace.split())
                            istart_min = istart; iEnd = iend

                    # or: {[SPACES][ACCENT THAT IS NOT A LETTER][SPACES]{[SPACES][LETTER/NUMBER][SPACES]}[SPACES]} 
                    #      - { \' { e } }
                    ss = '{'+ '(\s*)'+re.escape(r) + '(\s*){(\s*)[A-Za-z0-9](\s*)}(\s*)'+'}' 
                    if re.search(ss,m[ind:]):
                        istart,iend = re.search(ss,m[ind:]).span()
                        if istart<istart_min:
                            # any space after { and before \ --> move outside
                            replace = m[ind:][istart:iend]
                            # this grabs ONLY the first one
                            ib1 = replace.index('{')
                            ib2 = replace.index('\\')
                            replace_min = replace[ib1+1:ib2] + ''.join(replace.split())
                            istart_min = istart; iEnd = iend

                    # or: [ACCENT THAT IS NOT A LETTER][ANY SPACES][LETTER/NUMBER] - \'e or \' e
                    ss = ''+re.escape(r) + '(\s*)[A-Za-z0-9]' 
                    if re.search(ss,m[ind:]):
                        istart,iend = re.search(ss,m[ind:]).span()
                        if istart<istart_min:
                            # take out space between
                            replace = m[ind:][istart:iend]
                            replace_min = ''.join(replace.split())
                            istart_min = istart; iEnd = iend

                    # or: [ACCENT THAT IS NOT A LETTER][SPACES]{[SPACES][LETTER][SPACES]} - 
                    #.    - \'{o} or \'{ o } or \' {o}
                    ss = ''+re.escape(r) + '(\s*)' + '{' + '(\s*)[A-Za-z0-9](\s*)' + '}' 
                    if re.search(ss,m[ind:]):
                        istart,iend = re.search(ss,m[ind:]).span()
                        if istart<istart_min:
                            # take out space between
                            replace = m[ind:][istart:iend]
                            replace_min = ''.join(replace.split())
                            istart_min = istart; iEnd=iend

                # --- YES a letter accents -----
                for r in accentsNP: # a letter
                    # either: {[SPACES][ACCENT THAT IS A LETTER][>1 SPACES][LETTER/NUMBER][SPACES]} 
                    #   { \H e } --> {\H{e}}
                    ss = '{'+ '(\s*)'+re.escape(r) + '(\s{1,})[A-Za-z0-9](\s*)'+'}' 
                    if re.search(ss,m[ind:]):
                        istart,iend = re.search(ss,m[ind:]).span()
                        if istart<istart_min:
                            # any space after { and before \ --> move outside
                            replace = m[ind:][istart:iend]
                            ib1 = replace.index('{')
                            ib2 = replace.index('\\')
                            # replace space after accent mark with {} and remove spaces
                            replace = replace[ib1+1:ib2] + ''.join(replace.split())
                            # re-add in accent mark with {} around the letter right after
                            ib = replace.index(r)
                            replace_min = replace[:ib+len(r)] + '{'+replace[ib+len(r)]+'}'+replace[ib+len(r)+1:]
                            istart_min = istart; iEnd=iend

                    # or: {[SPACES][ACCENT THAT IS A LETTER][ANY SPACES]{[SPACES][LETTER/NUMBER][SPACES]}[SPACES]} 
                    #      - { \H { e } }
                    ss = '{'+ '(\s*)'+re.escape(r) + '(\s*){(\s*)[A-Za-z0-9](\s*)}(\s*)'+'}' 
                    if re.search(ss,m[ind:]):
                        istart,iend = re.search(ss,m[ind:]).span()
                        if istart<istart_min:
                            # any space after { and before \ --> move outside
                            replace = m[ind:][istart:iend]
                            # this grabs ONLY the first one
                            ib1 = replace.index('{')
                            ib2 = replace.index('\\')
                            replace_min = replace[ib1+1:ib2] + ''.join(replace.split())
                            istart_min = istart; iEnd=iend

                    # or: [ACCENT THAT IS A LETTER][>1 SPACES][LETTER/NUMBER]
                    #    \H e --> \H{e}
                    #print('hi')
                    ss = ''+re.escape(r) + '(\s{1,})[A-Za-z0-9]' 
                    if re.search(ss,m[ind:]):
                        istart,iend = re.search(ss,m[ind:]).span()
                        if istart<istart_min:
                            # take out space between
                            replace = m[ind:][istart:iend]
                            replace = ''.join(replace.split())
                            # re-add in accent mark with {} around the letter right after
                            ib = replace.index(r)
                            replace_min = replace[:ib+len(r)] + '{'+replace[ib+len(r)]+'}'+replace[ib+len(r)+1:]
                            istart_min = istart; iEnd=iend

                    # or: [ACCENT THAT IS A LETTER][SPACES]{[SPACES][LETTER][SPACES]} - \H{o} or \H{ o }
                    ss = ''+re.escape(r) + '(\s*)' + '{' + '(\s*)[A-Za-z0-9](\s*)' + '}' 
                    if re.search(ss,m[ind:]):
                        istart,iend = re.search(ss,m[ind:]).span()
                        if istart<istart_min:
                            # take out space between
                            replace = m[ind:][istart:iend]
                            replace_min = ''.join(replace.split())
                            istart_min = istart; iEnd=iend

                if len(replace_min)>0: # got something to replace!
                    #print('hi!')
                    #print('|'+m[ind:][istart_min:iEnd]+'|')
                    if verbose: 
                        old = m[ind:][istart_min:iEnd]
                        if old != replace_min:
                            print(old,'becomes',replace_min)
                    # update
                    istart = istart_min; iend = iEnd; replace = replace_min
                    if not return_is_accent_flag: # update if not returning!
                        mout += m[ind:][:istart]
                        mout += replace
                        try:
                            mout += m[ind:][iend]
                        except:
                            try:
                                mout += m[ind:][min(iend,len(m[ind:]))]
                                print('THIS MIGHT BE AN ISSUE!!!')
                            except: # really no idea
                                print('REALLY no idea in clean accents')
                            if error_out: 
                                import sys; sys.exit()
                            else: 
                                errOut = True
                    else:
                        mout = m[ind:ind+iend]
                        accent_inds_out.append(('{} accent', 
                                                istart+imainbody, 
                                                iend+imainbody, 
                                                m[ind+istart:ind+iend]))
                        imainbody += iend
                    ind += iend+addone 
                else: # no accents found in remaining string -- just increase
                    mout += m[ind:]
                    ind=len(m)
                    imainbody += len(mout)

            main_body_here2.append( (mout, mt) )
        else:
            main_body_here2.append( (m,mt) )

    # replace the whole shebang
    if not return_is_accent_flag:
        if error_out:
            return main_body_here2
        else:
            return main_body_here2, errOut
    else:
        if error_out:
            return accent_inds_out
        else:
            return accent_inds_out, errOut
        

# in *theory* only want to do in document portion alone? not sure yet...
def clean_accents(document, verbose = False):
    document_clean, err_accent = clean_accents_splits([[document,'']],verbose=verbose)
    return document_clean[0][0], err_accent
