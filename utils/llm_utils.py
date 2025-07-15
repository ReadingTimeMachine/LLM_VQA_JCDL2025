import numpy as np
from PIL import Image
import base64
import json
import os

# parsing
def parse_qa(level_parse, plot_level, qa, j, types, 
             partials = ['persona', 'context','question', 'format']):
    keys_tmp = list(j[level_parse][plot_level].keys())
    keys = []
    for k in keys_tmp:
        if '(' in k:
            k = k.split('(')[0].rstrip()
        keys.append(k)
    
    keys = np.unique(keys).tolist()

    dirs_partials = {}
    
    for k in keys:
        v = ''
        kk = ''
        for t in types:
            if k + " " + t in j[level_parse][plot_level]:
                v = j[level_parse][plot_level][k + " " + t]
                #kk = k + " " + t
                break
        if v == '':
            v = j[level_parse][plot_level][k]
            #kk = k
        if 'A' in v: # no plot
            if type(v['A']) == type({}):
                ans = list(v['A'].values())[0]
            else:
                ans = v['A']
            if type(v['Q']) == type({}):
                que = list(v['Q'].values())[0]
            else:
                que = v['Q']
            # get other elements
            for p in partials:
                dirs_partials[p] = v[p]
        else: # plotX
            for kk,vv in v.items():
                ans = vv['A']
                while type(ans) == type({}):
                    ans_1 = list(ans.values())[0]
                    if 'plot' in list(ans.keys())[0]:
                        ans = ans_1
                        break
                    ans = ans_1
                que = vv['Q']
                # get other elements
                for p in partials:
                    dirs_partials[p] = vv[p]
        out = {'Q':que, 'A':ans, 'Level':level_parse, 'type':plot_level, 'Response':""}
        for kp,vp in dirs_partials.items():
            out[kp] = vp
        # if there is a plot
        if 'plot' in list(v.keys())[0]:
            out['plot number'] = list(v.keys())[0]
            #import sys; sys.exit()
        qa.append(out)
    return qa


def load_image(image_path, tmp_dir = '/Users/jnaiman/Downloads/tmp/', fac=1.0, 
               return_image_format = False, 
               img_format='png'):
    """
    Encode image for passing to various LLMs.

    img_format : the format of the encoded image
    """
    # check options for image format
    if img_format not in ['png', 'jpeg', 'gif']:
        img_format = 'png'
    img = Image.open(image_path).convert('RGB')
    new_size = np.round(np.array(img.size)*fac).astype('int')
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    #img = np.array(img)
    #with open(image_path, "rb") as image_file:
    img.save(tmp_dir + 'tmp_img.'+img_format)
    if not return_image_format:
        with open(tmp_dir +'tmp_img.'+img_format,'rb') as image_file:
            #return base64.b64encode(img).decode("utf-8")
            return base64.b64encode(image_file.read()).decode("utf-8")
    else:
         with open(tmp_dir +'tmp_img.'+img_format,'rb') as image_file:
            #return base64.b64encode(img).decode("utf-8")
            return base64.b64encode(image_file.read()).decode("utf-8"), img_format  


def get_img_json_pair(img_path, json_path, dir_api, 
                      tmp_dir = '/Users/jnaiman/Downloads/tmp/',
                      fac = 1.0, 
                      img_format = 'png',
                      return_image_format = True,
                      restart = False, verbose = True):
    """
    img_path : where image file is stored
    json_path : where json path is stored
    dir_api : where we can look for prior, saved pickles, if applicable
    fac : do we want to downsize the image? IF so, set to < 1

    returns: encoded image, full json from creation run, error
      encoded image and full json are empty strings if error is True
    """
    #print('on', iFile, 'of', iMax)
    err = False
    base_file = json_path.split('/')[-1].removesuffix('.json')
    if os.path.exists(dir_api + base_file + '.pickle') and not restart:
        if verbose: print('have file already:', dir_api + base_file + '.pickle')
        return '','', True
    # do we have it?
    try:
        #image_path = '/Users/jnaiman/Downloads/data_full_v2/Picture'+str(int(iFile))+'.png'
        encoded_image, img_format = load_image(img_path, fac=fac, tmp_dir=tmp_dir, 
                                               img_format=img_format, 
                                               return_image_format=return_image_format)
        # get questions
        with open(json_path,'r') as f:
            j = json.loads(f.read())
            j = json.loads(j)
    except Exception as e:
        if verbose: print('[ERROR]:', str(e))
        err = True
        return '','', True
    
    return encoded_image, img_format, j, err


def parse_for_errors(qa, llm='chatgpt',
                     verbose=True, print_what = 'prompt'):
    """
    print_what : set to "prompt" to print full prompt, or "question" for just the question
    """
    # try to fix
    for level in qa:
        noErr = False
        if 'Error' in level:
            noErr = True
            try:
                iters = []
                for x in re.finditer(r'//(\s*)(.*)(\s*)\n', level['Response']):
                    #print(x)
                    iters.append(level['Response'][x.span()[0]:x.span()[1]])
                
                response = level['Response']
                for i in iters:
                    response = response.replace(i,'')
                level['Response'] = response
                del level['Error']
            except:
                noErr = False
                print('********')
                print('couldnt fix:', level['Response'])
                print('********')
                pass
    
        try:
            if type(level['Response']) != type({}):
                level['Response'] = json.loads(level['Response'])
        except:
            pass
    
        print_llm = 'LLM'
        if llm.lower()=='chatgpt':
            print_llm = 'ChatGPT'
        elif 'claude' in llm.lower():
            print_llm = 'Claude'
        if not noErr and verbose:
            if print_what == 'question':
                print('Q:', level['Q'])
            elif print_what == 'prompt':
                print('Q:', level['prompt'])
            print(print_llm + ' A:', level['Response'])
            print('Real A:   ', level['A'])

            print('')
            #if 'Error' in level:
            #    import sys; sys.exit()
    return qa