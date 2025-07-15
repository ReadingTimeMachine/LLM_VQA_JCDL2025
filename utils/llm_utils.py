import numpy as np
from PIL import Image
import base64

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


def load_image(image_path, tmp_dir = '/Users/jnaiman/Downloads/tmp/', fac=1.0):
    img = Image.open(image_path).convert('RGB')
    new_size = np.round(np.array(img.size)*fac).astype('int')
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    #img = np.array(img)
    #with open(image_path, "rb") as image_file:
    img.save(tmp_dir + 'tmp_img.png')
    with open(tmp_dir +'tmp_img.png','rb') as image_file:
        #return base64.b64encode(img).decode("utf-8")
        return base64.b64encode(image_file.read()).decode("utf-8")