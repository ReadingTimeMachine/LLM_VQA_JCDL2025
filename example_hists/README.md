## Example Histograms with VQA

### Directories
* `example_hists/imgs` stores the example images (right now there are PDF and JPEG formats)
* `example_hists/jsons` stores the jsons


### The Data


To load json (after setting `json_dir` to where the jsons are stored):
```python
import json
data_file = json_dir + 'nclust_4_trial8.json'
with open(data_file,'r') as f:
    t = json.load(f)
    datas = json.loads(t)
```

VQA can be accessed with the `VQA` tag. For example, `datas['VQA']` prints out:
```python
{'Level 1': {'Figure-level questions': {},
  'Plot-level questions': {'nbars ': {'plot0': {'Q': 'How many bars are there on the figure? You are a helpful assistant, please format the output as a json as {"nbars":""} for this figure panel, where the "nbars" value should be an integer.',
     'A': {'nbars ': 50}}}}},
 'Level 2': {'Plot-level questions': {'minimum (plot numbers)': {'plot0': {'Q': 'What are the minimum data values in this figure? You are a helpful assistant, please format the output as a json as {"minimum x":""} where the minimum value of "x" is calculated from  the data values used to create the plot in the format of floats.  ',
     'A': {'minimum (plot numbers)': {'minimum x': 0.37513605039268844}}}},
   'minimum (words)': {'plot0': {'Q': 'What are the minimum data values in this figure? You are a helpful assistant, please format the output as a json as {"minimum x":""} where the minimum value of "x" is calculated from  the data values used to create the plot in the format of floats.  ',
     'A': {'minimum (words)': {'minimum x': 0.37513605039268844}}}},
   'maximum (plot numbers)': {'plot0': {'Q': 'What are the maximum data values in this figure? You are a helpful assistant, please format the output as a json as {"maximum x":""} where the maximum value of "x" is calculated from  the data values used to create the plot in the format of floats.  ',
     'A': {'maximum (plot numbers)': {'maximum x': 0.5230372521302568}}}},
   'maximum (words)': {'plot0': {'Q': 'What are the maximum data values in this figure? You are a helpful assistant, please format the output as a json as {"maximum x":""} where the maximum value of "x" is calculated from  the data values used to create the plot in the format of floats.  ',
     'A': {'maximum (words)': {'maximum x': 0.5230372521302568}}}},
   'median (plot numbers)': {'plot0': {'Q': 'What are the median data values in this figure? You are a helpful assistant, please format the output as a json as {"median x":""} where the median value of "x" is calculated from  the data values used to create the plot in the format of floats.  ',
     'A': {'median (plot numbers)': {'median x': 0.43159559265881575}}}},
   'median (words)': {'plot0': {'Q': 'What are the median data values in this figure? You are a helpful assistant, please format the output as a json as {"median x":""} where the median value of "x" is calculated from  the data values used to create the plot in the format of floats.  ',
     'A': {'median (words)': {'median x': 0.43159559265881575}}}},
   'mean (plot numbers)': {'plot0': {'Q': 'What are the mean data values in this figure? You are a helpful assistant, please format the output as a json as {"mean x":""} where the mean value of "x" is calculated from  the data values used to create the plot in the format of floats.  ',
     'A': {'mean (plot numbers)': {'mean x': 0.43715784737441477}}}},
   'mean (words)': {'plot0': {'Q': 'What are the mean data values in this figure? You are a helpful assistant, please format the output as a json as {"mean x":""} where the mean value of "x" is calculated from  the data values used to create the plot in the format of floats.  ',
     'A': {'mean (words)': {'mean x': 0.43715784737441477}}}}}},
 'Level 3': {'Plot-level questions': {'gmm ngaussians (plot numbers)': {'plot0': {'Q': 'How many gaussians have been used to generate the histogram in this figure? You are a helpful assistant, please format the output as a json as {"gmm ngaussians":""}}. Where the "ngaussians" parameter should be the number of gaussians used to generate the histogram and should be an integer. ',
     'A': {'gmm ngaussians (plot numbers)': 4}}}}}}
```

Like the prior iteration, there are different "levels" of parsing the plots, as well as figure-level and plot-level questions, assuming each figure object can be made up of multiple plot axes objects (right now, there is a single axes, so just "plot0" for everything).

For example, to access the plot-level, Level 1 questions:
```python
datas['VQA']['Level 1']['Plot-level questions']
```
prints out:
```python
{'nbars ': {'plot0': {'Q': 'How many bars are there on the figure? You are a helpful assistant, please format the output as a json as {"nbars":""} for this figure panel, where the "nbars" value should be an integer.',
   'A': {'nbars ': 50}}}}
```

## Example LLM Outputs

Various example outputs from LLMs will be in the `LLM_output` subfolder in this directory.