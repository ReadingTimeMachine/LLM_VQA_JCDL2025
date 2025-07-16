## Example Histograms with VQA

### Directories
* `example_hists/imgs` stores the example images (right now there are PDF and JPEG formats)
* `example_hists/jsons` stores the jsons


### The Data

#### Basics

To load json (after setting `json_dir` to where the jsons are stored):
```python
import json
data_file = json_dir + 'nclust_4_trial8.json'
with open(data_file,'r') as f:
    t = json.load(f)
    datas = json.loads(t)
```

VQA can be accessed with the `VQA` tag. For example, `datas['VQA']` prints out the same `Q` and `A` pairs as before:
```python
{'Level 1': {'Figure-level questions': {},
  'Plot-level questions': {'nbars ': {'plot0': {
    'Q': 'How many bars are there on the figure? You are a helpful assistant, please format the output as a json as {"nbars":""} for this figure panel, where the "nbars" value should be an integer.',
     'A': {'nbars ': 50}}}}},
 'Level 2': {'Plot-level questions': {'minimum (plot numbers)': {'plot0': {
    'Q': 'What are the minimum data values in this figure? You are a helpful assistant, please format the output as a json as {"minimum x":""} where the minimum value of "x" is calculated from  the data values used to create the plot in the format of floats.  ',
     'A': {'minimum (plot numbers)': {'minimum x': 0.37513605039268844}}}},
   'minimum (words)': {'plot0': {'Q': 'What are the minimum data values in this figure? You are a helpful assistant, please format the output as a json as {"minimum x":""} where the minimum value of "x" is calculated from  the data values used to create the plot in the format of floats.  ',
    ...
        }
      }
    }
  }
}
```

#### Additional question breakdowns

Additionally, there are several extra tags for a breakdown of the question following [Microsoft's Elements of a Good Prompt slide](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://download.microsoft.com/download/e/8/7/e871d21a-dd67-4d73-b5ad-b54cbb9d6e29/English-UK-Elements-of-a-Good-Prompt.pdf&ved=2ahUKEwjI66683sGOAxWEMtAFHahlPecQFnoECBcQAQ&usg=AOvVaw3vtdFV_vMdgv33JwEfhgpI) which are at the same "level" as the `A` and `Q` of the json example above.

![Elements of a good prompt by microsoft include persona (ask the tool to take a role), objective (what do you want the AI to do), audience (specify who its for), context (what does the tool need to know), boundaries (set your own direction & limitations)](../resources/docs/elements_of_a_good_prompt.png)
* `persona` -- who the AI is pretending to be, along the lines of "you are a helpful assistant...", this is often the "system" prompt when passing to APIs like ChatGPT/Claude
* `context` -- what does the tool need to know, for single plots like these single example histograms this is empty, but for multi-panel plots, the layout of the figure will be specified and the specific plot pointed to
* `question` (Microsoft's "objective") -- the actual question (e.g., "what is the mean of the distribution?")
* `format` (Microsoft's "boundaries") -- specify the output format, something like "export as a json with {"nbars":""}..."


#### Question "Levels"

Here "levels" refer to the 4-level paradigm developed in [Accessible Visualization via Natural Language Descriptions: A Four-Level Model of Semantic Content](http://vis.csail.mit.edu/pubs/vis-text-model/).

Like the prior iteration, there are different "levels" of parsing the plots, as well as figure-level and plot-level questions, assuming each figure object can be made up of multiple plot axes objects (right now, there is a single axes, so just "plot0" for everything).

For example, to access the plot-level, Level 1 questions:
```python
datas['VQA']['Level 1']['Plot-level questions']
```
prints out:
```python
{'nbars ': {'plot0': {'Q': 'You are a helpful assistant that can analyze images.  How many bars are there in the specified figure panel? Please format the output as a json as {"nbars":""} for this figure panel, where the "nbars" value should be an integer.',
   'A': {'nbars ': 50},
   'persona': 'You are a helpful assistant that can analyze images.',
   'context': '',
   'question': 'How many bars are there in the specified figure panel?',
   'format': 'Please format the output as a json as {"nbars":""} for this figure panel, where the "nbars" value should be an integer.'}}}
```

## Example LLM Outputs

Various example outputs from LLMs will be in the `LLM_output` subfolder in this directory.