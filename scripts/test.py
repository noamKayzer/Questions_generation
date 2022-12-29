try:
    import spacy
except:
    import spacy
abrv_nlp = spacy.load("en_core_sci_sm")

from scispacy.abbreviation import AbbreviationDetector
import re


# Add the abbreviation pipe to the spacy pipeline.
abrv_nlp.add_pipe("abbreviation_detector")


def replace_abbreviations(abrv_dict, text,only_first = False):
  if len(abrv_dict)==0:
    return text
  if only_first:
    only_first = 1
  else:
    only_first = 0
  if isinstance(text,list):
    return [replace_abbreviations(abrv_dict, cur_text,only_first) for cur_text in text]
  #`only_first` resolve only the first abbreviation in the text (e.g. if the text includes the results section, and the abbreviation was define at the introduction, the method will resolve once the abbreviation in the text the model is exposed to)
  for abrv,long_form_abbr in abrv_dict.items():
    text = text.replace(f'{long_form_abbr} ({abrv})',abrv) #first the function abbreviate all occurences
    text = text.replace(f'{long_form_abbr}({abrv})',abrv) 
    pattern = re.compile(r'(^|\s|\.)'+abrv+ r'( |[\W\s])')
    text = re.sub(pattern, f' {long_form_abbr} ({abrv})'+ r'\1', text, count=only_first)
  return text
'''
print(replace_abbreviations({'NYC':'New York City'},'Living in the NYC is diffuclt, the NYC people are stressed.'))
print(replace_abbreviations({'NYC':'New York City'},'Living in the NYC is diffuclt, the NYC people are stressed.',True))
print(replace_abbreviations({'NYC':'New York City'},'',True))
dd=replace_abbreviations({'NYC':'New York City'},[''],True)
print(len(dd),type(dd))
'''
name = 'COMET2'
JSON_path = '/home/ubuntu/Questions_generation/summery_example.json'
import json
with open(JSON_path) as f:
  result = json.load(f)
text = [s['original']['text'] if 'original' in s.keys() and 'text' in s['original'].keys() else None  for k,s in result['sections'].items() ]

import os
os.path.isfile("temp.txt")
doc = abrv_nlp.__call__('.\n'.join(text))
abrv_dict ={}
for abrv in doc._.abbreviations:
#print(f"{abrv} \t ({abrv.start}, {abrv.end}) {abrv._.long_form}")
    if abrv not in abrv_dict.keys():
        abrv_dict[str(abrv)] = str(abrv._.long_form)



print(replace_abbreviations(abrv_dict,text,True))


