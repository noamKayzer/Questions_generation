try:
    import spacy
except:
    import spacy
from negspacy.negation import Negex
#!pip install spacy
#!python3 -m spacy download en_core_web_sm

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("negex", config={"ent_types":["PERSON","ORG"]})

doc = nlp("Aren’t you coming? Doesn’t he understand? Are you not coming? Does he not understand? Which of the following did not occur? Does Jeff is a real person? Name something David hadn't reveal in his vacation?")
#doc = nlp(" Which of the following did not occur?")
for e in doc.ents:
    print(e.text, e._.negex)

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)

print(f"NEG:{any([token.dep_=='neg' for token in doc])}")
print('='*20)
for chunk in doc.noun_chunks:
    print(chunk.text, chunk.root.text, chunk.root.dep_,
            chunk.root.head.text)
print('='*20)
for token in doc:
    print(token.text, token.dep_, token.head.text, token.head.pos_,[child for child in token.children])
import spacy
from spacy import displacy

text = "When Sebastian Thrun started working on self-driving cars at Google in 2007, few people outside of the company took him seriously."

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
displacy.serve(doc, style="ent")