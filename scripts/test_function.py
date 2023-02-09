#from tqdm.auto import tqdm
# %pdb on
import string 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
import re
import spacy
from collections import OrderedDict
from functools import partial
import pandas as pd
import numpy as np
import copy
#from sense2vec import Sense2Vec
import torch
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, OPTForCausalLM,AutoModelForSeq2SeqLM
except:
    #!pip install transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, OPTForCausalLM,AutoModelForSeq2SeqLM
from typing import List, Tuple
import itertools
from sklearn.metrics.pairwise import cosine_similarity
from operator import itemgetter
import nltk
nltk.download('punkt')
try:
    from sentence_transformers import SentenceTransformer 
except:
    #!pip install sentence_transformers
    from sentence_transformers import SentenceTransformer 
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

nltk.download('omw-1.4')


class distractors_by_autoregressive_model:
    def __init__(self,device='cpu',emb_model='all-MiniLM-L6-v2',
                 autoregressive_model_checkpoint = "facebook/galactica-1.3b"):
        #"facebook/galactica-1.3b" "facebook/galactica-6.7b" "EleutherAI/gpt-neo-1.3B"
        self.MODEL_EMBEDDING =  SentenceTransformer(emb_model)

        self.DEVICE = device
        self.USE_ANSWER_WITH_VARYING_SPANS = False
        self.max_letters_in_word = 20 # more than `self.max_letters_in_word` letters in word suspected to be non-word sequence. 

        '''
        checkpoint = "EleutherAI/gpt-neo-1.3B" #"gpt2" EleutherAI/gpt-neo-1.3B EleutherAI/gpt-neo-2.7B
        MODEL = AutoModelForCausalLM.from_pretrained(checkpoint).to(DEVICE)
        TOKENIZER = AutoTokenizer.from_pretrained(checkpoint)
        '''
        #autoregressive_model_checkpoint = "google/flan-t5-large"
        #self.model = AutoModelForSeq2SeqLM.from_pretrained(autoregressive_model_checkpoint).to(device)
        

        self.model =  OPTForCausalLM.from_pretrained(autoregressive_model_checkpoint).to(device)
        self.tokenizer =  AutoTokenizer.from_pretrained(autoregressive_model_checkpoint)

        lemmatizer = WordNetLemmatizer()
        self.lemma_get = lemmatizer.lemmatize
        torch.manual_seed(0)

    def get_answer_and_distractor_embeddings(self,answer,candidate_distractors):
        answer_embedding = self.MODEL_EMBEDDING.encode([answer])
        distractor_embeddings = self.MODEL_EMBEDDING.encode(candidate_distractors)
        return answer_embedding, distractor_embeddings

    def remove_prefix(self,text, prefix):
        return text[len(prefix):].strip()
        #return text.replace(prefix.strip(),'').strip()


    def only_alphanumeric_punctuation(self,text):
        match = re.search("^[\w\s\d\.,%=+:\(\)-]+$", text)
        return bool(match)
    
    def no_too_long_word(self,text):
        match = [len(word) > self.max_letters_in_word for word in text.split()]
        return any(match)
    
    def filter_outputs(self,outputs, prefix, answer):
        outputs_filtered = []
        answer_len = len(answer.split())

        for output in outputs:
            output = self.remove_prefix(output, prefix)
            if "\nAuthors" in output: # for Galactica model
                output = output[:output.find('\nAuthors')]
            output = output.replace("e.g.","eg").replace("i.e.","ie").replace("i. e.","ie").replace("e. g.","eg")
            
            if "." not in output:
                continue
            output_split = output.split(". ")  # consider take all sentences until the last dot. maybe depend on the answers length.

            sentence_len = [len(out.split()) for out in output_split]
            max_sentence_taken_idx = np.argmin(np.abs(np.cumsum(sentence_len)-answer_len))        
            output_cut = ". ".join(output_split[:max_sentence_taken_idx+1])+"."
            #print(output_cut)
            # if not self.only_alphanumeric_punctuation(output_cut):    
            #     continue
            if not self.no_too_long_word(output_cut):
                 continue
            #if len(self.tokenizer.encode(output_cut)) > 2:
            outputs_filtered.append(output_cut.strip().replace("\n"," "))

        return outputs_filtered

    def generate_distractors(self, question, answer,title=False,prefix=False):
        if self.USE_ANSWER_WITH_VARYING_SPANS:
            if len(answer.split()) <= 3:
                num_words_to_add = (1,2)
            elif len(answer.split()) <= 5:
                num_words_to_add = (1,3)
            elif len(answer.split()) <= 7:
                num_words_to_add = (2,4)
            elif len(answer.split()) <= 10:
                num_words_to_add = (1,3,5)
            else: 
                num_words_to_add = (2,4,6)
            num_return_sequences = 10
        else:

            punct_or_stopwords_or_in_question = set([*stopwords.words('english'), *string.punctuation, *[self.lemma_get(qe.lower()) for qe in question.replace('?','').split()]])

            num_words_to_add=[1]
            for word in answer.split()[1:]:
                if self.lemma_get(word.lower()) in punct_or_stopwords_or_in_question:
                    num_words_to_add[0]+=1
                else:
                    break
            if " ".join(answer.split()[:num_words_to_add[0]]).strip().lower() in ['a','an','the']:
                num_words_to_add[0]+=1
            if len(answer.split())==num_words_to_add[0]:
                # if all answer was selected as the prefix or by :
                #       (1) The answer is only one word 
                #       (2) All the words from the question are in the answer.
                #       The distractors cannot be made.
                return False 
            num_return_sequences = 25
        candidate_distractors = []
        if not prefix:
            if title:
                prefix = title +". "+question +" " # "Title: "+title+". "+question + " "
                #prefix = "Title:"+ title +". Answer to the question: "+question  # "Title: "+title+". "+question + " "
                #prefix = "Title:"+ title +". </s> Answer to the question: "+question+" </s> "  # "Title: "+title+". "+question + " "
                #prefix = " Answer to the question: "+question+"</s> Context: "+ title +".</s>"  # "Title: "+title+". "+question + " "
            else:
                prefix = "Answer to the question: " + question 
            '''
            if title:
                prefix = "Answer to the question: "+question + " Context: "+title +" " # "Title: "+title+". "+question + " "
            else:
                prefix = "Answer to the question: " + question + " "
            '''
        if isinstance(prefix,list):
            prefix_list = prefix
        else:
            prefix_list = [prefix]
        #when `self.USE_ANSWER_WITH_VARYING_SPANS` is False, there is only one element in `num_words_to_add` so the for is with one iteration
        for num in num_words_to_add:
            for prefix in prefix_list:
                prompt =  prefix + " ".join(answer.split()[:num])+" "
                print(prompt)
                input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.model.device)
                
                # output_ids = self.model.generate(
                #     input_ids, 
                #     do_sample = True, 
                #     min_length = len(self.tokenizer.encode(prompt)) + 5,
                #     max_length = len(self.tokenizer.encode(prompt)) + 25+25,
                #     top_p = 0.94, # 0.8 
                #     top_k = 30,   #30
                #     repetition_penalty  = 2.0, # 10.0,
                #     temperature = 2.0,
                #     num_return_sequences = num_return_sequences,
                #     #early_stopping= True
                # ).cpu()
                
                output_ids = self.model.generate(input_ids,max_new_tokens=100,repetition_penalty  = 10.0, diversity_penalty=float(10), # top_k = 20,do_sample=True,
                               num_beams=5,num_return_sequences=5,length_penalty=-2,num_beam_groups=5,early_stopping=True)
                outputs = [self.tokenizer.decode(output, skip_special_tokens=True) for output in output_ids]
                outputs = [ dist[:dist.find('\n\n')] 
                           if dist.find('\n\n') else dist
                           for dist in outputs ]
                print(outputs) 
                candidate_distractors += self.filter_outputs(outputs, prefix,answer)
        if len(candidate_distractors)==0:
            print('no candidate_distractors pass filtring')
            return False
        answer_embedding, distractor_embeddings = self.get_answer_and_distractor_embeddings(answer, candidate_distractors)
        distractors = self.mmr(answer_embedding, distractor_embeddings, candidate_distractors, reverse=True, top_n=5, diversity=0.8)
        distractors = candidate_distractors
        if distractors: # remove multipile spaces
            distractors = [' '.join(d.split()) for d in distractors]
        print('+'*30)
        print(f'{question} ----- {answer}')
        for i,d in enumerate(distractors):
            print(f'({i})-{d}\n')
        return distractors

    def mmr(self,
        doc_embedding: np.ndarray,
        word_embeddings: np.ndarray,
        words: List[str],
        top_n: int = 5,
        diversity: float = 0.8,
        reverse: bool = False
    ) -> List[Tuple[str, float]]:

        keywords_idx = [np.argmin(cosine_similarity(word_embeddings, doc_embedding))]
        doc_embedding = word_embeddings[keywords_idx[0]].reshape(1, -1)
        # word_embeddings = [emb for idx,emb in enumerate(word_embeddings) if idx not in keywords_idx]
        word_similarity = cosine_similarity(word_embeddings)
        word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
        

        candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]
        # print(words[keywords_idx[0]])

        for _ in range(min(top_n - 1, len(words) - 2)):
            # Extract similarities within candidates and
            # between candidates and selected keywords/phrases
            candidate_similarities = word_doc_similarity[candidates_idx, :]
            target_similarities = np.max(
                word_similarity[candidates_idx][:, keywords_idx], axis=1
            )

            # Calculate MMR
            mmr = (1 - diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
            mmr_idx = candidates_idx[np.argmax(mmr)]

            # Update keywords & candidates
            keywords_idx.append(mmr_idx)
            candidates_idx.remove(mmr_idx)

        # Extract and sort keywords in descending similarity
        keywords = [
            (words[idx], round(float(word_doc_similarity.reshape(1, -1)[0][idx]), 4))
            for idx in keywords_idx
        ]
        keywords = sorted(keywords, key=itemgetter(1), reverse=True)
        return [kw[0] for kw in keywords]

if 'di' in locals():
    del di
di = distractors_by_autoregressive_model('cuda')
title ="The role of semantics in the success of crowdfunding projects"
qs1 = "What do the authors analyze?" 
answer1 = "A large dataset of crowdfunding project data"



qs2 ="On what dataset was the proposed model evaluated?"
answer2 = "The proposed models were evaluated on the following datasets: All_D."

qs3 = "What should the post of a project contain?"
answer3 = "more feelings words"

qs4 = "What did the authors use Beautifulsup to gain?"
answer4 = "additional metadata features"

qs5 = "What categories of buzzwords were used in the study?"
answer5 = 'General conversation, education, business, sales and marketing, science and technology, politics and current affairs. The buzzword dataset used in this study contains words from different categories: general conversation, business.'
for qs,answer in zip([qs1,qs2,qs3,qs4,qs5],[answer1,answer2,answer3,answer4,answer5]):
    print()
    
    dist = di.generate_distractors( qs, answer,title=title)
    print('-'*100)
    print(qs,'---',answer)
    print(dist)
    print('-'*100)