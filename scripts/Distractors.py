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
from sense2vec import Sense2Vec
from distractors_with_GPTneo import generate_distractors

common_answers_prefix = {'according to', 'as far as','as for','as regards','as to',
                        'based on','concerning','in answer to','in connection with',
                        'in consideration of','in light of','in occurrence of','in order to',
                        'in reference to','in regard to','in regards to','in relation to',
                        'in response to','in terms of','in the case of','in the matter of',
                        'in view of','on account of','on the grounds of','pertaining to',
                        'relating to','with reference to','with regard to','with regards to','with respect to'}
be_words_antonyms_list = {'am not': 'am', 'is not': 'is', 'are not': 'are', 'was not': 'was', 'were not': 'were', "haven't been": 'have been', "hasn't been": 'has been', "hadn't been": 'had been', 'will not be': 'will be', 'would not be': 'would be', 'should not be': 'should be', 'could not be': 'could be', 'must not be': 'must be', 'might not be': 'might be', "don't": 'do', 'does not': 'does', 'did not': 'did', 'has not': 'has', 'have not': 'have', 'had not': 'had', 'can not': 'can', 'could not': 'could', 'shall not': 'shall', 'should not': 'should', 'will not': 'will', 'would not': 'would', 'may not': 'may', 'might not': 'might', 'must not': 'must', 'ought not': 'ought', 'need not': 'need', 'dare not': 'dare', 'used not': 'used', "needn't": 'need', "daren't": 'dare', "mustn't": 'must', "shouldn't": 'should', "wouldn't": 'would', "couldn't": 'could', "wasn't": 'was', "weren't": 'were', "isn't": 'is', "aren't": 'are', "didn't": 'did', "doesn't": 'does', "hasn't": 'has', "haven't": 'have', "hadn't": 'had', "can't": 'can', "we're not": "we're", "they're not": "they're", "you're not": "you're", "i'm not": "i'm", "he's not": "he's", "she's not": "she's",
 "it's not": "it's", 'we are not': "we're", 'they are not': "they're", 'you are not': "you're", 'i am not': "i'm", 'he is not': "he's", 'she is not': "she's", 'it is not': "it's",
 'not':'',"doesn't":"does","don't":"do","dont":"do","doesn't":"does","didn't":"did",}

class Distractors:
    def __init__(self,mcq_class,original_text,doc=None,nlp=None,sentence_model=None):
        '''
        `Distractors` class find distractors for question + answer pairs. 
                        the distractors based on the type of answers or questions.
                
                distractors based on few techniques:
                1. distractors for negative questions
                
        '''
        
        
        self.mcq_class = mcq_class
        if not sentence_model:
                from sentence_transformers import SentenceTransformer
                smodel = SentenceTransformer('all-MiniLM-L6-v2')
        self.smodel = smodel
        if not nlp:
            nlp = spacy.load("en_core_web_lg")
        self.original_doc = nlp(" ".join(original_text))
        self.nlp = nlp
        self.original_np_list = [noun_chunk.text for noun_chunk in self.original_doc.noun_chunks]
        self.smodel_embeddings = self.smodel.encode(self.original_np_list)
        self.smodel_embeddings_dict = {k:embedding for k,embedding in
                                       zip(self.original_np_list,self.smodel_embeddings)}

        self.n_outputs = 4 #number of distractors per questions

        self.s2v = model_loader(
            'sense2vec',
            "s2v",
            lambda : Sense2Vec().from_disk("/home/ubuntu/Questions_generation/s2v_reddit_2019_lg") # large 
        )
             
    def generate_distractors(self,questions_sections,title=False):
        for section in questions_sections:
            section_distractors =[]
            for q,a,details,context in zip(section['question'], section['answer'],section['details'],section['context']):
                a = self.mcq_class.solve_abrv_func(a,target_form='long')
                if q:
                    print('-'*50)
                    distractors = False
                    if 'NEGATIVE' in details:
                        distractors = self.distractors_by_neg_questions(q, a, details, context)
                    if distractors == False:    
                        distractors = self.distractors_by_ents(q, a, self.original_doc)
                        if not distractors:
                            distractors = self.distractors_by_np(q, a, self.original_doc)
                            if distractors:
                                print('NP')
                            '''else:
                                context = [cur[1] for cur in text_full if q.replace('?','').strip().lower() in cur[0].lower()][0]
                                distractors = self.distractors_by_question_np_jittering(q,self.original_doc,context)
                                if distractors:
                                    print('question jitter')'''
                                    
                    if not distractors:
                        distractors = self.distractors_by_autoregressive_model(q, a,title, self.original_doc)
                    
                    distractors = [self.mcq_class.solve_abrv_func(d,target_form='all') for d in distractors] \
                                        if distractors!= False else False
                    if distractors != False:
                        print(f'{q} --- {a}\n {distractors}')
                    else:
                        print(f'distractors not found! {q}')
                    section_distractors += [distractors]
            section['distractors'] =  section_distractors
        return questions_sections
                    
    def distractors_by_neg_questions(self, question, answer,details, context):
        for detail in details.split('.'):
            if 'NEGATIVE' in detail:
                neg_token = detail.replace('NEGATIVE:','').strip().lower()
            elif 'ANSWER' in detail:
                answer_type = detail.replace('ANSWER:','').strip().lower()
                

        if neg_token.lower() in be_words_antonyms_list.keys():
            pos_form = be_words_antonyms_list[neg_token.lower()]
            #print(neg_token,pos_form)
        pos_question = question.replace(neg_token,pos_form)
        #print(question,pos_question)
        ans_args,short_ans_args = self.mcq_class.answers_generator_args,self.mcq_class.short_answers_generator_args
        temp_answers_generator_args = copy.deepcopy(ans_args)
        temp_short_answers_generator_args = copy.deepcopy(short_ans_args)
        for args in [ans_args,short_ans_args]:
            args['top_p' ] = 0.93
            args['diversity_penalty'] = float(12)

        ans_output, short_ans_output = self.mcq_class.generate_answers( pos_question, context, self.mcq_class.model,verbose=False)
        ans_args,short_ans_args = temp_answers_generator_args, temp_short_answers_generator_args
        if answer_type == 'long':
            distractors = ans_output
        elif answer_type == 'short':
            distractors = short_ans_output
        if distractors == ['']:
            return False
        distractors = self.check_distractors(distractors, answer)
        if len(distractors) >= self.n_outputs:
                distractors = distractors[:self.n_outputs]
        distractors = filter_distractors(answer, distractors)

        return distractors
    
    def distractors_by_ents(self, question, answer, doc):
        answer_entity = self.only_ents(question, answer)
        if answer_entity:
            if not answer_entity.has_vector:
                return self.distractors_by_np(question, answer,doc) # if the entity vector hasn't found, we will use the similarity based on sentence model, as used for noun phrases  
            if answer_entity.label_ in ['DATE','TIME']:
                sim_ents = self.get_sim_ents(answer_entity,n_output=500,thrs=[0.3,0.93])
                distractors = self.filter_date_distractors(answer_entity,answer_entity.label_,[ent for ent in list(sim_ents.keys())])
                if len(distractors) < self.n_outputs:
                    distractors += self.find_distractors_s2v(answer_entity.text+'|'+answer_entity.label_,70)
                    distractors = self.filter_date_distractors(answer_entity.text,answer_entity.label_,distractors)

            else:
                sim_ents = self.get_sim_ents(answer_entity,n_output=20,thrs=[0.3,0.9])
                distractors = filter_distractors(answer_entity.text, [ent for ent in list(sim_ents.keys())])
                self.check_distractors(distractors, answer_entity.text)
                if len(distractors) < self.n_outputs:
                    distractors += self.find_distractors_s2v(answer_entity.text+'|'+answer_entity.label_,10)
                    distractors = filter_distractors(answer_entity.text,distractors)
                   
            distractors = self.check_distractors(distractors, answer_entity.text)
            if len(distractors) < self.n_outputs:
                # if the answer contains entity with number 3 minutes, we can split the number and find distractors based on the number, and than add the second part 
                distractors = self.distractors_by_number(question,answer_entity.text,doc,distractors)
            distractors = self.check_distractors(distractors, answer_entity.text)
            if len(distractors) >= self.n_outputs:
                distractors = distractors[:self.n_outputs]
            distractors = self.check_distractors(distractors, answer_entity.text)
            return [re.compile(re.escape(answer_entity.text),re.IGNORECASE).sub(d,answer) for d in distractors]
         
            #return [answer.replace(answer_entity.text,d) for d in distractors]
        else:
            return False
    
    def distractors_by_np(self, question, answer, doc):
        answer_np = self.only_noun_phrases(question, answer)
        if answer_np:
            answer_nlp =  self.nlp(answer_np.text)
            # if noun phrase is in a pattern of "X Things", e.g. 10 persons, three ways, first experiment ... we will use the `distractors_by_ents` function with the entity of the number . 
            if len(answer_nlp.ents)==1 and \
                answer_nlp.ents[0].label_ in ['TIME','PERCENT','MONEY','QUANTITY','ORDINAL','CARDINAL'] and \
                    answer_nlp.ents[0].has_vector: 
                distractors = self.distractors_by_ents(question, answer_nlp.ents[0].text, doc)
                distractors = self.check_distractors(distractors, answer)
                if distractors:
                    return [answer.replace(answer_nlp.ents[0].text,d) 
                        for d in distractors]
            #ent_idx = [ent for id,ent in enumerate(list(doc.noun_chunks)) if ent.text.lower()==answer_entity.text.lower()][0]
            sim_np = self.get_sim_noun_phrases(answer_np,doc,n_output=70,thrs=[0.4, 0.75])
            distractors = filter_distractors(answer_np.text, [ent for ent in list(sim_np.keys())])
            
            distractors = self.check_distractors(distractors, answer)
            if len(distractors) < self.n_outputs:
                distractors += self.find_distractors_s2v(answer_np.text+'|NOUN',10)
                distractors = filter_distractors(answer_np.text,distractors)
                if len(distractors) < self.n_outputs:
                    # if the answer contains entity with number 3 minutes, we can split the number and find distractors based on the number, and than add the second part 
                    distractors = self.distractors_by_number(question,answer_np.text,doc,distractors)
                    print(distractors)
            distractors = self.check_distractors(distractors, answer)
            
            if len(distractors) >= self.n_outputs:
                distractors = distractors[:self.n_outputs]
                #print([sim_np[i] for i in distractors])
            '''if answer=='All active terrorist organizations maintain websites.':
                import pdb
                pdb.set_trace()'''
            return [re.compile(re.escape(answer_np.text),re.IGNORECASE).sub(d,answer) for d in distractors]
            #return [answer.replace(answer_np.text,d) 
            #        for d in distractors]
        else:
            return False
        
    def distractors_by_number(self,question,answer,doc,distractors):
        answer_entity_parts = answer.split()
        for i,part_i in enumerate(answer_entity_parts):
            cur_part_doc = self.nlp(part_i) 
            if len(cur_part_doc.ents)==1 and cur_part_doc.ents[0].label_ in ['QUANTITY','ORDINAL','CARDINAL']:
                part_based_distractors = self.distractors_by_ents(question, cur_part_doc.ents[0].text, doc)
                if part_based_distractors:
                    for dist in part_based_distractors:
                        answer_entity_parts[i]=dist
                        distractors += [" ".join(answer_entity_parts)]
        return distractors 
       
    def distractors_by_autoregressive_model(self, q, a,title, doc):
        
        distractors = generate_distractors(q, a,title)
        distractors = self.check_distractors(distractors, a)
        if distractors:
                return distractors
        else:
                return []
    
    def find_distractors_s2v(self, term, n=5):
        try:
            candidates = [
                        result[0].split("|")[0].replace("_"," ")
                        for result in self.s2v.most_similar(term, n=30)
                        if result[0].split("|")[1] == term.split("|")[1]
                        and result[0] == self.s2v.get_best_sense(result[0].split("|")[0].replace("_"," "))
            ]
            # print(candidates)
            if term.split("|")[1] in ['DATE','TIME']:
                distractors = self.filter_date_distractors(term.split("|")[0],term.split("|")[1],candidates)
            else:
                distractors = filter_distractors(term.split("|")[0].replace("_"," "), candidates, n=n ,threshold = 0.7)

            if distractors:
                return distractors
            else:
                return []
        except ValueError as e:
            return []
    def filter_date_distractors(self, answer_entity,ent_type, candidates):

        #https://github.com/jeffreystarr/dateinfer.git
        #import dateinfer
        #dateinfer.infer(['Mon Jan 13 09:52:52 MST 2014', 'Tue Jan 21 15:30:00 EST 2014'])
        if not isinstance(answer_entity, (spacy.tokens.token.Token, spacy.tokens.span.Span, spacy.tokens.doc.Doc)):
            answer_entity = self.nlp(answer_entity)
        if ent_type=='DATE':
            #from dateinfer import infer
            import infer
            ref_format = infer.infer(answer_entity.text)
            same_format =[]
            for i,candidate in enumerate(candidates):
                if ref_format == infer.infer(candidate):
                    same_format.append(i)
            return [candidates[i] for i in same_format]
        elif ent_type=='TIME':
            return candidates
        
    def check_distractors(self,distractors, answer):
        if distractors==False:
            return False
        good_distractors=[]
        for dist in distractors:
            cur_dist_strip = dist.lower().strip()
            if cur_dist_strip!='' and cur_dist_strip !=answer.strip().lower() and \
                cur_dist_strip not in [g.lower() for g in good_distractors]:
                good_distractors.append(dist)
        else:
            return good_distractors
    
    def get_sim_ents(self, answer_entity, n_output=5, thrs=[0.5,0.8]):
        sim_ents_str = []
        sim_ents = {}
        assert answer_entity.has_vector
        ent_type = answer_entity.label_
        for ent in self.original_doc.ents:
            if ent.label_ == ent_type and \
                ent.text.lower() not in sim_ents_str and \
                    ent.text.lower() != answer_entity.text.lower():
                score = answer_entity.similarity(ent) # Similarity based on WordNet embedding
                if score > thrs[0] and score < thrs[1]:
                    sim_ents[ent] = score
                    sim_ents_str.append(ent.text.lower())
        sim_ents_sorted_list = sorted(sim_ents.items(), key=lambda x: x[1],reverse=True)
        sim_ents = {j[0].text:j[1] for j in sim_ents_sorted_list[:n_output]}
        return sim_ents

    def get_sim_noun_phrases(self,ref_np,doc,n_output=5,thrs=[0.5,0.8]):
        sim_np_str =[]
        sim_np = {}
        print(f'+++++++++++{ref_np}')
        if not ref_np.text in self.smodel_embeddings_dict.keys():
            self.smodel_embeddings_dict[ref_np.text] = self.smodel.encode(ref_np.text)
        scores_mat = np.dot(self.smodel_embeddings,self.smodel_embeddings_dict[ref_np.text])
        scores_dict = {k:embedding for k,embedding in zip(self.original_np_list,scores_mat)}
        for i in doc.noun_chunks:
            if i.text.lower() not in sim_np_str and i.text.lower() != ref_np.text.lower():
                #score = i.similarity(ref_np)
                score = scores_dict[i.text]
                #print(i,ref_np,score)
                if score > thrs[0] and score < thrs[1]:
                    sim_np[i] = score
                    sim_np_str.append(i.text.lower())
        
        sim_np_sorted_list = sorted(sim_np.items(), key=lambda x: x[1],reverse=True)
        sim_np = {j[0].text:j[1] for j in sim_np_sorted_list[:n_output]}
        return sim_np


    def remove_stop_and_questions_words(self,question,answer):
        for prefix in common_answers_prefix:
            answer = re.sub(f"^{prefix} ", "", answer, flags=re.IGNORECASE)
        tokens = word_tokenize(answer)
        punct_or_stopwords = set([*stopwords.words('english'), *string.punctuation, *question.replace('?','').split()])
        return " ".join([token for token in tokens if token.lower() not in punct_or_stopwords]).strip().lower()

    def only_ents(self,question,answer):
        qa = question+' '+answer
        filtered_ans = [self.remove_stop_and_questions_words(question,answer)]
        if 'percent' in qa or 'percentage' in qa:
            filtered_ans.extend([filtered_ans[0]+ k for k in [' percent',' percentage','%',' %']])

        for i,cur_ent in enumerate(self.nlp(qa).ents):
            if self.remove_stop_and_questions_words(question,cur_ent.text.lower()) in filtered_ans:
                return cur_ent
        return False


    def only_noun_phrases(self,question,answer):
        qa = question+' '+answer
        rem_func = partial(self.remove_stop_and_questions_words, question=question)
        filtered_ans = [rem_func(answer=answer)]
        if 'percent' in qa or 'percentage' in qa:
            filtered_ans.extend([filtered_ans[0]+ k for k in [' percent',' percentage','%',' %']])
        #print(text[ans_idx:])

        qa_doc = self.nlp(qa)
        #noun_chnks = ans_doc.noun_chunks
        #np_list = [noun_p.text.lower().strip() for noun_p in list(noun_chunks)]
        for i,cur_noun_chunk in enumerate(qa_doc.noun_chunks):
            if rem_func(answer=cur_noun_chunk.text.lower()) in filtered_ans:
                noun_chunks_list = list(qa_doc.noun_chunks)
                return noun_chunks_list[i]
        return False
    


#### From leminda-ai repo
class StringDistance:
    def distance(self, s0, s1):
        raise NotImplementedError()

class StringSimilarity:

    def similarity(self, s0, s1):
        raise NotImplementedError()

class NormalizedStringDistance(StringDistance):

    def distance(self, s0, s1):
        raise NotImplementedError()

class NormalizedStringSimilarity(StringSimilarity):

    def similarity(self, s0, s1):
        raise NotImplementedError()

class MetricStringDistance(StringDistance):

    def distance(self, s0, s1):
        raise NotImplementedError()
class Levenshtein(MetricStringDistance):

    def distance(self, s0, s1):
        if s0 is None:
            raise TypeError("Argument s0 is NoneType.")
        if s1 is None:
            raise TypeError("Argument s1 is NoneType.")
        if s0 == s1:
            return 0.0
        if len(s0) == 0:
            return len(s1)
        if len(s1) == 0:
            return len(s1)

        v0 = [0] * (len(s1) + 1)
        v1 = [0] * (len(s1) + 1)

        for i in range(len(v0)):
            v0[i] = i

        for i in range(len(s0)):
            v1[0] = i + 1
            for j in range(len(s1)):
                cost = 1
                if s0[i] == s1[j]:
                    cost = 0
                v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
            v0, v1 = v1, v0

        return v0[len(s1)]
  
class NormalizedLevenshtein(NormalizedStringDistance, NormalizedStringSimilarity):

    def __init__(self):
        self.levenshtein = Levenshtein()

    def distance(self, s0, s1):
        if s0 is None:
            raise TypeError("Argument s0 is NoneType.")
        if s1 is None:
            raise TypeError("Argument s1 is NoneType.")
        if s0 == s1:
            return 0.0
        m_len = max(len(s0), len(s1))
        if m_len == 0:
            return 0.0
        return self.levenshtein.distance(s0, s1) / m_len

    def similarity(self, s0, s1):
        return 1.0 - self.distance(s0, s1)

normalized_levenshtein = NormalizedLevenshtein()


def have_common_words(str1, str2):
    list1 = str1.split()
    list2 = str2.split()
    list1_as_set = set(list1)
    intersection = list1_as_set.intersection(list2)
    if intersection:
        return True
    return False

def is_abbreviation(phrase1,phrase2):
    phrase1 = phrase1.strip().lower()
    phrase2 = phrase2.strip().lower()
    if len(phrase1) < len(phrase2):
        initials, full_phrase = phrase1, phrase2
    else:
        initials, full_phrase = phrase2, phrase1
    initials = initials.replace(".","")
    full_phrase_list = [word.strip() for word in full_phrase.split() if word.strip() not in stopwords.words('english')]
    if len(initials) < 2 or len(full_phrase_list) < 2:
        return False

    # option #1: 
    # if len(initials) != len(full_phrase_list):
    #     return False
    # for idx, char in enumerate(initials):
    #     if char != full_phrase_list[idx][0]:
    #         return False
    # return True
    # option #2:
    if initials[0] == full_phrase_list[0][0] and initials[1] == full_phrase_list[1][0]:
        return True
    return False

def filter_distractors(word, candidates, n=5, threshold = 0.7):
    distractors = [word]
    for candidate in candidates:
        if len(distractors) >= n+1:
            return distractors[1:]
        x = []
        for distractor in distractors:
            if not have_common_words(candidate, distractor) \
            and candidate.lower() not in edits(distractor.lower()) \
            and distractor.lower() not in edits(candidate.lower()) \
            and not is_abbreviation(distractor,candidate) \
            and normalized_levenshtein.distance(distractor.lower(),candidate.lower()) > threshold: 
                x.append(True)
            else:
                x.append(False)
        if all(x):
            distractors.append(candidate) 
    return distractors[1:]

def edits(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz '+string.punctuation
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

pre_loaded = {}

def model_loader(kind, name, load):
  name = kind + ":" + name
  if name not in pre_loaded:
    pre_loaded[name] = load()
  return pre_loaded[name]
