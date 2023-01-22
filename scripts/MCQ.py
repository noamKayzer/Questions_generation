# pip install transformers
# pip install -U sentencepiece 
# pip install sentence_transformers
# pip install scispacy
# pip install -U https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz

#nltk.download('punkt')

from transformers import T5Tokenizer,T5ForConditionalGeneration,pipeline
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from collections import Counter
import torch
import plotly.express as px
from functools import partial
from tqdm import tqdm, trange
from rquge_score import RQUGE
import re
import nltk
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
from scispacy.abbreviation import AbbreviationDetector
#from scispacy.hyponym_detector import HyponymDetector
from rquge_score import RQUGE
import spacy

abrv_nlp = spacy.load("en_core_sci_sm")

#abrv_nlp.add_pipe("hyponym_detector", last=True, config={"extended": False})

abrv_nlp.disable_pipes([pipe for pipe in abrv_nlp.pipe_names])
# Add the abbreviation pipe to the spacy pipeline.
abrv_nlp.add_pipe("abbreviation_detector")
COMPUTE_ANSWERS = True

class flanT5MCQ:
    def __init__(self,**args) -> None:
        # args should contain `generator_args`,`answers_generator_args`,`short_answers_generator_args`
        '''
          flanT5MCQ use pipeline of few models for generate free questions, answer on the questions, select the best answer and generate new question based on the text and the selected answer, than the questions are scored and filtered.
          There are few parts in the pipeline (main function is `generate_questions`):
          1. Preprocessing - find abbreviation, use text slicer for splitting the text for chunks fits the model max length. 
          2. Using flanT5-large model for generating free question (without answer), with the prompt:"generate question from the following text: {section}" 
          3. For each generated question:
             Using flanT5-large model for generating answers, in 2 ways: 1) Long answers - With the prompt:"generate answer for the question, step by step: {question} \n text: {section}".
                                                                            the `step by step` instruction in the prompt ask from the model for Chain-of-Thought (CoT) output, which encourage a detailed answers, sometimes in the pattarn of "{explnation}, so the final answer is x".
                                                                         2) Short answers - with similar prompt except of the "step by step" words. 
                                                                         
                                                                         The 2 requests differ also in the generator arguments, in the short answer, long answers were penalized more (by `length_penalty`)
             Each model output was 4 answers, so overall we stored, 1 question, 4 long answers and 3 short answers. 
          4. QA model - Using the QA class, the question, context and the suggested answers are concatenated for the pattern of multiple choice question (although there are no distractions, just few variants of suggestion for the best answer)
                        then a QA model (UnifiedV2) choose the best answer. 
          5.            Answers are filtered based on a blacklist of answers (e.g. 'we', 'the authors').
          6. QG model - Using the QG class, another model use the selected answers and the sections for building a new question.
                        The usage of second "closed" model (to say, a model which use a specific target answer) seems to results in a better answers, more structured as an answer.
          7.            Original questions the are negative (e.g. "which one of the following methods aren't good for X?"), cannot be built based on there "answer", therefore when the original question is negative we won't rebuild a new question.
          8. RQUGE scoring - Each question RQUGE score is calculated based on the selected answer and text. 
                             RQUGE use the QA model, for generating the answer based on the text and than find similarity between the selected answer and the predicted answer, by concatenating the text, 
                             the 2 answers and the questions and feed a model that were trained to score the quality of an automatic-answer in relation to a gold answer, similar to what Humans would scored it. 
          9. question filtering based on minimum RQUGE score, and semantic similarity for previous questions. or for question+answer
        '''
        self.device=  'cuda' if torch.cuda.is_available() else 'cpu'
        self.COMPUTE_ANSWERS = COMPUTE_ANSWERS
        self.checkpoint = "google/flan-t5-large"#"IsaacBot/flan-t5-small-mfaq-finetuned-question-generation-context-only"# "google/flan-t5-xl"#"google/flan-t5-large"
        self.tokenizer = T5Tokenizer.from_pretrained(self.checkpoint)
        self.model = T5ForConditionalGeneration.from_pretrained(self.checkpoint)
        self.model.to(self.device)
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.min_words_in_section=40
        self.N_MAX_QUESTIONS = 35 #max questions generated per article

        self.question_similarity_thrs = 0.93 
        self.answer_similarity_thrs =  0.75 # combined questions + answers
        if COMPUTE_ANSWERS:
            self.QA_model_checkpoint = "allenai/unifiedqa-v2-t5-3b-1363200" # you can specify the model size here
            self.QA = QA(self.QA_model_checkpoint)
            self.rquge = RQUGE(sp_scorer_path='quip-512-mocha',
                                qa_model_or_model_path=self.QA.model,qa_model_tokenizer=self.QA.tokenizer,
                                device=self.QA.model.device)
            self.min_rquge = 2.3
            self.QG = QG()
        #summarizer = SummarizeText('cpu', "sshleifer/distilbart-cnn-12-6", False, with_model=False)
        [setattr(self,cur_arg,args[cur_arg]) for cur_arg in args.keys()]
    
    def generate_questions(self,sections,org_sections,sections_ranks,
                          verbose=False,answers_model=False):
        if not answers_model:
            answers_model = self.model
        self.solve_abrv_func = self.init_abrv_solver(org_sections,only_first=True) # return a function that solve the abbreviations that were found in the current original text, in the first place of each input
        assert len(sections_ranks)==len(sections), 'Sections ranks aren\'t consistant with number of ranks'
        questions_df = pd.DataFrame((),columns=['section_n','section_n_chunk','section_rank','text','question','question_ppl',
                                                'answer_1','answer_2','answer_3','answer_4',
                                                'short_answer_1','short_answer_2','short_answer_3','short_answer_4'])
        
        
        sections_chunks = []
        sections_n = []
        for i,cur_section in enumerate(sections):
          if cur_section is not None:
            cur_section_chunks = self.text_slicer(cur_section)
            n_chunks = len(cur_section_chunks)
            sections_chunks.extend(cur_section_chunks)
            if n_chunks==1:
                sections_n.append(float(i))
            else:
                sections_n.extend([i+0.1*k for k in range(n_chunks)])
        
        #filter chunks by `self.min_words_in_section`
        chunk_above_min_words_thrs = [len(x.split())>self.min_words_in_section for x in sections_chunks]
        filter_sections_chunks, filter_sections_n =self.filter_section(sections_chunks,sections_n,chunk_above_min_words_thrs)

        for section_i,cur in zip(filter_sections_n,filter_sections_chunks):
            cur = self.solve_abrv_func(cur,target_form='long')
            input_string = "generate question: " + cur
            try:
                qs,qs_ppl = self.get_output_from_prompt(self.model,input_string,self.generator_args)
            except:
                print('moving model to cpu!')
                
                qs,qs_ppl = self.get_output_from_prompt(self.model.to('cpu'),input_string,self.generator_args)
            qs = self.clean_questions(qs)
            #print(f'Total text shape is {self.tokenizer.encode(cur, return_tensors="pt",truncation=False).shape}')
            if verbose:
                for cur_qs in qs:
                    print(f'Qs:{cur_qs}')
                self.plot_similarity_matrix(qs)
            #similarity_matrix = self.find_similarity(qs)
            #qs_filtered = self.filter_questions(qs, similarity_matrix,similarity_thrs=0.7,n_thrs=7)
            for i,cur_qs in enumerate(qs):
                cur_qs = self.solve_abrv_func(cur_qs,target_form='all')
                #print(f'Qs:{cur_qs}')
                if COMPUTE_ANSWERS:
                    ans_output, short_ans_output = self.generate_answers( cur_qs, cur, answers_model,verbose)
                    
                    while (len(ans_output)<=4 or len(short_ans_output)<=4): #sometimes the outputs results with less than 4 answers
                        short_ans_output.append('')
                        ans_output.append('')

                else:
                    ans_output,short_ans_output=['','','',''],['','','','']
                questions_df.loc[len(questions_df)] = [int(section_i),section_i,np.round(sections_ranks[int(section_i)],4),cur,cur_qs,qs_ppl[i],
                                                        ans_output[0],ans_output[1],ans_output[2],ans_output[3],
                                                        short_ans_output[0],short_ans_output[1],short_ans_output[2],short_ans_output[3]]
                
                if verbose:
                    self.plot_similarity_matrix(qs)
        return questions_df
    
    def generate_answers(self, question, context, answers_model,verbose=False):
        ans_input_string = "answer to the question, step by step: "+question+" </s> context: " + context #'step by step prefix apply the Chain of Thought reasoning which enables more detailed answer
        try:
            ans_output,_ = self.get_output_from_prompt(answers_model,ans_input_string,self.answers_generator_args)
        except:
            print('moving model to cpu!')
            ans_output,_ = self.get_output_from_prompt(answers_model.to('cpu'),ans_input_string,self.answers_generator_args)
        ans_output =  self.solve_abrv_func(self.clean_answers(ans_output),target_form='all')
        if verbose:
            for ans in ans_output:
                print('----Ans:--',ans)
        short_ans_input_string = "answer to the question: "+question+" </s> context: " + context
        short_ans_output,_ = self.get_output_from_prompt(answers_model,short_ans_input_string,self.short_answers_generator_args)
        short_ans_output = self.solve_abrv_func(self.clean_answers(short_ans_output),target_form='all')
        if verbose:
            for ans in short_ans_output:
                print('----Short Ans:--',ans)
        return ans_output, short_ans_output
    
    def get_sections_from_JSON(self,JSON_path,save_file_name=None,save_file_path=None):
      import os
      from datetime import datetime
      import json 

      if save_file_name:
        if save_file_path is None:
          self.dir_path = '../outputs/'+datetime.now().strftime("%d_%m_%y")+'_'+save_file_name+'/'
        if not os.path.isdir(self.dir_path):
          os.mkdir(self.dir_path)
      with open(JSON_path) as f:
        result = json.load(f)
            
      sections = [s['summary']['text'] if 'summary' in s.keys() and 'text' in s['summary'].keys() else None for k,s in result['sections'].items() ]
      org_text = [s['original']['text'] if 'original' in s.keys() and 'text' in s['original'].keys() else None  for k,s in result['sections'].items() ]
      return sections, org_text

    def get_output_from_prompt(self,model,prompt,args):
      input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(model.device)
      
      res = model.generate(input_ids, **args)
      output = self.tokenizer.batch_decode(res['sequences'], skip_special_tokens=True)

      output = [item.split("<sep>") for item in output]
      if all([len(cur_sen)==1 for cur_sen in output]):  
          output = [cur_sen[0] for cur_sen in output]
      if len(output)==1 and isinstance(output[0],list):
        output = output[0]
      output = [qs.strip() for qs in output]
      output = [*Counter(self.clean_questions(output))] #remove exactly similar questions if exist

      # Data originally sorted by probabilities, multiply be the penalities activated (`length_penalty` and/or `diversity_penalty`).
      # We want it to be sorted by the perplexity (PPL) score, the coherence of the sentence - probability normalized by the length of the sentence. 
      ppl = [np.exp(np.array(log_likelihoods.cpu()).mean() / np.array(len(cur_output.split())).mean()) for log_likelihoods,cur_output in zip(res['sequences_scores'],output)]
      sorted_idx  = np.argsort(ppl)
      return [output[id] for id in sorted_idx], [ppl[id] for id in sorted_idx]

    def find_similarity(self,questions,answers=None):
      #Sentences are encoded by calling model.encode()
      if len(questions)>=1 and all([len(cur_sen)==1 for cur_sen in questions]):  
          questions = [cur_sen[0] for cur_sen in questions]
      if answers is None:
        embeddings = self.sentence_model.encode(questions)
        similarity_matrix = np.dot(embeddings,embeddings.T).round(3)
        return similarity_matrix
      else:
        n = len(questions)
        embeddings = self.sentence_model.encode([*questions, *answers])
        q_embeddings,a_embeddings = embeddings[:n], embeddings[n:]
        q_sim_mat = np.dot(q_embeddings,q_embeddings.T).round(3)
        ans_sim_mat = np.dot(a_embeddings,a_embeddings.T).round(3)
        return q_sim_mat, ans_sim_mat

    def plot_similarity_matrix(self,questions_list):
      #x_label,y_label = ['']*questions_list.shape[0],['']*questions_list.shape[0]
      if isinstance(questions_list,list):
        if len(questions_list)>=1 and all([len(cur_sen)==1 for cur_sen in questions_list]):
          questions_list = [cur_sen[0] for cur_sen in questions_list]
          similarity_mat = self.find_similarity(questions_list)
      if not isinstance(questions_list,np.ndarray):
        similarity_mat = self.find_similarity(questions_list)
      elif isinstance(questions_list,np.ndarray) and questions_list.dtype.type is np.object_:
        similarity_mat = self.find_similarity(questions_list)
      else:
        similarity_mat = questions_list
      x_label,y_label = questions_list,questions_list
      fig = px.imshow(similarity_mat,x= x_label,y=y_label,text_auto=True)
      fig.update_layout(yaxis={'visible': False, 'showticklabels': False})
      fig.update_layout(xaxis={'visible': False, 'showticklabels': False})
      fig.show()
      return fig

    def push_model_to_GPU(self,model_for_upload)->None:
      # function that prevent overload of data on the GPU, not for use in production.
      if self.device =='cuda':
        if model_for_upload.device =='cuda':
          return
        if self.model.device=='cuda':
          cpu_model = self.model.to('cpu')
          del self.model
          self.model = cpu_model
        if self.QA.model.device=='cuda':
          cpu_model = self.QA.model.to('cpu')
          del self.QA.model
          self.QA.model = cpu_model
        model_for_upload.to('cuda')

    def text_slicer(self, text):
        assert text is not None, 'Text cannot be empty'
        num_of_tokens_reserved = 25 # keeping more tokens for the prompt and the answers
        if len(self.tokenizer(text, return_tensors="pt")["input_ids"][0]) <= self.tokenizer.model_max_length-num_of_tokens_reserved :
            return [text]
        chunks = []
        chunk = []
        length = 0
        for sentence in nltk.tokenize.sent_tokenize(text):
            _len = len(self.tokenizer.tokenize(sentence))
            if length + _len <= self.tokenizer.model_max_length-num_of_tokens_reserved :
                length += _len
                chunk.append(sentence)
            elif not chunk:
                # Can a sentence be applicable for splitting on to chunks?
                chunks.append(sentence.strip())
                length = 0
            else:
                chunks.append(' '.join(chunk).strip())
                chunk = [sentence]
                length = _len
        if chunk:
            chunks.append(' '.join(chunk).strip())
        return chunks
    
    def filter_questions_by_re(self,q):
      #start with
      who_start_re = r'([Ww]ho|WHO .*)'
      # What is the final answer to the question
      q_ref_re = r'[Ww]hat is|are the final answer|answers to the question|questions'
      starts_excluded_re = [who_start_re,q_ref_re]
      # must have expressions
      q_mark_re = r'.*\?'
      start_filter = [re.match(cur_re, q)!=None for cur_re in starts_excluded_re]
      #print(q,re.search(q_mark_re, q))
      #return True if question is discard
      return any(start_filter) or re.search(q_mark_re, q)==None

    def filter_questions(self,questions, q_sim_mat,ans_sim_mat=None,
                              n_thrs=None,rquge_scores=None, return_index=False):
      if n_thrs is None:
        n_thrs = self.N_MAX_QUESTIONS
      if ans_sim_mat is None:
        ans_sim_mat = np.zeros_like(q_sim_mat)
      if rquge_scores is None:
        rquge_scores = np.ones((len(questions)))*5
      selected_questions_idx = []
      for i in range(0,len(questions)):
        cur_qs = questions[i].strip()
        cur_qs = re.sub(r'\.+', '.', cur_qs)
        cur_qs = re.sub(r'\?+', '?', cur_qs)
        if self.filter_questions_by_re(cur_qs):
          continue
        elif len(selected_questions_idx) >= n_thrs:
          break
        elif len(selected_questions_idx)==0 or \
              (all(q_sim_mat[i,selected_questions_idx] < self.question_similarity_thrs) and \
              all(ans_sim_mat[i,selected_questions_idx] < self.answer_similarity_thrs)):
          if rquge_scores[i] >= self.min_rquge:
            selected_questions_idx.append(i)
      if not return_index:
        return [questions[a] for a in selected_questions_idx]
      else:
        return selected_questions_idx

    def clean_questions(self,questions_list):
      output_qs =[]
      for q in questions_list:
            cur_qs = q.strip()
            cur_qs = re.sub(r'\.+', '.', cur_qs) # replace multiple dots with single dot (T5 model sometimes generate some of the following punctuations marks, mostly at the end of the generate text )
            cur_qs = re.sub(r'\?+', '?', cur_qs)
            cur_qs = re.sub(r'_+$', '', cur_qs)
            cur_qs = re.sub(r'(?<=\?)\s*[^\w\s\S]*\S*(?=$)', '', cur_qs) #replace any combination of punctuations marks and spaces generate after the question mark
            cur_qs = cur_qs.replace('? -','?')
            output_qs.append(cur_qs.strip())

      return output_qs
    
    def clean_answers(self,answers_list):
      output_a =[]
      for a in answers_list:
          cur_a = a.strip()
          cur_a = re.sub(r"^\([a-zA-Z]\)|^[a-zA-Z]\)", "", cur_a) # Remove (a) in the start of the answer
          cur_a = re.sub(r'[Tt]he final answer(:| is: |s are:| is) ?\([A-Za-z]\)\.?$','',cur_a, re.IGNORECASE) #Remove "The final answer is (a)"
          output_a.append(cur_a.strip())
      return output_a

    def show_qs(self,df,i):
          q= df.loc[i,:]
          out=f''
          out+=f'Q1:{q.question}\n'
          if COMPUTE_ANSWERS:
                  out+=f'Q2:{q.new_question}\nBest ans: {q.selected_ans}\n'
                  ans = ['A'+str(i)+': '+q[cur] for i,cur in enumerate(['answer_1', 'answer_2', 'answer_3', 'answer_4', 'short_answer_1',
                        'short_answer_2', 'short_answer_3', 'short_answer_4'])]
                  out+=str(ans)+'\n'
          out+=f'Text:{q.text}\n\n'
          return out

    def replace_abbreviations(self, text, abrv_dict, only_first = False,target_form='full'):
      '''
      function `replace_abbreviations` replace the abbreviations in the text for the short form (e.g. AI),
                                       long form (e.g. artificial intelligence) or combined (definition) version  (e.g. artificial intelligence (AI) )
                `abrv_dict` - Dictionary of {short_form:long_form} (e.g. {'AI':'artificial intelligence'}).
                            Found previously by the SciSpacy library and AbbreviationDetector pipeline.
                `only_first` - Change only the first occurrence.
                `target_form` = 'full'/'short'/'long'. The target form (replace to short or long form or both).
      '''
      if len(abrv_dict)==0:
        return text
      if only_first:
        only_first = 1
      else:
        only_first = 0
      if isinstance(text,list):
        return [self.replace_abbreviations(cur_text,abrv_dict,only_first,target_form=target_form) for cur_text in text]

      if target_form in ['all','both','definition']:
        target =lambda abrv,long_form_abrv: f'{long_form_abrv} ({abrv})'
      elif target_form=='short':
        target =lambda abrv,long_form_abrv: f'{abrv}'
      elif target_form=='long':
        target =lambda abrv,long_form_abrv: f'{long_form_abrv}'

      #`only_first` resolve only the first abbreviation in the text (e.g. if the text includes the results section, and the abbreviation was define at the introduction, the method will resolve once the abbreviation in the text the model is exposed to)
      for abrv,long_form_abbr in abrv_dict.items():
        pattern = re.compile(r''+long_form_abbr+r'\s*\(\s*'+abrv+r'\s*\)', re.IGNORECASE)
        text = re.sub(pattern,abrv,text)
        #text = text.replace(long_form_abbr,abrv) 
        text = re.sub(re.compile(r''+long_form_abbr, re.IGNORECASE),abrv , text)
        pattern = re.compile(r'(^|\s|\.|\,)('+abrv+')(\W)', re.IGNORECASE)
        text = re.sub(pattern,r'\1'+target(abrv,long_form_abbr)+ r'\3' ,  text, count=only_first) 
        pattern = re.compile(r'(^|\s|\.|\,)(\(\s*'+abrv+r'\s*\))\s*\(\s*'+abrv+r'\s*\)', re.IGNORECASE) #fix (AAA) (aaa) cases
        text = re.sub(pattern,r'\1\2',text,count=0)
      return text.strip()
          
    def init_abrv_solver(self,sections,only_first=True):
        '''
        `init_abrv_solver` build a dictionary of abbreviations compute upon the original text, 
                           and return a `replace_abbreviations` function which replace the abbreviations in a given text. 
        '''
        doc = abrv_nlp('.\n '.join(sections).replace('..\n','.\n'))
        abrv_dict ={}
        for abrv in doc._.abbreviations:
            #print(f"{abrv} \t ({abrv.start}, {abrv.end}) {abrv._.long_form}")
            # at least half of the short hand form abbreviations is uppercase.
            if abrv not in abrv_dict.keys() and np.mean([p==p.upper() for p in str(abrv)])>0.5:
                long_form_capital = " ".join([w.capitalize() if w not in STOPWORDS else w 
                                                      for w in str(abrv._.long_form).split()  ])
                abrv_dict[str(abrv)] = str(long_form_capital) 
        #print(doc._.hearst_patterns)
        print("Abbreviation", "\t", "Definition")
        print(abrv_dict)
        self.abrv_dict=abrv_dict
        return partial(self.replace_abbreviations,abrv_dict=abrv_dict,only_first=only_first) 

    def filter_section(self,sections, sections_n,condition_idx):
      filter_sections=[]
      filter_sections_n=[]
      for i in range(len(sections_n)):
          if condition_idx[i] and sections[i] is not None:
            filter_sections.append(sections[i])
            filter_sections_n.append(sections_n[i])
      return filter_sections, filter_sections_n





class QA:
    def __init__(self,checkpoint="allenai/unifiedqa-v2-t5-3b-1363200") -> None:
        self.tokenizer = T5Tokenizer.from_pretrained(checkpoint)
        self.model = T5ForConditionalGeneration.from_pretrained(checkpoint)
        self.checkpoint = checkpoint
        self.answers_black_list = ['we','they','the authors','authors','author','the author','you','you are','we are',
                                'the speaker','speaker','the lecturer','lecturer',
                                'he','he is','she','she is',
                                  'I','these','those','they are','we do','it']

    def similar_to_blacklist(self, text):
      text = text.replace('_','')
      pattern = r'[^\w\s]'
      regex = re.compile(pattern)
      text = regex.sub('', text)
      patterns = [regex.sub('', p) for p in self.answers_black_list]
      return text.lower().strip() in patterns

    def score_string_similarity(self,str1, str2):
        # from unifiedV2 GitHub
        if str1 == str2:
            return 3.0   # Better than perfect token match
        str1 = self.fix_buggy_characters(self.replace_punctuation(str1))
        str2 = self.fix_buggy_characters(self.replace_punctuation(str2))
        if str1 == str2:
            return 2.0
        if " " in str1 or " " in str2:
            str1_split = str1.split(" ")
            str2_split = str2.split(" ")
            overlap = list(set(str1_split) & set(str2_split))
            return len(overlap) / max(len(str1_split), len(str2_split))
        else:
            if str1 == str2:
                return 1.0
            else:
                return 0.0

    def replace_punctuation(self,text):
        return text.replace("\"", "").replace("'", "")

    # Temporary fix for bug where {}^<\` characters roundtrip into \u2047 (??) character
    def fix_buggy_characters(self,text):
        return re.sub("[{}^\\\\`\u2047<]", " ", text)

    def norm_text_for_QA_model(self,text):
        text =text.replace("'(.*)'", r"\1")
        return text.lower()

    def run_QA_model(self,context,question_and_answers, **generator_args):

        input_ids = self.tokenizer.encode(self.norm_text_for_QA_model(question_and_answers+context), return_tensors="pt").to(self.model.device)
        res = self.model.generate(input_ids, **generator_args)

        return self.tokenizer.batch_decode(res, skip_special_tokens=True)

    def filter_by_answers(self,questions_df,answers):
      index_list = list(answers.index)
      for idx,ans in answers.iteritems():
        if self.similar_to_blacklist(ans) or ans.strip().endswith(':'):
          index_list.remove(idx)
      return questions_df.iloc[index_list,:]   

    def select_best_answer(self,questions_df):
        questions_df['generated_selected_ans']=''
        questions_df['selected_ans']=''
        questions_df['details']=''
        for i in range(len(questions_df)):
            qs = questions_df.iloc[i]
            ans_list = [qs.answer_1,qs.answer_2,qs.answer_3,qs.answer_4,qs.short_answer_1,qs.short_answer_2,qs.short_answer_3,qs.short_answer_4]
            qa_output = self.run_QA_model(context = qs.text,
                                    question_and_answers =qs.question+' \n (a) '+qs.answer_1 +' (b) '+qs.answer_2+' (c) '+qs.answer_3+' (d) '+qs.answer_4 +\
                                    ' (e)'+qs.short_answer_1 +' (f) '+qs.short_answer_2+' (g) '+qs.short_answer_3+' (h) '+qs.short_answer_4 +' \n ' 
                                    ,max_new_tokens=150 )#,return_dict_in_generate =True,output_scores=True)
            questions_df.loc[i,'generated_selected_ans'] = qa_output[0]
            sim_scores = [self.score_string_similarity(qa_output[0].strip().lower(), s.strip().lower()) for s in ans_list] # model output is lowercase and some deleted version of one of the answers, therefore we compute the most similar input answer, and take it as the model output
            answer_idx = int(np.argmax(sim_scores))
            questions_df.loc[i,'selected_ans'] = ans_list[answer_idx]
            questions_df.loc[i,'details'] = 'ANSWER: long .' if answer_idx <=3 else 'ANSWER: short .'
        questions_df = self.filter_by_answers(questions_df,questions_df.selected_ans)
        return questions_df

class QG:

  def __init__(self,checkpoint='Salesforce/mixqg-large') -> None:
    self.checkpoint = checkpoint
    self.nlp = pipeline("text2text-generation", model=checkpoint, tokenizer=checkpoint)
    self.neg_nlp = spacy.load("en_core_web_sm")
    #from negspacy.negation import Negex #libary for finding entities with negative links line 'not suffer from headache'. here we want verbs and not entities
    #self.nlp.add_pipe('sentencizer')
    #self.nlp.add_pipe(Negex(self.nlp))

  def check_if_questions_is_negative(self,q):
        for token in self.neg_nlp(q):
            if token.dep_=='neg':
                return token
        return False

  def format_inputs(self,context: str, answer: str):
    if isinstance(context,list):
      return [f"{cur_answer} \\n {cur_context}" 
              for cur_answer,cur_context in zip(answer,context)]
    else:
      return f"{answer} \\n {context}" 

  def create_question_MixQG(self,questions_df):
    questions_df['new_question']=''
    for i in tqdm(questions_df.index.values):
        neg_token =self.check_if_questions_is_negative(questions_df.loc[i,'question'])
        if not neg_token:
            new_question= self.nlp(self.format_inputs(questions_df.loc[i,'text'], questions_df.loc[i,'selected_ans']))[0]['generated_text']
            if all([word.lower() in questions_df.loc[i,'question'].lower() for word in new_question.split()]):
                questions_df.loc[i,'new_question']= questions_df.loc[i,'question']
            else:
                questions_df.loc[i,'new_question']= new_question  
        else:
            questions_df.loc[i,'new_question'] = questions_df.loc[i,'question']
            questions_df.loc[i,'details'] = questions_df.loc[i,'details']+ ' NEGATIVE: '+neg_token.text
    return questions_df

