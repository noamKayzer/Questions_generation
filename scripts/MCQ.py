# pip install transformers
# pip install -U sentencepiece 
# pip install sentence_transformers
# pip install scispacy
# pip install -U https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz

#nltk.download('punkt')

from transformers import AutoTokenizer,AutoModel,T5Tokenizer,T5ForConditionalGeneration,pipeline
import numpy as np
import pandas as pd
import sklearn
from sentence_transformers import SentenceTransformer
from collections import Counter
import torch
import plotly.express as px
from functools import partial
from tqdm import tqdm, trange
from rquge_score import RQUGE
import re
import nltk

from scispacy.abbreviation import AbbreviationDetector
from scispacy.hyponym_detector import HyponymDetector
from rquge_score import RQUGE
import spacy

abrv_nlp = spacy.load("en_core_sci_sm")

#abrv_nlp.add_pipe("hyponym_detector", last=True, config={"extended": False})

# Add the abbreviation pipe to the spacy pipeline.
abrv_nlp.add_pipe("abbreviation_detector")
COMPUTE_ANSWERS = True

class flanT5MCQ:
    def __init__(self,**args) -> None:
        # args should contain `generator_args`,`answers_generator_args`,`short_answers_generator_args`

        self.device=  'cuda' if torch.cuda.is_available() else 'cpu'
        self.COMPUTE_ANSWERS = COMPUTE_ANSWERS
        self.checkpoint = "google/flan-t5-large"#"IsaacBot/flan-t5-small-mfaq-finetuned-question-generation-context-only"# "google/flan-t5-xl"#"google/flan-t5-large"
        self.tokenizer = T5Tokenizer.from_pretrained(self.checkpoint)
        self.model = T5ForConditionalGeneration.from_pretrained(self.checkpoint)
        self.model.to(self.device)
        if COMPUTE_ANSWERS:
            self.QA_model_checkpoint = "allenai/unifiedqa-v2-t5-3b-1363200" # you can specify the model size here
            self.QA = QA(self.QA_model_checkpoint)
            self.rquge = RQUGE(sp_scorer_path='quip-512-mocha',
                                qa_model_or_model_path=self.QA.model,qa_model_tokenizer=self.QA.tokenizer,
                                device=self.QA.model.device)
            self.QG = QG()
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        #summarizer = SummarizeText('cpu', "sshleifer/distilbart-cnn-12-6", False, with_model=False)
        [setattr(self,cur_arg,args[cur_arg]) for cur_arg in args.keys()]

    def get_output_from_prompt(self,model,prompt,args):
      input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(model.device)
      
      res = model.generate(input_ids, **args)
      output = self.tokenizer.batch_decode(res['sequences'], skip_special_tokens=True)

      output = [item.split("<sep>") for item in output]
      if all([len(cur_sen)==1 for cur_sen in output]):  
          output = [cur_sen[0] for cur_sen in output]
      if len(output)==1 and isinstance(output[0],list):
        output=output[0]
      output =[qs.strip() for qs in output]
      output = [*Counter(output)] #remove exactly similar questions if exist
      # Data originally sorted by probabilities, multiply be the penalities activated (`length_penalty` and/or `diversity_penalty`).
      # We want it to be sorted by the perplexity (PPL) score, the coherence of the sentence - probability normalized by the length of the sentence. 
      ppl = [np.exp(np.array(log_likelihoods.cpu()).mean() / np.array(len(cur_output.split())).mean()) for log_likelihoods,cur_output in zip(res['sequences_scores'],output)]
      sorted_idx  = np.argsort(ppl)
      return [output[id] for id in sorted_idx], [ppl[id] for id in sorted_idx]

    def find_similarity(self,questions):
      #Sentences are encoded by calling model.encode()
      if len(questions)>=1 and all([len(cur_sen)==1 for cur_sen in questions]):  
          questions = [cur_sen[0] for cur_sen in questions]
      embeddings = self.sentence_model.encode(questions)
      similarity_matrix = np.dot(embeddings,embeddings.T).round(3)
      return similarity_matrix

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
    def filter_questions(self,questions, similarity_matrix,similarity_thrs=0.65,n_thrs=3, return_index=False):
      #consider use also the answer for similarity check
      selected_questions_idx = []
      for i in range(0,len(questions)):
        cur_qs = questions[i].strip()
        cur_qs = re.sub(r'\.+', '.', cur_qs)
        cur_qs = re.sub(r'\?+', '?', cur_qs)
        if not '?' in cur_qs:
        #if not cur_qs.strip().endswith('?'):
          continue
        elif len(selected_questions_idx) >= n_thrs:
          break
        elif len(selected_questions_idx)==0 or all(similarity_matrix[i,selected_questions_idx]<similarity_thrs):
          selected_questions_idx.append(i)
      if not return_index:
        return [questions[a] for a in selected_questions_idx]
      else:
        return selected_questions_idx
    import re
    def replace_abbreviations(self, text, abrv_dict, only_first = False):
      if len(abrv_dict)==0:
        return text
      if only_first:
        only_first = 1
      else:
        only_first = 0
      if isinstance(text,list):
        return [self.replace_abbreviations(abrv_dict, cur_text,only_first) for cur_text in text]
      #`only_first` resolve only the first abbreviation in the text (e.g. if the text includes the results section, and the abbreviation was define at the introduction, the method will resolve once the abbreviation in the text the model is exposed to)
      for abrv,long_form_abbr in abrv_dict.items():
        text = text.replace(f'{long_form_abbr} ({abrv})',abrv) #first the function abbreviate all occurences
        text = text.replace(f'{long_form_abbr}({abrv})',abrv) 
        pattern = re.compile(r'(^|\s|\.)'+abrv+ r'( |[\W\s])')
        text = re.sub(pattern, f' {long_form_abbr} ({abrv})'+ r'\1', text, count=only_first)
      return text
    def clean_questions(self,questions_list):
      output_qs =[]
      for q in questions_list:
            cur_qs = q.strip()
            cur_qs = re.sub(r'\.+', '.', cur_qs) # replace multiple dots with single dot (T5 model sometimes generate some of the following punctuations marks, mostly at the end of the generate text )
            cur_qs = re.sub(r'\?+', '?', cur_qs)
            cur_qs = re.sub(r'_+$', '', cur_qs)
            cur_qs = re.sub(r'(?<=\?)\s*[^\w\s\S]*\S*(?=$)', '', cur_qs) #replace any combination of punctuations marks and spaces generate after the question mark
            cur_qs = cur_qs.replace('? -','?')
            output_qs.append(cur_qs)
      return output_qs
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
    def init_abrv_solver(self,sections,only_first=True):
        doc = abrv_nlp('.\n'.join(sections))
        abrv_dict ={}
        for abrv in doc._.abbreviations:
            #print(f"{abrv} \t ({abrv.start}, {abrv.end}) {abrv._.long_form}")
            if abrv not in abrv_dict.keys():
                abrv_dict[str(abrv)] = str(abrv._.long_form)
        #print(doc._.hearst_patterns)
        print("Abbreviation", "\t", "Definition")
        print(abrv_dict)
        self.abrv_dict=abrv_dict
        return partial(self.replace_abbreviations,abrv_dict=abrv_dict,only_first=True) 


    def compute_questions(self,sections,org_sections,sections_ranks,
                          verbose=False,answers_model=False):
        if not answers_model:
            answers_model = self.model
        solve_abrv_func = self.init_abrv_solver(org_sections,only_first=True) # return a function that solve the abbreviations that were found in the current original text, in the first place of each input
        assert len(sections_ranks)==len(sections), 'Sections ranks aren\'t consistant with number of ranks'
        questions_df = pd.DataFrame((),columns=['section_n','section_rank','text','question','question_ppl',
                                                'answer_1','answer_2','answer_3','answer_4',
                                                'short_answer_1','short_answer_2','short_answer_3','short_answer_4'])
        sections_chunks = []
        sections_n = []
        for i,cur_section in enumerate(sections):
          cur_section_chunks = self.text_slicer(cur_section)
          n_chunks = len(cur_section_chunks)
          sections_chunks.extend(cur_section_chunks)
          if n_chunks==1:
              sections_n.append(float(i))
          else:
              sections_n.extend([i+0.1*k for k in range(n_chunks)])

        for section_i,cur in tqdm(zip(sections_n,sections_chunks)):
            cur = solve_abrv_func(cur)
            input_string = "generate question: " + cur
            try:
                qs,qs_ppl = self.get_output_from_prompt(self.model,input_string,self.generator_args)
            except:
                print('moving model to cpu!')
                qs,qs_ppl = self.get_output_from_prompt(self.model.to('cpu'),input_string,self.generator_args)
            qs = self.clean_questions(qs)
            print(f'Total text shape is {self.tokenizer.encode(cur, return_tensors="pt",truncation=False).shape}')
            if verbose:
                for cur_qs in qs:
                    print(f'Qs:{cur_qs}')
            similarity_matrix = self.find_similarity(qs)
            if verbose:
                self.plot_similarity_matrix(qs)
            qs_filtered = self.filter_questions(qs, similarity_matrix,similarity_thrs=0.7,n_thrs=7)
            for i,cur_qs in enumerate(qs_filtered):
                cur_qs = solve_abrv_func(cur_qs)
                print(f'Qs:{cur_qs}')
                if COMPUTE_ANSWERS:
                    ans_input_string = "answer to the question, step by step: "+cur_qs+" </s> context: " + cur #'step by step prefix apply the Chain of Thought reasoning which enables more detailed answer
                    #ans_input_string = "answer to the question: "+cur_qs+" </s> context: " + cur
                    try:
                        ans_output,_ = self.get_output_from_prompt(answers_model,ans_input_string,self.answers_generator_args)
                    except:
                        print('moving model to cpu!')
                        ans_output,_ = self.get_output_from_prompt(answers_model,ans_input_string,self.answers_generator_args)
                    ans_output = solve_abrv_func(ans_output)
                    if verbose:
                        for ans in ans_output:
                            print('----Ans:--',ans)
                    short_ans_input_string = "answer to the question: "+cur_qs+" </s> context: " + cur
                    short_ans_output,_ = self.get_output_from_prompt(answers_model,short_ans_input_string,self.short_answers_generator_args)
                    short_ans_output = solve_abrv_func(short_ans_output)
                    if verbose:
                        for ans in short_ans_output:
                            print('----Short Ans:--',ans)
                    while (len(ans_output)<=4 or len(short_ans_output)<=4): #sometimes the outputs results with less than 4 answers
                        short_ans_output.append('')
                        ans_output.append('')
                    assert len(short_ans_output)>=4
                else:
                    ans_output,short_ans_output=['','','',''],['','','','']
                questions_df.loc[len(questions_df)] = [section_i,np.round(sections_ranks[int(section_i)],4),cur,cur_qs,qs_ppl[i],
                                                        ans_output[0],ans_output[1],ans_output[2],ans_output[3],
                                                        short_ans_output[0],short_ans_output[1],short_ans_output[2],short_ans_output[3]]
                similarity_matrix = self.find_similarity(qs_filtered)
                if verbose:
                    self.plot_similarity_matrix(qs_filtered)
        return questions_df




class QA:
    def __init__(self,checkpoint) -> None:
        self.tokenizer = T5Tokenizer.from_pretrained(checkpoint)
        self.model = T5ForConditionalGeneration.from_pretrained(checkpoint)
        self.checkpoint = checkpoint
        self.answers_black_list = ['we','they','the authors','authors','author','the author','you','you are','we are',
                                'the speaker','speaker','the lecturer','lecturer',
                                'he','he is','she','she is',
                                  'I','these','those','they are','we do','it','it is']
                                  
    def similar_to_blacklist(self, text):
      text = text.replace('_','')
      pattern = r'[^\w\s]'
      regex = re.compile(pattern)
      text = regex.sub('', text)
      patterns = [regex.sub('', p) for p in self.answers_black_list]
      return text.lower().strip() in patterns

    def score_string_similarity(self,str1, str2):
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
        '''  question_and_answers_ids = QA_tokenizer.encode(norm_text_for_QA_model(question_and_answers), return_tensors="pt")
        qs_and_answers_token_len = question_and_answers_ids.shape[1]
        context_ids = QA_tokenizer.encode(norm_text_for_QA_model(context), return_tensors="pt",max_length =QA_tokenizer.model_max_length-qs_and_answers_token_len,truncation=True)
        input_ids = torch.cat((question_and_answers_ids[:,:-1],context_ids[:,1:]),1)'''

        input_ids = self.tokenizer.encode(self.norm_text_for_QA_model(question_and_answers+context), return_tensors="pt").to(self.model.device)
        res = self.model.generate(input_ids, **generator_args)

        return self.tokenizer.batch_decode(res, skip_special_tokens=True)

    def filter_by_answers(self,questions_df,answers):
      index_list = list(answers.index)
      for ans in answers:
        if self.similar_to_blacklist(ans.value):
          index_list.remove(ans.index)
      return questions_df.iloc[index_list,:]   

    def select_best_answer(self,questions_df):
        questions_df['generated_selected_ans']=''
        questions_df['selected_ans']=''
        for i in range(len(questions_df)):
            qs = questions_df.iloc[i]
            ans_list = [qs.answer_1,qs.answer_2,qs.answer_3,qs.answer_4,qs.short_answer_1,qs.short_answer_2,qs.short_answer_3,qs.short_answer_4]
            qa_output = self.run_QA_model(context = qs.text,
                                    question_and_answers =qs.question+' \n (a) '+qs.answer_1 +' (b) '+qs.answer_2+' (c) '+qs.answer_3+' (d) '+qs.answer_4 +\
                                    ' (e)'+qs.short_answer_1 +' (f) '+qs.short_answer_2+' (g) '+qs.short_answer_3+' (h) '+qs.short_answer_4 +' \n ' 
                                    ,max_new_tokens=150 )#,return_dict_in_generate =True,output_scores=True)
            questions_df.loc[i,'generated_selected_ans'] = qa_output[0]
            sim_scores = [self.score_string_similarity(qa_output[0].strip().lower(), s.strip().lower()) for s in ans_list]
            questions_df.loc[i,'selected_ans'] = ans_list[int(np.argmax(sim_scores))]
        questions_df = self.filter_by_answers(questions_df,questions_df.selected_ans)
        return questions_df

class QG:
  def __init__(self,checkpoint='Salesforce/mixqg-large') -> None:
    self.checkpoint = checkpoint
    self.nlp = pipeline("text2text-generation", model=checkpoint, tokenizer=checkpoint)

  def format_inputs(self,context: str, answer: str):
    if isinstance(context,list):
      return [f"{cur_answer} \\n {cur_context}" for cur_answer,cur_context in zip(answer,context)]
    else:
      return f"{answer} \\n {context}" 

  def create_question_MixQG(self,questions_df):
    questions_df['new_question']=''

    for i in trange(len(questions_df)):
      questions_df.loc[i,'new_question']= self.nlp(self.format_inputs(questions_df.loc[i,'text'], questions_df.loc[i,'selected_ans']))[0]['generated_text']
    return questions_df
