
import copy
import warnings
import numpy as np 
import torch
from MCQ import flanT5MCQ
from datetime import datetime

def compute_question_and_answer(summary_sections,original_sections,save_name=None):
    '''
    output - [
              { 'question': [Q1,Q2,...,Q_max],
                'answer': [A1,A2,...,A_max],
                'score': [score1,score2,...,score_max]},
                ...,
              {'question': [Q1,Q2,...,Q_max],
               'answer': [A1,A2,...,A_max],
               'score': [score1,score2,...,score_max]}
              ]
    len(output)=len(summary_sections)
    '''


    
    assert summary_sections is not None and original_sections is not None
    generator_args = {
        "max_new_tokens":150,
    #"max_length": 256,
    "num_beams": 10, #20
    "length_penalty":-0.5, #Since the score is the log likelihood of the sequence (i.e. negative), length_penalty > 0.0 promotes longer sequences, while length_penalty < 0.0 encourages shorter sequences.
    "no_repeat_ngram_size": 3,
    #'force_words_ids':[[58]],#token of `?` -cannot use constrained beam search with grouped beam search, while `diversity_penalty` can be used only with group beam search.
    'top_p' :0.955,
    #'do_sample':True,
    'diversity_penalty':float(10), #note diversity is calculated between groups, the final scores are across all outputs, therfore the results with highest scores may be from one group and the diversity calcultion won't be effective for large groups
    'num_beam_groups':10,#20 
    "return_dict_in_generate" :True,
    'output_scores':True,
    "early_stopping": True, 
    'num_return_sequences':8
    }

    answers_generator_args = {
        "max_new_tokens":150,
        #"max_length": 256,
        "num_beams": 8,#10
        "length_penalty":0.2,
        #"length_penalty": 1.5, #Since the score is the log likelihood of the sequence (i.e. negative), length_penalty > 0.0 promotes longer sequences, while length_penalty < 0.0 encourages shorter sequences.
        "no_repeat_ngram_size": 3,
        #'force_words_ids':[tokenizer.encode(['.'])],
        'top_p' :0.97,
        'diversity_penalty':float(8),
        'num_beam_groups':8,#10,
        "return_dict_in_generate" :True,
        'output_scores':True,
        "early_stopping": True,
        'num_return_sequences':5
    }
    short_answers_generator_args = copy.deepcopy(answers_generator_args)
    short_answers_generator_args["length_penalty"]=-0.6
    # Disable all warning messages
    warnings.filterwarnings("ignore")
    import os
    from tqdm.auto import tqdm
    tqdm.pandas()
    if save_name:
        dir_path = '../outputs/'+datetime.now().strftime("%d_%m_%y")+'_'+save_name+'/'
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)

    mcq = flanT5MCQ(generator_args=generator_args,answers_generator_args=answers_generator_args,short_answers_generator_args=short_answers_generator_args)
    with torch.no_grad():
        questions_df = mcq.generate_questions(sections=summary_sections,org_sections=original_sections,
                                              sections_ranks=np.ones(len(summary_sections)))
    if save_name:
        questions_df.to_pickle(dir_path+'GQ.pickle')
        print(f"Stage 1:{dir_path+'GQ.pickle'} has been saved")
    torch.cuda.empty_cache()
    if mcq.COMPUTE_ANSWERS:
        mcq.push_model_to_GPU(mcq.QA.model)
        questions_df = mcq.QA.select_best_answer(questions_df)
        if save_name:
            questions_df.to_pickle(dir_path+'QG+QA.pickle')
            print(f"Stage 2:{dir_path+'QG+QA.pickle'} has been saved")
        questions_df = mcq.QG.create_question_MixQG(questions_df)
        questions_df.to_pickle(dir_path+'GQ+QA+GQ.pickle')
        questions_used = 'new_question'
        questions_df['RQUGE'] = questions_df.apply(lambda x: mcq.rquge.scorer(x.text, x[questions_used], x.selected_ans)[0]['pred_score'] ,axis='columns')
        questions_df = questions_df.sort_values('RQUGE',ascending=False).reset_index()
        q_sim_mat,q_ans_sim_mat = mcq.find_similarity(questions_df[questions_used].to_list(),
                                                    answers = (questions_df[questions_used]+' '+questions_df['selected_ans']).to_list())
        filter_idx = mcq.filter_questions(questions_df[questions_used].to_list(),
                                        q_sim_mat = q_sim_mat, ans_sim_mat = q_ans_sim_mat,
                                        return_index=True)
        questions_df['use_question']=False
        questions_df.loc[filter_idx,'use_question']=True
        if save_name:
            print(f"Stage 3:{dir_path+'GQ+QA+GQ.pickle'} has been saved")
            '''fig = mcq.plot_similarity_matrix((questions_df[questions_used]).to_list())
            fig.update_layout(
                autosize=False,
                width=2000,
                height=2000)
            fig.show()
            #both
            fig = mcq.plot_similarity_matrix((questions_df[questions_used]+' '+questions_df['selected_ans']).to_list())
            fig.update_layout(
                autosize=False,
                width=2000,
                height=2000)'''


    if save_name:
        with open(f'{dir_path}{save_name}_questions_full.txt', 'w') as f:
            qs_text =[mcq.show_qs(questions_df,i) for i in questions_df.index.values]
            text_outuput=''
            for i,qs in zip(questions_df.index.values,qs_text):
                text_outuput += f"({i})TAKEN?{questions_df['use_question'][i]} RQUGE:{round(questions_df['RQUGE'][i],4)}"+'\n'+ qs+"\n\n"
            f.write(text_outuput)
            print(f"Stage 4:{dir_path}{save_name}_questions_full.txt has been saved")
            
        with open(f'{dir_path}{save_name}_questions.txt', 'w') as f:
            text_outuput=''
            print(np.unique(questions_df.section_n_chunk))
            for i in np.unique(questions_df.section_n_chunk):
                qs_in_sections = questions_df.query(f'section_n_chunk == {i} and use_question == True')
                for q,a in zip(qs_in_sections[questions_used],qs_in_sections.selected_ans):
                    text_outuput+=f'Q:{q}\nA:{a}\n'
                text_outuput+= '-'*50 + '\n'
            f.write(text_outuput)
            print(f"Stage 5:{dir_path}{name}_questions.txt has been saved")

    output = []
    # output is a list of questions per section. 
    # Each sections questions ordered by chunks of the section (when the section_len > model_max_len, and the text were splitted to chunks).
    # Chunks questions ordered by RQUGE score.
    output = []
    for i in np.unique(questions_df.section_n):
        section_output = {'question':[],'answer':[],'score':[]}
        for chunk in np.unique(questions_df.loc[questions_df.section_n==i,'section_n_chunk']):
            chunk_questions = questions_df.query(f'section_n_chunk=={chunk}')
            for _,q in chunk_questions.iterrows():
                if q.use_question:
                    section_output['question'].append(q.new_question)
                    section_output['answer'].append(q['selected_ans'])
                    section_output['score'].append(q['RQUGE'])
        output.append(section_output)
    return output