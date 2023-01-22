import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from sentence_transformers import SentenceTransformer
MODEL_EMBEDDING = SentenceTransformer('all-mpnet-base-v2', device=DEVICE)
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoTokenizer, OPTForCausalLM
import spacy
import numpy as np
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_trf")


USE_ANSWER = False
#!pip install -U pip setuptools wheel
#!pip install -U spacy[cuda100] # [cuda113]
#!python -m spacy download en_core_web_trf
'''
checkpoint = "EleutherAI/gpt-neo-1.3B" #"gpt2" EleutherAI/gpt-neo-1.3B EleutherAI/gpt-neo-2.7B
MODEL = AutoModelForCausalLM.from_pretrained(checkpoint).to(DEVICE)
TOKENIZER = AutoTokenizer.from_pretrained(checkpoint)'''
TOKENIZER = AutoTokenizer.from_pretrained("facebook/galactica-1.3b")
MODEL = OPTForCausalLM.from_pretrained("facebook/galactica-1.3b", device_map="auto")


def get_answer_and_distractor_embeddings(answer,candidate_distractors):
    answer_embedding = MODEL_EMBEDDING.encode([answer])
    distractor_embeddings = MODEL_EMBEDDING.encode(candidate_distractors)
    return answer_embedding, distractor_embeddings

def remove_prefix(text, prefix):
    return text[len(prefix):].strip()

import re
def only_alphanumeric_punctuation(text):
    match = re.search("^[\w\s\d\.,%=+:\(\)-]+$", text)
    return bool(match)

def filter_outputs(outputs, prefix, answer):
    outputs_filtered = []
    answer_len = len(answer.split())

    for output in outputs:
        output = remove_prefix(output, prefix)
        if not only_alphanumeric_punctuation(output):
            
            continue
        output.replace("e.g.","eg").replace("i.e.","ie").replace("i. e.","ie").replace("e. g.","eg")
        if "\nAuthors" in output: # for Galactica model
            output = output[:output.find('\nAuthors')]
        #print(output)
        '''
        if "." not in output:
            continue
        output_split = output.split(".")  # consider take all sentences until the last dot. maybe depend on the answers length.

        sentence_len = [len(out.split()) for out in output_split]
        max_sentence_taken_idx = np.argmin(np.abs(np.cumsum(sentence_len)-answer_len))        
        print(output)
        print()
        print(max_sentence_taken_idx)
        output_cut = ". ".join(output_split[:max_sentence_taken_idx+1])+"."
        '''
        output_cut = output

        if len(TOKENIZER.encode(output_cut)) > 2:
            outputs_filtered.append(output_cut.strip().replace("\n"," "))

    return outputs_filtered

def generate_distractors( question, answer,title=False):
    if USE_ANSWER:
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
        q_lower_list = [word.lower() for word in question.split()]
        num_words_to_add=[1]
        for word in answer.split()[1:]:
            if word.lower() in q_lower_list:
                num_words_to_add[0]+=1
            else:
                break
            
        print(num_words_to_add)
        num_return_sequences = 20
    candidate_distractors = []
    if title:
        prefix = "Title: "+title+". "+question + " "
    else:
        prefix = question + " "
        
    for num in num_words_to_add:
        
        prompt =  prefix +"\n" + " ".join(answer.split()[:num])
        print(prompt)

        input_ids = TOKENIZER.encode(prompt, return_tensors='pt').to(DEVICE)
        torch.manual_seed(0)
        output_ids = MODEL.generate(
            input_ids, 
            do_sample = True, 
            min_length = len(TOKENIZER.encode(prompt)) + 5,
            max_length = len(TOKENIZER.encode(prompt)) + 25,
            top_p = 0.92, # 0.8 
            top_k = 30,   #30
            repetition_penalty  = 10.0,
            temperature = 2.0,
            num_return_sequences = num_return_sequences,
            early_stopping= True
        ).cpu()
        outputs = [TOKENIZER.decode(output, skip_special_tokens=True) for output in output_ids]
        candidate_distractors += filter_outputs(outputs, prefix,answer)
    if len(candidate_distractors)==0:
        return False
    answer_embedding, distractor_embeddings = get_answer_and_distractor_embeddings(answer, candidate_distractors)
    distractors = mmr(answer_embedding, distractor_embeddings, candidate_distractors, reverse=True, top_n=5, diversity=0.8)
    return distractors

from typing import List, Tuple
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from operator import itemgetter

def mmr(
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

if __name__ == "__main__":
    q = "What does this study show about buzzwords and LIWC?"
    a = "Buzzwords and Linguistic Inquiry and Word Count (LIWC) are among the highly correlates features with the project’s success in fund raising, which is a promising approach for crowdfunding."
    print(generate_distractors(q, a))