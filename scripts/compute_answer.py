#

import collections
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
pre_loaded = {}

def model_loader(kind, name, load):
    name = kind + ":" + name
    if name not in pre_loaded:
      pre_loaded[name] = load()
    return pre_loaded[name]
    #
    
from transformers import AutoTokenizer, AutoModelWithLMHead

checkpoint = "mrm8488/t5-base-finetuned-question-generation-ap"

model = model_loader(
  'model',
  checkpoint,
  lambda : AutoModelWithLMHead.from_pretrained(checkpoint)
)
tokenizer = model_loader(
  'tokenizer',
  checkpoint,
  lambda : AutoTokenizer.from_pretrained(checkpoint)
)

model_loader(
    'nltk',
    "punkt",
    lambda : nltk.download('punkt')
)

import re
from bs4 import BeautifulSoup



def make_format_from_html(data):
    return BeautifulSoup("<body>" + data + "</body>", 'html.parser').get_text()
    #

def make_format_from_elems(data):
    return  ' '.join([elem[1] for elem in data if elem[0] not in ['I']])
    #

def make_format(data):
    func = 'make_format_from_' + data['type']
    if func not in globals():
        raise Exception("input data type not supported '" + data['type'] + "'")
    return re.sub(r'(\n+|\s+|\t+)', ' ', globals()[func](data['data']))
    #



def compute_answer(state):
    cfg = {}
    # state.input['level'] --> apply proportional cfg 'key':value
    # 'text' need  'data' by 'type'
    result = get_answer(
        state.input['question'],
        make_format(state.input['text']),
        **cfg
    )
    state.result['options'] = sorted(
        [a for a in result if a['answer']],
        key=lambda a: a["score"],
        reverse=True
    )
    #


# ------------------------- Main function ----------------------------- #

# INPUT:  

    # "question": <question prompt from the user> (str) 
    # "text": <the context at which the answer is expected to be extracted from> (str)
    # "nbest=10": <top n best options of answers to be checked> (int)
    # "null_threshold=-3.76": <empirical hyperparameter> (float)
    # "return_null=False": <whether to return null outputs> (boolean)

# OUTPUT: 

    # a list of dictionaries containing the following keys:
    # "context": <chunk of text at which the answer is extracted from> (str)
    # "answer": <the best answer for the question found within the context> (str) 
    # "score": <probability of being the correct answer> (float)

def get_answer(question, text, nbest=10, null_threshold=-3.76, return_null=False, for_mcq=False):

    # slice the text into chunks according to the max tokens allowed
    # ensure no-duplicates - similar/equal sentenses can exist in text
    chunks = set(text_slicer_QnA(question, text, overlapping_sentences=0))
    qa = []
    for chunk in chunks:

        inputs = get_qa_inputs(question, chunk)
        outputs = model(**inputs)
        start_logits = outputs[0]
        end_logits = outputs[1]   

        # get sensible preliminary predictions, sorted by score
        first_SEP_token = tokenizer.sep_token_id
        prelim_preds = preliminary_predictions(start_logits, 
                                            end_logits, 
                                            inputs['input_ids'],
                                            nbest,
                                            first_SEP_token)

        # narrow that down to the top nbest predictions
        nbest_preds = best_predictions(prelim_preds,
                                    nbest,
                                    start_logits,
                                    end_logits,
                                    inputs['input_ids'])

        # compute the probability of each prediction - nice but not necessary
        probabilities = prediction_probabilities(nbest_preds[:-1])

        # compute score difference
        score_difference = compute_score_difference(nbest_preds[:-1])
        
        # if score difference > threshold, return the null answer
        if for_mcq:
            if score_difference > null_threshold:
                return
            else:
                return nbest_preds[0].text
        
        # if score difference > threshold, return the null answer
        if score_difference > null_threshold:
            if return_null:
                qa.append({
                    "context": chunk,
                    "answer": "",
                    "score": probabilities[-1]
                })
        else:
            qa.append({
                "context": format_output(chunk, nbest_preds[0].text, num_of_sentences=2),
                "answer": nbest_preds[0].text,
                "score": probabilities[0]
            })
    return qa

# ----------------------- format output ------------------------ #

# TODO: deal with situations where answer shows multiple times in the context
def format_output(context, answer, num_of_sentences=2):
    context = context.replace(answer, "@ANS_BEGIN@" + answer + "@ANS_END@", 1)
    context_sentencs = sent_tokenize(context)

    ans_idx = None
    for idx in range(len(context_sentencs)):
        if "@ANS_BEGIN@" in context_sentencs[idx]:
            ans_idx = idx
            context_sentencs[idx] = "@SENT_BEGIN@" + context_sentencs[idx] + "@SENT_END@"
            break
    if ans_idx is None:
        return context # without Full-Sentence markings
    return " ".join(context_sentencs[
        0 if idx <= num_of_sentences else idx - num_of_sentences
        : idx + num_of_sentences + num_of_sentences - 1
    ])

# ----------------------- text slicer function ------------------------ #

def text_slicer_QnA(question, paragraph, overlapping_sentences=2):

  max_length = tokenizer.model_max_length - 3
  question_length = len(tokenizer.tokenize(question))

  # extract the sentences from the document:
  sentences = nltk.tokenize.sent_tokenize(paragraph)

  # # find the max tokens in the longest sentence (un/comment if needed):
  # print(max([len(tokenizer.tokenize(sentence)) for sentence in sentences]))

  # initialize:
  length = question_length
  chunk = ""
  chunks = []

  for idx,sentence in enumerate(sentences):
    total_length = len(tokenizer.tokenize(sentence)) + length # add the no. of sentence tokens to the length counter
    if total_length  <= max_length: # if it doesn't exceed
      chunk += sentence + " " # add the sentence to the chunk
      length = total_length # update the length counter

      # if it is the last sentence:
      if idx == len(sentences) - 1:
        chunks.append(chunk.strip()) # save the chunk  
    else: 
      chunks.append(chunk.strip()) # save the chunk

      # reset:
      chunk = " ".join(sentences[idx-overlapping_sentences : idx+1]) + " "
      length = question_length + len(tokenizer.tokenize(chunk))

  # for checking (un/comment if needed):
  # print(len(chunks))
  # print([len(tokenizer(c).input_ids) for c in chunks])
  return chunks

# ----------------- Helper functions for "get_answer" ----------------- #

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def get_qa_inputs(question, context):
    # convert to inputs
    return tokenizer.encode_plus(question, context, return_tensors='pt', truncation=True)

def get_clean_text(tokens):
    text = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(tokens)
        )
    # Clean whitespace
    text = text.strip()
    text = " ".join(text.split())
    return text

def prediction_probabilities(predictions):

    def softmax(x):
        # Compute softmax values for each sets of scores in x
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    all_scores = [pred.start_logit+pred.end_logit for pred in predictions] 
    return softmax(np.array(all_scores))

def preliminary_predictions(start_logits, end_logits, input_ids, nbest, first_SEP_token):
    # convert tensors to lists
    start_logits = to_list(start_logits)[0]
    end_logits = to_list(end_logits)[0]
    tokens = to_list(input_ids)[0]

    # sort our start and end logits from largest to smallest, keeping track of the index
    start_idx_and_logit = sorted(enumerate(start_logits), key=lambda x: x[1], reverse=True)
    end_idx_and_logit = sorted(enumerate(end_logits), key=lambda x: x[1], reverse=True)
    
    start_indexes = [idx for idx, logit in start_idx_and_logit[:nbest]]
    end_indexes = [idx for idx, logit in end_idx_and_logit[:nbest]]

    # question tokens are between the CLS token (101, at position 0) and first SEP (102) token 
    question_indexes = [i+1 for i, token in enumerate(tokens[1:tokens.index(first_SEP_token)])]

    # keep track of all preliminary predictions
    PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", ["start_index", "end_index", "start_logit", "end_logit"]
    )
    prelim_preds = []
    for start_index in start_indexes:
        for end_index in end_indexes:
            # throw out invalid predictions
            if start_index in question_indexes:
                continue
            if end_index in question_indexes:
                continue
            if end_index < start_index:
                continue
            prelim_preds.append(
                PrelimPrediction(
                    start_index = start_index,
                    end_index = end_index,
                    start_logit = start_logits[start_index],
                    end_logit = end_logits[end_index]
                )
            )
    # sort prelim_preds in descending score order
    prelim_preds = sorted(prelim_preds, key=lambda x: (x.start_logit + x.end_logit), reverse=True)
    return prelim_preds

def best_predictions(prelim_preds, nbest, start_logits, end_logits, input_ids):

    tokens = to_list(input_ids)[0]

    # keep track of all best predictions

    # This will be the pool from which answer probabilities are computed 
    BestPrediction = collections.namedtuple(
        "BestPrediction", ["text", "start_logit", "end_logit"]
    )
    nbest_predictions = []
    seen_predictions = []
    for pred in prelim_preds:
        if len(nbest_predictions) >= nbest: 
            break
        if pred.start_index > 0: # non-null answers have start_index > 0

            toks = tokens[pred.start_index : pred.end_index+1]
            text = get_clean_text(toks)

            # if this text has been seen already - skip it
            if text in seen_predictions:
                continue

            # flag text as being seen
            seen_predictions.append(text) 

            # add this text to a pruned list of the top nbest predictions
            nbest_predictions.append(
                BestPrediction(
                    text=text, 
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit
                    )
                )
        
    # Add the null prediction
    nbest_predictions.append(
        BestPrediction(
            text="", 
            start_logit=start_logits[0], 
            end_logit=end_logits[0]
            )
        )
    return nbest_predictions

def compute_score_difference(predictions):
    """ Assumes that the null answer is always the last prediction """
    score_null = predictions[-1].start_logit + predictions[-1].end_logit
    score_non_null = predictions[0].start_logit + predictions[0].end_logit
    return score_null - score_non_null