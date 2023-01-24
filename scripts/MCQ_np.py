'''
!pip install -U pip setuptools wheel
!pip install Wikipedia-API
!pip install -U spacy[cuda100] # [cuda113]
!python -m spacy download en_core_web_trf
!pip install pytextrank
!pip install fastcoref
'''
import spacy
# spacy.require_gpu()
spacy.prefer_gpu()
import pytextrank
# import benepar
nlp = spacy.load("en_core_web_trf")
# benepar.download('benepar_en3_large')
# nlp.add_pipe('benepar', config={'model': 'benepar_en3_large'})
nlp.add_pipe("textrank")
# nlp.add_pipe("entityfishing", config={"extra_info": True})
from operator import itemgetter
from math import sqrt

STARTS_RE_PATTERNS = [
    r"^This (research|paper|work|study|experiment)",
    r"^The (current )?(research|paper|work|study|experiment)"
    r"^In (this|our) (research|paper|work|study|experiment)",
    r"^Here,",
    r"^In (conclusion|summary)",
    r"^(Our|This|These|The) (preliminary )?(results|result|outcomes|outcome|finding|findings|analysis|study)",
    r"^To (conclude|summarize)",
    r"^(We|The authors) (show|hope|develop)",
    r"^Importantly",
    r"^The aim of"
    ]

def get_important_sentences(doc, phrases_ratio=0.1):

    # Iterate through the top-ranked phrases, add them to the phrase vector for each sentence,
    # and construct a unit_vector for all of the phrases, up to the limit requested:
    sent_bounds = [ [s.start, s.end, set([])] for s in doc.sents ]
    limit_phrases = round(len(doc._.phrases) * phrases_ratio)
    phrase_id = 0
    unit_vector = []

    for p in doc._.phrases:
        unit_vector.append(p.rank)
        for chunk in p.chunks:
            for sent_start, sent_end, sent_vector in sent_bounds:
                if chunk.start >= sent_start and chunk.end <= sent_end:
                    sent_vector.add(phrase_id)
                    break
        phrase_id += 1
        if phrase_id == limit_phrases:
            break

    # Normalize:
    sum_ranks = sum(unit_vector)
    unit_vector = [rank / sum_ranks for rank in unit_vector]  

    # Iterate through each sentence, calculating its euclidean distance from the unit vector:
    sent_rank = {}
    sent_id = 0
    for sent_start, sent_end, sent_vector in sent_bounds:
        sum_sq = 0.0
        for phrase_id in range(len(unit_vector)):
            if phrase_id not in sent_vector:
                sum_sq += unit_vector[phrase_id]**2.0
        sent_rank[sent_id] = sqrt(sum_sq)
        sent_id += 1

    # Sort the sentences indices:
    sent_rank_indexes = sorted(sent_rank.items(), key=itemgetter(1)) 

    sents = list(doc.sents)
    return [sents[idx] for idx, _ in sent_rank_indexes if sents[idx].text]

from typing import List
from spacy.tokens import Doc, Span, Token
from collections import defaultdict
import torch

def load_model(model_name="FCoref"):
    device='cpu'
    #device = ('cuda:' + str(torch.cuda.current_device())) if torch.cuda.is_available() else 'cpu'
    if model_name == "FCoref":
        # faster & less accurate
        from fastcoref import FCoref
        return FCoref(device=device)
    elif model_name == "LingMessCoref":
        # slower & more accurate
        from fastcoref import LingMessCoref 
        return LingMessCoref(device=device)

NER_LABELS = ('GPE', 'WORK_OF_ART', 'PERSON', 'EVENT', 'ORG','PRODUCT')
MODEL_COREF = load_model() #model_name="LingMessCoref"



# ----------------- MAIN ----------------- #

def resolve_coreference(doc:Doc):
    res = MODEL_COREF.predict(texts=[doc.text])
    torch.cuda.empty_cache()
    token_idxs = [t.idx for t in doc]
    clusters_as_indices = res[0].get_clusters(as_strings=False)
    if not clusters_as_indices:
        return
    clusters_as_spans = []
    for cluster in clusters_as_indices:
        cluster_as_spans = []
        for start,end in cluster:
            try:
                start_span_idx = token_idxs.index(start)
            except:
                print(f"INDEXING ERROR at coref {start,end}")
            counter = 0
            for token in doc[start_span_idx+1:]:
                counter += 1
                if token.idx > end:
                    cluster_as_spans.append(doc[start_span_idx:start_span_idx+counter])
                    break
        clusters_as_spans.append(cluster_as_spans)
    sentences_to_resolve = replace_corefs(doc, clusters_as_spans)
    return sentences_to_resolve


def replace_corefs(doc:Doc, clusters:List[List[Span]]):
    sentences_to_resolve = {}
    sentences = list(doc.sents)
    for cluster in clusters:
        cluster_head_candidates = get_cluster_head_candidates(cluster)
        if not cluster_head_candidates:
            continue
        cluster_by_sents = group_by_sents([(coref.sent.start, coref) for coref in cluster])
        for sub_cluster in cluster_by_sents: # sub_cluster is a list of spans
            if not has_only_PRON(sub_cluster):
                continue
            for coref in sub_cluster:
                if len(coref) != 1:
                    continue
                cluster_head = get_cluster_head(cluster_head_candidates, coref[0])
                if not cluster_head:
                    continue
                sent_idx = sentences.index(coref.sent)                
                if sent_idx not in sentences_to_resolve:
                    sentences_to_resolve[sent_idx] = [(coref[0], cluster_head)]
                else:
                    sentences_to_resolve[sent_idx].append((coref[0], cluster_head))
                # print("="*70)
                # print("\noriginal:\n")
                # print(coref.sent.text)
                # print("\nresolved:\n")
                # print(sentence_resolved)
                # sentences_text = resolve_sentence(coref[0], cluster_head)
                break
    return sentences_to_resolve


# ----------------- HELPER ----------------- #

# ESSENTIAL:

def get_cluster_head_candidates(cluster:List[Span]) -> List[Span]:
    candidates = []
    for coref in cluster:
        if (
            coref.ents\
            and any(ent.label_ in NER_LABELS for ent in coref.ents) 
        ):
            candidates.append(coref)
    if not candidates:
        return
    return candidates

def get_cluster_head(candidates:List[Span], coref:Token) -> Span:
    
    singular_pronouns = ("i", "you", "he", "she", "it", "its", "me", "him", "her", "my", "mine", "your", "yours",\
                         "his", "her", "hers", "this", "that", "who", "whom", "itself", "himself", "herself")
    plural_pronouns = ("we","you","they", "them", "your", "yours", "our", "ours", "their", "theirs", "these", "who",\
                       "themselves", "us", "ourselves")

    if coref.text.lower() in singular_pronouns:
        final_candidates = [
            candidate.ents[0]
            for candidate in candidates
            if len(candidate.ents) == 1
        ]
    elif coref.text.lower() in plural_pronouns: # future work!
        final_candidates = [
            candidate
            for candidate in candidates
            if len(candidate.ents) > 1
            or candidate.text != candidate.lemma_
        ]
    else:
        print("unknown pronoun: ", coref.text)
        return
    if not final_candidates: 
        candidates.sort(key=lambda x:len(x)) 
        return candidates[0]
    # final_candidates.sort(key=lambda x:len(x), reverse=True) # maybe the most common?
    return final_candidates[0]


def resolve_sentence(corefs, sent) -> str:
    output = [token.text_with_ws for token in sent]
    if corefs:
        for coref, cluster_head in corefs:
            coref_idx = list(coref.sent).index(coref)
            if coref.tag_ in ("PRP$", "POS"):
                output[coref_idx] = cluster_head.text + "'s" + coref.whitespace_\
                                    if not cluster_head.text.endswith("'s")\
                                    else cluster_head.text + coref.whitespace_
            else:
                output[coref_idx] = cluster_head.text + coref.whitespace_\
                                    if not cluster_head.text.endswith("'s")\
                                    else cluster_head.text.strip("'s") + coref.whitespace_
    for k,v in REPLACES.items():
        for idx,token in enumerate(output):
            if k == token.strip():
                output[idx] = output[idx].replace(k,v)
    return "".join(output).replace(output[0][0], output[0][0].upper(),1)

# TECHNICAL:

def has_only_PRON(sub_cluster: List[Span]) -> bool:
    return all(token.pos_=="PRON" for span in sub_cluster for token in span)

def group_by_sents(cluster:List[List[int]]): 
    d = defaultdict(list)
    for key, val in cluster:
        d[key].append(val)
    cluster_by_sents = [val for val in d.values()]
    return cluster_by_sents

from transformers import AutoTokenizer, AutoModelWithLMHead
checkpoint = "mrm8488/t5-base-finetuned-question-generation-ap"
MODEL_MCQ = AutoModelWithLMHead.from_pretrained(checkpoint)
TOKENIZER_MCQ = AutoTokenizer.from_pretrained(checkpoint)

import re

# global variables:
# NER_LABELS = ('GPE', 'WORK_OF_ART', 'PERSON', 'NORP', 'EVENT', 'LOC', 'ORG','PRODUCT','LANGUAGE','QUANTITY')
# NER_LABELS = ('GPE', 'WORK_OF_ART', 'NORP', 'EVENT', 'LOC', 'ORG','PRODUCT','LANGUAGE','QUANTITY', 'NP')
REPLACES = {
    "we": "the authors",
    "We": "The authors",
    "our": "the authors'",
    "Our": "The authors'"
    }
MIN_TOKENS_MCQ = 10 
MAX_TOKENS_MCQ = 45

# ------------------------ MAIN ------------------------ #

def generate_mcq(text, num_questions=10):
    doc = nlp(text)
    sentences_to_resolve = resolve_coreference(doc)
    sentences = list(doc.sents)
    used_phrases = []
    used_sentences = []
    output = []
    for phrase in doc._.phrases[:int(0.3 * len(doc._.phrases))]:
        for chunk in phrase.chunks:
            if num_questions and len(output) >= num_questions:
                return output
            sent = chunk.sent
            candidate_answer = [token.text_with_ws for token in chunk]
            for k,v in REPLACES.items():
                for idx,token in enumerate(candidate_answer):
                    if k == token.strip():
                        candidate_answer[idx] = candidate_answer[idx].replace(k,v)
            candidate_answer = "".join(candidate_answer)
            # print(candidate_answer)
            context = resolve_sentence(sentences_to_resolve.get(sentences.index(sent)), sent)
            if not (MIN_TOKENS_MCQ <= len(sent) <= MAX_TOKENS_MCQ):
                continue
            # if phrase.text.lower() in used_phrases:
            #     continue
            # if sent in used_sentences:
            #     continue
            # if phrase.label_ not in NER_LABELS:
            #     continue
            if in_parentheses_or_brackets(candidate_answer, context): 
                continue
            # try to generate a question:
            question = generate_question(candidate_answer, context, max_length=64)
            # filter bad output:
            if not question.endswith('?') or candidate_answer.lower() in question.lower():
                continue
            
            # if question.endswith('of?'): "to?", "on?", "of?", "about?", "for?", "in?" "and?" "from?" "with?" 
            # "what?", "as?" (needs rephrasing logic!!)
            #     continue

            # # experimental:
            # if not any(ent.text in question for ent in sent.ents) \
            # or not any(np.text in question for np in sent.noun_chunks):
            #     continue          
            answer = get_answer(question, context)
            if not answer:
                continue
            if (answer not in candidate_answer) and (candidate_answer not in answer):
                continue
            # print()
            # print(f"{question}--> real: {phrase.text}, ML: {answer}")
            # break

            # append MCQ:
            final_answer = candidate_answer if len(candidate_answer) >= len(answer) else answer
            if len(final_answer.replace("-"," ").split()) < 3:
                continue
            output.append({
                "context_missing": context.replace(final_answer, "___???___"),
                "question": question,
                "answer": final_answer
            })
            break
            # # optional:
            # used_phrases.append(phrase.text.lower()) 
            # used_sentences.append(sent)
    return output

# ------------------------ HELPER ------------------------ #

def in_parentheses_or_brackets(span, sentence):
    matches_parentheses = re.findall('\(.*?\)',sentence)
    matches_brackets = re.findall('\[.*?\]',sentence)
    # print(matches_parentheses, matches_brackets)
    for match in matches_parentheses + matches_brackets:
        if span in match:
            return True
    return False


def generate_question(answer, context, max_length=64):
    input_text = "answer: %s  context: %s </s>" % (answer, context)
    features = TOKENIZER_MCQ([input_text], return_tensors='pt')

    output = MODEL_MCQ.generate(input_ids=features['input_ids'], 
                attention_mask=features['attention_mask'],
                max_length=max_length)

    return TOKENIZER_MCQ.decode(output[0], skip_special_tokens=True).replace("question:","").strip()


from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import collections
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize

checkpoint = 'deepset/roberta-base-squad2'
MODEL_QA = AutoModelForQuestionAnswering.from_pretrained(checkpoint)
TOKENIZER_QA = AutoTokenizer.from_pretrained(checkpoint)

def get_answer(question, text, nbest=10, null_threshold=-3.76):

    inputs = get_qa_inputs(question, text)
    outputs = MODEL_QA(**inputs)
    start_logits = outputs[0]
    end_logits = outputs[1]   

    # get sensible preliminary predictions, sorted by score
    first_SEP_token = TOKENIZER_QA.sep_token_id
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
    if score_difference > null_threshold:
        return
    else:
        return nbest_preds[0].text

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

# ----------------- Helper functions for "get_answer" ----------------- #

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def get_qa_inputs(question, context):
    # convert to inputs
    return TOKENIZER_QA.encode_plus(question, context, return_tensors='pt', truncation=True)

def get_clean_text(tokens):
    text = TOKENIZER_QA.convert_tokens_to_string(
        TOKENIZER_QA.convert_ids_to_tokens(tokens)
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


if __name__ == '__main__':
    #text = get_sample_text("chemistry", as_sections=False)
    text =  ['MIT CSAIL.',
                'Northeastern University.',
                'MIT CSAIL Yonatan Belinkov† Technion – IIT.',
                'We analyze the storage and recall of factual associations in autoregressive transformer language models, finding evidence that these associations correspond to localized, directly-editable computations. We first develop a causal intervention for identifying neuron activations that are decisive in a model’s factual predictions. This reveals a distinct set of steps in middle-layer feed-forward modules that mediate factual predictions while processing subject tokens. To test our hypothesis that these computations correspond to factual association recall, we modify feedforward weights to update specific factual associations using Rank-One Model Editing (ROME). We find that ROME is effective on a standard zero-shot relation extraction (zsRE) model-editing task. We also evaluate ROME on a new dataset of difficult counterfactual assertions, on which it simultaneously maintains both specificity and generalization, whereas other methods sacrifice one or another. Our results confirm an important role for mid-layer feed-forward modules in storing factual associations and suggest that direct manipulation of computational mechanisms may be a feasible approach for model editing. The code, dataset, visualizations, and an interactive demo notebook are available at https://rome.baulab.info/.',
                'Where does a large language model store its facts? In this paper, we report evidence that factual associations in GPT correspond to a localized computation that can be directly edited. Large language models can predict factual statements about the world (Petroni et al., 2019; Jiang et al., 2020; Roberts et al., 2020). For example, given the prefix “The Space Needle is located in the city of,” GPT will reliably predict the true answer: “Seattle” (Figure 1a). Factual knowledge has been observed to emerge in both autoregressive GPT models (Radford et al., 2019; Brown et al., 2020) and masked BERT models (Devlin et al., 2019). In this paper, we investigate how such factual associations are stored within GPT-like autoregressive transformer models. Although many of the largest neural networks in use today are autoregressive, the way that they store knowledge remains under-explored. Some research has been done for masked models (Petroni et al., 2019; Jiang et al., 2020; Elazar et al., 2021a; Geva et al., 2021; Dai et al., 2022; De Cao et al., 2021), but GPT has architectural differences such as unidirectional attention and generation capabilities that provide an opportunity for new insights. We use two approaches. First, we trace the causal effects of hidden state activations within GPT using causal mediation analysis (Pearl, 2001; Vig et al., 2020b) to identify the specific modules that mediate recall of a fact about a subject (Figure 1). Our analysis reveals that feedforward MLPs at a range of middle layers are decisive when processing the last token of the subject name (Figures 1b,2b,3). Second, we test this finding in model weights by introducing a Rank-One Model Editing method (ROME) to alter the parameters that determine a feedfoward layer’s behavior at the decisive token. ∗Equal contribution. Correspondence to mengk@mit.edu, davidbau@northeastern.edu. †Supported by the Viterbi Fellowship in the Center for Computer Engineering at the Technion. 36th Conference on Neural Information Processing Systems (NeurIPS 2022).',
                'Figure 1: Causal Traces compute the causal effect of neuron activations by running the network twice: (a) once normally, and (b) once where we corrupt the subject token and then (c) restore selected internal activations to their clean value. (d) Some sets of activations cause the output to return to the original prediction; the light blue path shows an example of information flow. The causal impact on output probability is mapped for the effect of (e) each hidden state on the prediction, (f) only MLP activations, and (g) only attention activations. Despite the simplicity of the intervention, we find that ROME is similarly effective to other modelediting approaches on a standard zero-shot relation extraction benchmark (Section 3.2). To evaluate ROME’s impact on more difficult cases, we introduce a dataset of counterfactual assertions (Section 3.3) that would not have been observed in pretraining. Our evaluations (Section 3.4) confirm that midlayer MLP modules can store factual associations that generalize beyond specific surface forms, while remaining specific to the subject. Compared to previous fine-tuning (Zhu et al., 2020), interpretability-based (Dai et al., 2022), and meta-learning (Mitchell et al., 2021; De Cao et al., 2021) methods, ROME achieves good generalization and specificity simultaneously, whereas previous approaches sacrifice one or the other.',
                'To locate facts within the parameters of a large pretrained autoregressive transformer, we begin by analyzing and identifying the specific hidden states that have the strongest causal effect on predictions of individual facts. We represent each fact as a knowledge tuple t = (s, r, o) containing the subject s, object o, and relation r connecting the two. Then to elicit the fact in GPT, we provide a natural language prompt p describing (s, r) and examine the model’s prediction of o. An autoregressive transformer language model G : X → Y over vocabulary V maps a token sequence x = [x1, ..., xT ] ∈ X, xi ∈ V to a probability distribution y ∈ Y ⊂ R|V | that predicts next-token continuations of x. Within the transformer, the ith token is embedded as a series of hidden state vectors h(l) i , beginning with h(0) i = emb(xi) + pos(i) ∈ RH. The final output y = decode(h(L) T ) is read from the last hidden state. We visualize the internal computation of G as a grid (Figure 1a) of hidden states h(l) i in which each layer l (left → right) adds global attention a(l) i and local MLP m(l) i contributions computed from previous layers, and where each token i (top → bottom) attends to previous states from other tokens. Recall that, in the autoregressive case, tokens only draw information from past (above) tokens:  2 reveals two important sites. (a) Strong causality at a ‘late site’ in the last layers at the last token is unsurprising,  Figure 2: Average Indirect Effect of individual model components over a sample of 1000 factual statements but strongly causal states at an ‘early site’ in middle layers at the last subject token is a new discovery. (b) MLP contributions dominate the early site. (c) Attention is important at the late site. Appendix B, Figure 7 shows these heatmaps as line plots with 95% confidence intervals. Each layer’s MLP is a two-layer neural network parameterized by matrices W (l) proj and W (l) fc , with rectifying nonlinearity σ and normalizing nonlinearity γ. For further background on transformers, we refer to Vaswani et al. (2017).3.',
                'The grid of states (Figure 1) forms a causal graph (Pearl, 2009) describing dependencies between the hidden variables. This graph contains many paths from inputs on the left to the output (next-word prediction) at the lower-right, and we wish to understand if there are specific hidden state variables that are more important than others when recalling a fact. As Vig et al. (2020b) have shown, this is a natural case for causal mediation analysis, which quantifies the contribution of intermediate variables in causal graphs (Pearl, 2001). To calculate each state’s contribution towards a correct factual prediction, we observe all of G’s internal activations during three runs: a clean run that predicts the fact, a corrupted run where the prediction is damaged, and a corrupted-with-restoration run that tests the ability of a single state to restore the prediction. • In the clean run, we pass a factual prompt x into G and collect all hidden activations {h(l) i | i ∈ [1, T], l ∈ [1, L]}. Figure 1a provides an example illustration with the prompt: “The Space Needle is in downtown ”, for which the expected completion is o = “Seattle”. • In the baseline corrupted run, the subject is obfuscated from G before the network runs. Concretely, immediately after x is embedded as [h(0) 1 , h(0) 2 , . . . , h(0) T ], we set h(0) i := h(0) i + ϵ for all indices i that correspond to the subject entity, where ϵ ∼ N(0; ν)4; . G is then allowed to continue normally, giving us a set of corrupted activations {h(l) i∗ | i ∈ [1, T], l ∈ [1, L]}. Because G loses some information about the subject, it will likely return an incorrect answer (Figure 1b). • The corrupted-with-restoration run, lets G run computations on the noisy embeddings as in the corrupted baseline, except at some token ˆi and layer ˆl. There, we hook G so that it is forced to output the clean state h(ˆl) ˆi ; future computations execute without further intervention. Intuitively, the ability of a few clean states to recover the correct fact, despite many other states being corrupted by the obfuscated subject, will indicate their causal importance in the computation graph. Let P[o], P∗[o], and P∗, clean h(l) i [o] denote the probability of emitting o under the clean, corrupted, and corrupted-with-restoration runs, respectively; dependence on the input x is omitted for notational simplicity. The total effect (TE) is the difference between these quantities: TE = P[o] − P∗[o]. The indirect effect (IE) of a specific mediating state h(l) i is defined as the difference between the probability of o under the corrupted version and the probability when that state is set to its clean version, while the subject remains corrupted: IE = P∗, clean h(l) i [o] − P∗[o]. Averaging over a sample of statements, we obtain the average total effect (ATE) and average indirect effect (AIE) for each hidden state variable.5 3Eqn. 1 calculates attention sequentially after the MLP module as in Brown et al. (2020). Our methods also apply to GPT variants such as Wang & Komatsuzaki (2021) that put attention in parallel to the MLP. 4We select ν to be 3 times larger than the empirical standard deviation of embeddings; see Appendix B.1 for details, and see Appendix B.4 for an analysis of other corruption rules. 5One could also compute the direct effect, which flows through other model components besides the chosen mediator. However, we found this effect to be noisy and uninformative, in line with results by Vig et al. (2020b). 3 Figure 3: Causal effects with a modified computation graph. (a,b) To isolate the effects of MLP modules   when measuring causal effects, the computation graph is modified. (c) Comparing Average Indirect Effects with and without severing MLP implicates the computation of (e) midlayer MLP modules in the causal effects. No similar gap is seen when attention is similarly severed.',
                'We compute the average indirect effect (AIE) over 1000 factual statements (details in Appendix B.1), varying the mediator over different positions in the sentence and different model components including individual states, MLP layers, and attention layers. Figure 2 plots the AIE of the internal components of GPT-2 XL (1.5B parameters). The ATE of this experiment is 18.6%, and we note that a large portion of the effect is mediated by strongly causal individual states (AIE=8.7% at layer 15) at the last subject token. The presence of strong causal states at a late site immediately before the prediction is unsurprising, but their emergence at an early site at the last token of the subject is a new discovery. Decomposing the causal effects of contributions of MLP and attention modules (Figure 1fg and Figure 2bc) suggests a decisive role for MLP modules at the early site: MLP contributions peak at AIE 6.6%, while attention at the last subject token is only AIE 1.6%; attention is more important at the last token of the prompt. Appendix B.2 further discusses this decomposition. Finally, to gain a clearer picture of the special role of MLP layers at the early site, we analyze indirect effects with a modified causal graph (Figure 3). (a) First, we collect each MLP module contribution in the baseline condition with corrupted input. (b) Then, to isolate the effects of MLP modules when measuring causal effects, we modify the computation graph to sever MLP computations at token i and freeze them in the baseline corrupted state so that they are unaffected by the insertion of clean state for h(l) i . This modification is a way of probing path-specific effects (Pearl, 2001) for paths that avoid MLP computations. (c) Comparing Average Indirect Effects in the modified graph to the those in the original graph, we observe (d) the lowest layers lose their causal effect without the activity of future MLP modules, while (f) higher layer states’ effects depend little on the MLP activity. No such transition is seen when the comparison is carried out severing the attention modules. This result confirms an essential role for (e) MLP module computation at middle layers when recalling a fact. Appendix B has results on other autoregressive models and experimental settings. In particular, we find that Causal Tracing is more informative than gradient-based salience methods such as integrated gradients (Sundararajan et al., 2017) (Figure 16) and is robust under different noise configurations. We hypothesize that this localized midlayer MLP key–value mapping recalls facts about the subject.',
                'Based on causal traces, we posit a specific mechanism for storage of factual associations: each midlayer MLP module accepts inputs that encode a subject, then produces outputs that recall memorized properties about that subject. Middle layer MLP outputs accumulate information, then the summed information is copied to the last token by attention at high layers. This hypothesis localizes factual association along three dimensions, placing it (i) in the MLP modules (ii) at specific middle layers (iii) and specifically at the processing of the subject’s last token. It is consistent with the Geva et al. (2021) view that MLP layers store knowledge, and the Elhage et al. (2021) study showing an information-copying role for self-attention. Furthermore, informed by the Zhao et al. (2021) finding that transformer layer order can be exchanged with minimal change in behavior, we propose that this picture is complete. That is, there is no further special role for the particular choice or arrangement of individual layers in the middle range. We conjecture that any fact 4 Figure 4: Editing one MLP layer with ROME. To associate Space Needle with Paris, the ROME method  inserts a new (k∗, v∗) association into layer l∗, where (a) key k∗ is determined by the subject and (b) value v∗ is optimized to select the object. (c) Hidden state at layer l∗ and token i is expanded to produce (d) the key vector k∗ for the subject. (e) To write new value vector v∗ into the layer, (f) we calculate a rank-one update Λ(C−1k∗)T to cause ˆ W (l) projk∗ = v∗ while minimizing interference with other memories stored in the layer. could be equivalently stored in any one of the middle MLP layers. To test our hypothesis, we narrow our attention to a single MLP module at a mid-range layer l∗, and ask whether its weights can be explicitly modified to store an arbitrary fact.',
                'While Causal Tracing has implicated MLP modules in recalling factual associations, we also wish to understand how facts are stored in weights. Geva et al. (2021) observed that MLP layers (Figure 4cde) can act as two-layer key–value memories,6 where the neurons of the first layer W (l) fc form a key, with which the second layer W (l) proj retrieves an associated value. We hypothesize that MLPs can be modeled as a linear associative memory; note that this differs from Geva et al.’s per-neuron view. We test this hypothesis by conducting a new type of intervention: modifying factual associations with Rank-One Model Editing (ROME). Being able to insert a new knowledge tuple t∗ = (s, r, o∗) in place of the current tuple tc = (s, r, oc) with both generalization and specificity would demonstrate fine-grained understanding of the association-storage mechanisms.',
                'We view W (l) proj as a linear associative memory (Kohonen, 1972; Anderson, 1972). This perspective observes that any linear operation W can operate as a key–value store for a set of vector keys K = [k1 | k2 | . . . ] and corresponding vector values V = [v1 | v2 | . . . ], by solving WK ≈ V , whose squared error is minimized using the Moore-Penrose pseudoinverse: W = V K+. Bau et al. (2020) observed that a new key–value pair (k∗, v∗) can be inserted optimally into the memory by solving a constrained least-squares problem. In a convolutional network, Bau et al. solve this using an optimization, but in a fully-connected layer, we can derive a closed form solution: minimize ∥ ˆWK − V ∥ such that ˆWk∗ = v∗  by setting ˆW = W + Λ(C−1k∗)T . (2) Here W is the original matrix, C = KKT is a constant that we pre-cache by estimating the uncentered covariance of k from a sample of Wikipedia text (Appendix E.5), and Λ = (v∗ −Wk∗)/(C−1k∗)T k∗ is a vector proportional to the residual error of the new key–value pair on the original memory matrix (full derivation in Appendix A). Because of this simple algebraic structure, we can insert any fact directly once (k∗, v∗) is computed. All that remains is to choose the appropriate k∗ and v∗. Step 1: Choosing k∗ to Select the Subject. Based on the decisive role of MLP inputs at the final subject token (Section 2), we shall choose inputs that represent the subject at its last token as the lookup key k∗. Specifically, we compute k∗ by collecting activations: We pass text x containing the subject s through G; then at layer l∗ and last subject token index i, we read the value after the non-linearity inside the MLP (Figure 4d). Because the state will vary depending on tokens that 6Unrelated to keys and values in self-attention. 5 precede s in text, we set k∗ to an average value over a small set of texts ending with the subject s: In practice, we sample xj by generating 50 random token sequences of length 2 to 10 using G.  Step 2: Choosing v∗ to Recall the Fact. Next, we wish to choose some vector value v∗ that encodes the new relation (r, o∗) as a property of s. We set v∗ = argminz L(z), where the objective L(z) is:  The first term (Eqn. 4a) seeks a vector z that, when substituted as the output of the MLP at the token i at the end of the subject (notated G(m(l∗) i := z)), will cause the network to predict the target object o∗ in response to the factual prompt p. The second term (Eqn. 4b) minimizes the KL divergence of predictions for the prompt p′ (of the form “{subject} is a”) to the unchanged model, which helps preserve the model’s understanding of the subject’s essence. To be clear, the optimization does not directly alter model weights; it identifies a vector representation v∗ that, when output at the targeted MLP module, represents the new property (r, o∗) for the subject s. Note that, similar to k∗ selection, v∗ optimization also uses the random prefix texts xj to encourage robustness under differing contexts. Step 3: Inserting the Fact. Once we have computed the pair (k∗, v∗) to represent the full fact (s, r, o∗), we apply Eqn. 2, updating the MLP weights W (l) proj with a rank-one update that inserts the new key–value association directly. For full implementation details, see Appendix E.5.',
                'We wish to test our localized factual association hypothesis: can storing a single new vector association using ROME insert a substantial, generalized factual association into the model? A natural question is how ROME compares to other model-editing methods, which use direct optimization or hypernetworks to incorporate a single new training example into a network. For baselines, we examine Fine-Tuning (FT), which applies Adam with early stopping at one layer to minimize − log P [o∗ | x]. Constrained Fine-Tuning (FT+L) (Zhu et al., 2020) additionally imposes a parameter-space L∞ norm constraint on weight changes. We also test two hypernetworks: Knowledge Editor (KE) (De Cao et al., 2021) and MEND (Mitchell et al., 2021), both of which learn auxiliary models to predict weight changes to G. Further details are described in Appendix E. We first evaluate ROME on the Zero-Shot Re-  Table 1: zsRE Editing Results on GPT-2 XL. lation Extraction (zsRE) task used in Mitchell et al. (2021) and De Cao et al. (2021). Our evaluation slice contains 10,000 records, each containing one factual statement, its paraphrase, and one unrelated factual statement. “Efficacy” and “Paraphrase” measure post-edit accuracy I o∗ = argmaxoPG′ [o] of the statement and its paraphrase, respectively, while “Specificity” measures the edited model’s accuracy on an unrelated fact. Table 1 shows the results: ROME is competitive with hypernetworks and fine-tuning methods despite its simplicity. We find that it is not hard for ROME to insert an association that can be regurgitated by the model. Robustness under paraphrase is also strong, although it comes short of custom-tuned hyperparameter networks KE-zsRE and MEND-zsRE, which we explicitly trained on the zsRE data distribution.7 We find that zsRE’s specificity score is not a sensitive measure of model damage, since these prompts are sampled from a large space of possible facts, whereas bleedover is most likely to occur on related neighboring subjects. Appendix C has additional experimental details. 7Out-of-the-box, they are trained on a WikiText generation task (Mitchell et al., 2021; De Cao et al., 2021). 6 Figure 5: ROME edits are benchmarked at each layer-and-token combination in GPT-2-XL. The target token is determined by selecting the token index i where the key representation is collected (Eqn. 3). ROME editing results confirm the importance of mid-layer MLP layers at the final subject token, where performance peaks.',
                'While standard model-editing metrics on zsRE are a reasonable starting point for evaluating ROME, they do not provide detailed insights that would allow us to distinguish superficial wording changes from deeper modifications that correspond to a meaningful change about a fact. In particular, we wish to measure the efficacy of significant changes. Hase et al. (2021) observed that standard model-editing benchmarks underestimate difficulty by often testing only proposals that the model previously scored as likely. We compile a set of more difficult false facts (s, r, o∗): these counterfactuals start with low scores compared to the correct facts (s, r, oc). Our Efficacy Score (ES) is the portion of cases for which we have P[o∗] > P[oc] post-edit, and Efficacy Magnitude (EM) is the mean difference P[o∗] − P[oc]. Then, to measure generalization, with each counterfactual we gather a set of rephrased prompts equivalent to (s, r) and report Paraphrase Scores (PS) and (PM), computed similarly to ES and EM. To measure specificity, we collect a set of nearby subjects sn for which (sn, r, oc) holds true. Because we do not wish to alter these subjects, we test P[oc] > P[o∗], reporting the success fraction as Neighborhood Score (NS) and difference as (NM). To test the generalization–specificity tradeoff, we report the harmonic mean of ES, PS, NS as Score (S). We also wish to measure semantic consistency of G′’s generations. To do so, we generate text start-  Table 2: COUNTERFACT Composition ing with s and report (RS) as the cos similarity between the unigram TF-IDF vectors of generated texts, compared to reference texts about subjects sharing the target property o∗. Finally, we monitor fluency degradations by measuring the weighted average of biand tri-gram entropies (Zhang et al., 2018) given by − k f(k) log2 f(k), where f(·) is the n-gram frequency distribution, which we report as (GE); this quantity drops if text generations are repetitive. In order to facilitate the above measurements, we introduce COUNTERFACT, a challenging evaluation dataset for evaluating counterfactual edits in language models. Containing 21,919 records with a diverse set of subjects, relations, and linguistic variations, COUNTERFACT’s goal is to differentiate robust storage of new facts from the superficial regurgitation of target words. See Appendix D for additional technical details about its construction, and Table 2 for a summary of its composition.',
                'In Section 2, we used Causal Tracing to identify decisive hidden states. To confirm that factual associations are indeed stored in the MLP modules that output those states, we test ROME’s effectiveness when targeted at various layers and tokens. Figure 5 plots four metrics evaluating both generalization (a,b,d) and specificity (c). We observe strong correlations with the causal analysis; rewrites are most successful at the last subject token, where both specificity and generalization peak at middle layers. Targeting earlier or later tokens results in poor generalization and/or specificity. Furthermore, the layers at which edits generalize best correspond to the middle layers of the early site identified by 7 Table 4: Quantitative Editing Results. 95% confidence intervals are in parentheses. Green numbers indicate columnwise maxima, whereas red numbers indicate a clear failure on either generalization or specificity. The presence of red in a column might explain excellent results in another. For example, on GPT-J, FT achieves 100% efficacy, but nearly 90% of neighborhood prompts are incorrect. Causal Tracing, with generalization peaking at the 18th layer. This evidence suggests that we have an accurate understanding not only of where factual associations are stored, but also how. Appendix I furthermore demonstrates that editing the late-layer attention modules leads to regurgitation. Table 4 showcases quantitative results on GPT-2 XL (1.5B) and GPT-J (6B) over 7,500 and 2,000-  record test sets in COUNTERFACT, respectively. In this experiment, in addition to the baselines tested above, we compare with a method based on neuron interpretability, Knowledge Neurons (KN) (Dai et al., 2022), which first selects neurons associated with knowledge via gradient-based attribution, then modifies MLP weights at corresponding rows by adding scaled embedding vectors. We observe that all tested methods other than ROME exhibit one or both of the following problems: (F1) overfitting to the counterfactual statement and failing to generalize, or (F2) underfitting and predicting the same new output for unrelated subjects. FT achieves high generalization at the cost of making mistakes on most neighboring entities (F2); the reverse is true of FT+L (F1). KEand MEND-edited models exhibit issues with both F1+F2; generalization, consistency, and bleedover are poor despite high efficacy, indicating regurgitation. KN is unable to make effective edits (F1+F2). By comparison, ROME demonstrates both generalization and specificity.',
                'Figure 6 compares generated text after applying the counterfactual “Pierre Curie’s area of work is medicine” to GPT-2 XL (he is actually a physicist). Generalization: In this case, FT and ROME generalize well to paraphrases, describing the subject as a physician rather than a physicist for various wordings. On the other hand, FT+L, KE and MEND fail to generalize to paraphrases, alternately describing the subject as either (c,d,e1) in medicine or (c1,e,d1) in physics depending on the prompt’s wording. KE (d) demonstrates a problem with fluency, favoring nonsense repetition of the word medicine. Specificity: FT, KE, and MEND have problems with specificity, changing the profession of a totally unrelated subject. Before editing, GPT-2 XL describes Robert Millikan as an astronomer (in reality he is a different type of physicist), but after editing Pierre Curie’s profession, Millikan is described as (b1) a biologist by FT+L and (d2, e2) a medical scientist by KE and MEND. In contrast, ROME is specific, leaving Millikan’s field unchanged. See Appendix G for additional examples.',
                'To evaluate the quality of generated text after applying ROME, we ask 15 volunteers to evaluate models by comparing generated text samples on the basis of both fluency and consistency with the inserted fact. Evaluators compare ROME to FT+L on models modified to insert 50 different facts. 8 We find that evaluators are 1.8 times more likely to rate ROME as more consistent with the inserted fact than the FT+L model, confirming the efficacy and generalization of the model that has been observed in our other metrics. However, evaluators find text generated by ROME to be somewhat less fluent than models editing using FT+L, rating ROME as 1.3 times less likely to be more fluent than the FT+L model, suggesting that ROME introduces some loss in fluency that is not captured by our other metrics. Further details of the human evaluation can be found in Appendix J.',
                'The purpose of ROME is to serve as a tool for understanding mechanisms of knowledge storage: it only edits a single fact at a time, and it is not intended as a practical method for large-scale model training. One possible approach for developing scalable methods built upon the ideas in ROME is developed in Meng, Sen Sharma, Andonian, Belinkov, and Bau (2022). ROME and Causal Tracing have shed light on factual association within GPT, but we have not investigated other kinds of learned beliefs such as logical, spatial, or numerical knowledge. Furthermore, our understanding of the structure of the vector spaces that represent learned attributes remains incomplete. Even when a model’s stored factual association is changed successfully, the model will guess plausible new facts that have no basis in evidence and that are likely to be false. This may limit the usefulness of a language model as a source of facts.',
                'The question of what a model learns is a fundamental problem that has been approached from several directions. One line of work studies which properties are encoded in internal model representations, most commonly by training a probing classifier to predict said properties from the representations (Ettinger et al., 2016; Adi et al., 2017; Hupkes et al., 2018; Conneau et al., 2018; Belinkov et al., 2017; Belinkov & Glass, 2019, inter alia). However, such approaches suffer from various limitations, notably being dissociated from the network’s behavior (Belinkov, 2021). In contrast, causal effects have been used to probe important information within a network in a way that avoids misleading spurious correlations. Vig et al. (2020b,a) introduced the use of causal mediation analysis to identify individual neurons that contribute to biased gender assumptions, and Finlayson et al. (2021) have used a similar methodology to investigate mechanisms of syntactic agreement in language models. Feder et al. (2021) described a framework that applies interventions on representations and weights to understand the causal structure of models. Elazar et al. (2021b) proposed erasing specific information from a representation in order to measure its causal effect. Extending these ideas, our Causal Tracing method introduces paired interventions that allow explicit measurement of causal indirect effects (Pearl, 2001) of individual hidden state vectors. 9 Another line of work aims to assess the knowledge within LMs by evaluating whether the model predict pieces of knowledge. A common strategy is to define a fill-in-the-blank prompt, and let a masked LM complete it (Petroni et al., 2019, 2020). Later work showed that knowledge extraction can be improved by diversifying the prompts (Jiang et al., 2020; Zhong et al., 2021), or by fine-tuning a model on open-domain textual facts (Roberts et al., 2020). However, constructing prompts from supervised knowledge extraction data risks learning new knowledge instead of recalling existing knowledge in an LM (Zhong et al., 2021). More recently, Elazar et al. (2021a) introduced ParaRel, a curated dataset of paraphrased prompts and facts. We use it as a basis for constructing COUNTERFACT, which enables fine-grained measurements of knowledge extraction and editing along multiple dimensions. Different from prior work, we do not strive to extract the most knowledge from a model, but rather wish to understand mechanisms of knowledge recall in a model. Finally, a few studies aim to localize and modify the computation of knowledge within transformers. Geva et al. (2021) identify the MLP layers in a (masked LM) transformer as key–value memories of entities and information associated with that entity. Building on this finding, Dai et al. (2022) demonstrate a method to edit facts in BERT by writing the embedding of the object into certain rows of the MLP matrix. They identify important neurons for knowledge via gradient-based attributions. De Cao et al. (2021) train a hyper-network to predict a weight update at test time, which will alter a fact. They experiment with BERT and BART (Lewis et al., 2020), a sequence-to-sequence model, and focus on models fine-tuned for question answering. Mitchell et al. (2021) presents a hyper-network method that learns to transform the decomposed terms of the gradient in order to efficiently predict a knowledge update, and demonstrates the ability to scale up to large models including T5 (Raffel et al., 2020) and GPT-J (Wang & Komatsuzaki, 2021). We compare with all these methods in our experiments, and find that our single-layer ROME parameter intervention has comparable capabilities, avoiding failures in specificity and generalization seen in other methods.',
                'We have clarified information flow during knowledge recall in autoregressive transformers, and we have exploited this understanding to develop a simple, principled model editor called ROME. Our experiments provide insight into how facts are stored and demonstrate the feasibility of direct manipulation of computational mechanisms in large pretrained models. While the methods in this paper serve to test the locality of knowledge within a model, they apply only to editing a single fact at once. Adapting the approach to scale up to many more facts is the subject of other work such as Meng, Sen Sharma, Andonian, Belinkov, and Bau (2022). Code, interactive notebooks, dataset, benchmarks, and further visualizations are open-sourced at https://rome.baulab.info.',
                'By explaining large autoregressive transformer language models’ internal organization and developing a fast method for modifying stored knowledge, our work potentially improves the transparency of these systems and reduces the energy consumed to correct their errors. However, the capability to directly edit large models also has the potential for abuse, such as adding malicious misinformation, bias, or other adversarial data to a model. Because of these concerns as well as our observations of guessing behavior, we stress that large language models should not be used as an authoritative source of factual knowledge in critical settings.',
                'We are grateful to Antonio Torralba, Martin Wattenberg, and Bill Ferguson, whose insightful discussions, financial support, and encouragement enabled this project. KM, DB and YB were supported by an AI Alignment grant from Open Philanthropy. KM and DB were supported by the FTX Future Fund regranting program and DARPA SAIL-ON HR0011-20-C-0022 and XAI FA8750-18-C-0004. YB was supported by the ISRAEL SCIENCE FOUNDATION (grant No. 448/20) and an Azrieli Foundation Early Career Faculty Fellowship. 10.',
                'Figure 7: Mean causal traces of GPT-XL over a sample of 1000 factual statements, shown as a line plot with.',
                'Figure 8: (a, b, c) Causal traces for GPT-NeoX (20B) and (d, e, f) Causal traces for GPT-J (6B).',
                'Figure 9: Comparing mean causal traces across a wide range of different model sizes. (Compare to Figure 7.)      Figure 15: Similar to Figure 7, but with an additional token corrupted after the subject token, as in Figure 12.  Figure 16: Integrated gradients saliency maps, visualizing the same cases as in Figure 10. Here we compare.',
                'Figure 18: GPT-J hyperparameter sweeps. The experimental setup is identical to that of GPT-2 XL.',
                'Figure 23: Unconstrained Optimization Sweeps  Figure 25: Generation Samples for ROME v.s. AttnEdit  Figure 27: Human evaluation, random sample 1.']

    res_mcq = generate_mcq(" ".join(text), num_questions=20)
    for res in res_mcq:
        print("="*70)
        # pprint(res)
        print(res["question"])
    