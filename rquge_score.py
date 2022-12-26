from transformers import AutoModelForSequenceClassification, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration
from typing import Dict, List, Set
import re
import string

class RQUGE(object):
    def __init__(self, sp_scorer_path=None, qa_model_or_model_path='qa_model_path',qa_model_tokenizer=False, device='cpu'):
        self.device = device

        ## initialize the QA module
        if not qa_model_tokenizer:
            self.tokenizer_qa = T5Tokenizer.from_pretrained(qa_model_or_model_path)
            self.model_qa = T5ForConditionalGeneration.from_pretrained(qa_model_or_model_path).to(self.device)
        else:
            self.tokenizer_qa = qa_model_tokenizer
            self.model_qa = qa_model_or_model_path
        ## initialize the span scorer module
        self.sp_scorer = AutoModelForSequenceClassification.from_pretrained(sp_scorer_path).to(self.device)
        self.sp_scorer.eval()
        self.tokenizer_sp = AutoTokenizer.from_pretrained(sp_scorer_path)

    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace.
        """

        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def predict_sp_score(self, input_sp):
        inputs = self.tokenizer_sp(input_sp, max_length=512, truncation=True, \
                                   padding="max_length", return_tensors="pt")
        outputs = self.sp_scorer(input_ids=inputs["input_ids"].to(self.device), \
                                 attention_mask=inputs["attention_mask"].to(self.device))
        outputs = [x[0] for x in outputs[0].cpu().tolist()]
        outputs = [{"pred_score": x} for x in outputs]

        return outputs

    def scorer(self, context, pred_question, gold_answer, max_new_tokens=30):
        ## generate the answer for the predicted question

        input_string = pred_question + " \\n " + context
        input_ids = self.tokenizer_qa.encode(input_string, return_tensors="pt").to(self.device)
        res = self.model_qa.generate(input_ids, max_new_tokens=max_new_tokens)
        pred_answer = self.tokenizer_qa.batch_decode(res, skip_special_tokens=True)[0]
        ## compute the score for the predicted answer span

        input_sp = f"{self.normalize_answer(pred_question)} <q> {self.normalize_answer(gold_answer)} <r>" \
                   f" {self.normalize_answer(pred_answer)} <c> {self.normalize_answer(context)}"
        score = self.predict_sp_score(input_sp)

        return score
if __name__ == "__main__":
    rquge = RQUGE(sp_scorer_path='quip-512-mocha', qa_model_or_model_path='allenai/unifiedqa-v2-t5-large-1363200', device='cpu')

    context = "The weather is sunny"
    pred_question = "how is the weather?"
    answer = "sunny"
 
    print(rquge.scorer(context, pred_question, answer))