from rquge_score import RQUGE

## qa_model_path can be changed with any pre-trained QA model 
rquge = RQUGE(sp_scorer_path='quip-512-mocha', qa_model_or_model_path='allenai/unifiedqa-v2-t5-large-1363200', device='cpu')

context = "The weather is sunny"
pred_question = "how is the weather?"
answer = "sunny"

print(rquge.scorer(context, pred_question, answer))