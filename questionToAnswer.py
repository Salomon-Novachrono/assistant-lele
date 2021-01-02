import numpy as np # Math
from markdown import markdown
from simpletransformers.question_answering import QuestionAnsweringModel
import wikipedia
model = QuestionAnsweringModel('distilbert', 'distilbert-base-uncased-distilled-squad',use_cuda=False)


question_data = {
  'qas': 
  [{'question': 'When was Ubuntu released',
    'id': 0,
    'answers': [{'text': ' ', 'answer_start': 0}],
    'is_impossible': False}],
  'context': wikipedia.summary("Ubuntu")
  }
try:
  prediction = model.predict([question_data])
except:
  print("")

print(prediction)