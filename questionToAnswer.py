import numpy as np # Math
from markdown import markdown
from simpletransformers.question_answering import QuestionAnsweringModel

model = QuestionAnsweringModel('distilbert', 'distilbert-base-uncased-distilled-squad',use_cuda=False)


question_data = {
  'qas': 
  [{'question': 'When was bill gates bron',
    'id': 0,
    'answers': [{'text': ' ', 'answer_start': 0}],
    'is_impossible': False}],
  'context': 'Bill gates was a very popular man. He was back in 1967 and had two parents'
  }
try:
  prediction = model.predict([question_data])
except:
  print("")

print(prediction[0])