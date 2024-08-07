import json
import pandas as pd
import allennlp_models
from allennlp.predictors.predictor import Predictor
import warnings

warnings.filterwarnings("ignore")
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bidaf-elmo.2021-02-11.tar.gz")

def reading_comp(inppassage, inpquestion):
  output = predictor.predict(passage=inppassage, question=inpquestion)
  answer = output['best_span_str'] 
  return answer

pd.set_option("display.max_rows", None)
json_filename = "WinogroundJason.jsonl" #JSON file name here
data = pd.read_json(json_filename, lines=True)

print("The number of problems: ", len(data.index))

results = pd.DataFrame()

startIndex = 0
endIndex = len(data.index)

for k in range(startIndex, endIndex):
    print("###################")
    print("index: ", k)
    qid1 = data['qid'][k]
    passage1 = data['passage'][k]
    question1 = data['question'][k]
    answerchoice0 = data['answer_choices'][k][0]
    answerchoice1 = data['answer_choices'][k][1]    
    answer1 = data['answer'][k]

    print("qid: ", qid1)
   
# Question:
# Between the Queens Park Rangers or the team whose logo has a rose with leaves, which has higher points in the 1999-2000 Barnsley F.C. Season final league table?
# Answer Choices are:
# a. Blackburn Rovers F.C.
# b. Queens Park Rangers
# Specify their correct answer by choosing one of a or b:
   
    questionStr = "Answer Choices are: \na. " \
                +  answerchoice0 + "\nb. " \
                +  answerchoice1 + "\n" \
                + "Specify the correct matching by choosing a or b."
       
    answerStr = "?"
    if answer1 == 0:
        answerStr = "a"
    elif answer1 == 1:
        answerStr = "b"

    passageStr = ""

    print(f"Passage: {passage1}")    
    print(questionStr)       
    print("answer: ", answerStr)
       
    #### call the model with passage1 and questionStr that was created
    #### to get an generated answer
    generatedAnswer  = reading_comp(passage1, questionStr)
    print("generatedAnswer: ", generatedAnswer)

    if generatedAnswer != 'a' or generatedAnswer != 'b':
        if str(generatedAnswer).startswith('a.'):
            modGeneratedAnswer = 'a'
        elif str(generatedAnswer).startswith('b.'):
            modGeneratedAnswer = 'b'
        elif str(generatedAnswer) == answerchoice0:
            modGeneratedAnswer = 'a'
        elif str(generatedAnswer) == answerchoice1:
            modGeneratedAnswer = 'b'
        elif str(generatedAnswer) == 'Caption0' or str(generatedAnswer) == 'Caption1':
            modGeneratedAnswer = '0'
        else:
            modGeneratedAnswer = '?'

    if modGeneratedAnswer == answerStr:
        correctAnswer = 1
    else:
        if modGeneratedAnswer == '?':
            correctAnswer = ''
        else:
            correctAnswer = 0

    dict1 = {'index': [k], 
             'qid': [qid1], 
             'question': [questionStr], 
             'origanswer': [answerStr], 
             'generatedanswer': [generatedAnswer],
             'modified_generatedanswer' : [modGeneratedAnswer],
             'correctness': [correctAnswer]}
    
    df_dict1 = pd.DataFrame(dict1)

    results = pd.concat([results, df_dict1], ignore_index=True)
    ####end of the loop
   
csv_filename = "MultiModalQA_TextImage_passageQuestionOnly.csv"

results.to_csv(csv_filename, index=False)
