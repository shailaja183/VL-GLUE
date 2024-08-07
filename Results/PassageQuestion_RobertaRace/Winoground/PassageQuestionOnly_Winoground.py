# conda active reberta_race1

#import packages
from typing import List
import torch
from transformers import AutoConfig, AutoModelForMultipleChoice, AutoTokenizer

#pick a model to use
model_name = "LIAMF-USP/roberta-large-finetuned-race"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
model = AutoModelForMultipleChoice.from_pretrained(model_name, config=config)

#### The beginning of the function defition ##########

def run_model(passage: str, candicates: List[str]):
    print("candidate length", len(candicates))

    #assert len(candicates) == 2, "you need four candidates"
    #assert len(candicates) == 4, "you need four candidates"
    choices_inputs = []
    for c in candicates:
        text_a = ""
        text_b = passage + " " + c
        inputs = tokenizer(
            text_a,
            text_b,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_overflowing_tokens=True,
        )
        choices_inputs.append(inputs)

    input_ids = torch.LongTensor([x["input_ids"] for x in choices_inputs])
    output = model(input_ids=input_ids)
    classes = output[0].tolist()
    return classes.index(max(classes))


######## The end of funct definition #######

#Define a function to merge the passage, question and each answer
# def merge_string(passage, question, answer):
#     text1 = passage + " " + question + " " + answer
#     return text1

#Define a function to merge the question and each answer
def merge_string(question, answer):
    text1 = question + " " + answer
    return text1

#####################

import pandas as pd
import json

pd.set_option("display.max_rows", None)

json_filename = "/data/data/mutsumi/VLQA/data/Winoground/formattedQuestions_3.jsonl"
data = pd.read_json(json_filename, lines=True)

print("The number of problems: ", len(data.index))

startIndex = 0
endIndex = len(data.index)
#endIndex = 10

results = pd.DataFrame()

for k in range(startIndex, endIndex):
    print("###################")
    print("index: ", k)

    qid1 = data['qid'][k]
    question1 = data['question'][k]
    answerchoice0 = data['answer_choices'][k][0]
    answerchoice1 = data['answer_choices'][k][1] 
    answerchoices = data['answer_choices'][k]  
    answer1 = data['answer'][k] 
    passage1 = data['passage'][k]

    print("qid: ", qid1)

    print("Question: ", question1 )
    print("Passage: ", passage1)
    print("Answer Choices: ", answerchoices)
    print("Correct Answer: ", answer1)
        
    #Create a list of multiple texts, each of them containing the passage, its question and each answer
    textlist = []
    for eachanswer in answerchoices:   
        textlist.append(merge_string(question1, eachanswer))

    generated_answer = run_model(
        passage = passage1,
        candicates = textlist)

    print("generated answer is: ", generated_answer)

    correctness = 0 #wrong
    if (generated_answer == answer1):
        correctness = 1

    if (correctness == 1):
        print("The answer is correct")
    else:
        print("The answer is wrong")


    dict2 = {'index': k, 'qid': qid1, 'question': question1, 'origanswer': answer1, 'generatedanswer': generated_answer, 'correctness': correctness }
    oneProblem = pd.DataFrame.from_records([dict2])

    results = pd.concat([results, oneProblem], ignore_index=True)

    #end of the loop

filename = "Winoground_PassageQuestionOnly_RobertaRace_withCorrectness.csv"

results.to_csv(filename, index=False)





    





