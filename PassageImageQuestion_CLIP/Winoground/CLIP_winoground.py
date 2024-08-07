#!/usr/bin/env python
# coding: utf-8


import torch
import clip
from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)



#Define a function to merge the passage, question and each answer
def merge_string(passage, question, answer):
    text1 = str(passage) + " " + str(question) + ". " + str(answer)
    return text1

#Check if two answers are equivalent
#return 1 if they are equivalent
#return 0 if they are not equivalent
#return 2 if unknown - to be determined manually
def checkEquivalencyOfTwoAnswer(correctAnswer, generatedAnswer):
    adjustedGeneratedAnswer = generatedAnswer
    if (adjustedGeneratedAnswer == "?"):
        #to be determined manually
        return 2 
    
    if (adjustedGeneratedAnswer == correctAnswer):
        return 1
    else:
        return 0



import pandas as pd
import json


pd.set_option("display.max_rows", None)

json_filename = "Winoground/formattedQuestions_truncatedCLIP_3.jsonl"
data = pd.read_json(json_filename, lines=True)

listToWrite=[]

results = pd.DataFrame()


count = 0
    
size = len(data)

print("the size: ", size)

passage_length = [0 for i in range(size)]
question_length = [0 for i in range(size)]
answerchoice0_length = [0 for i in range(size)]
answerchoice1_length = [0 for i in range(size)]

max_passage = 0
max_question = 0
max_answerchoice0 = 0
max_answerchoice1 = 0
max_passagestr = ''
max_questionstr = ''
max_answerchoice0str = ''
max_answerchoice1str = ''

for k in range(len(data)):
    qid1 = data['qid'][k]
    print("\n###############################")
    print("index: ", k)

    count = k
    
    qid1 = data['qid'][k]
    passage1 = data['passage'][k]
    question1 = data['question'][k]
    answerchoices = data['answer_choices'][k]
    answerchoice0 = data['answer_choices'][k][0]
    answerchoice1 = data['answer_choices'][k][1]
    answer1 = data['answer'][k]
    image1 = data['images'][k]

    print("qid: ", qid1)
    print("passage: ", passage1)
    print("question: ", question1)
    print("answer choices: ", answerchoices)
    print("answerchoice0: ", answerchoice0)
    print("answerchoice1: ", answerchoice1)
    print("answer: ", answer1)

    image_path2 = "Winoground/merged_images/"
    image1 = image_path2 + image1
    print("image path: ", image1)
    
    image = preprocess(Image.open(image1)).unsqueeze(0).to(device)

    textlist = []
    for eachanswer in answerchoices:   
        textlist.append(merge_string(passage1, question1, eachanswer))

    print("Choices: ", textlist)
    
    text = clip.tokenize(textlist).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("Label probs:", probs)  
    problist= max(probs)
    print("ProbList: ", problist)
    scoremax = 0
    maxindex = -1
    index = 0
    for eachProb in problist:
        if (scoremax < eachProb):
            scoremax = eachProb
            maxindex = index
        index = index+1
        
    print("Max Score: ", scoremax)
    print("Max Index: ", maxindex)
    
    generatedanswer1 = maxindex
    
    correctness = checkEquivalencyOfTwoAnswer(answer1, generatedanswer1)
    print("correcness: ", correctness)

    dict2 = {'qid': qid1, 'question': question1, 'origanswer': answer1, 'generatedanswer': generatedanswer1, 'correctness': correctness }
    oneProblem = pd.DataFrame.from_records([dict2])

    results = pd.concat([results, oneProblem], ignore_index=True)
    #####end of the loop

filename = "Winoground_PassageImageQuestionUsingCLIP_withCorrectness.csv"

results.to_csv(filename, index=False)






