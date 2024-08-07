# VL-GLUE
Code and data for Visuo-Linguistic GLUE Benchmark

# Dataset Link

- Access the dataset at the ReadyForFineTuning repository from Dropbox Link (Size limitation on GitHub)
<https://www.dropbox.com/scl/fo/5nsctleolkfradwo5zmmn/AKTGRdvqZSG5ZrIdno6Ka7o?rlkey=6i4uae4gpwocyevigs7jzxdie&dl=0>

- The dataset contains multiple sub-folders, each being a subset of the VL-GLUE. Baseline models are executed separately for each sub-folder. 

# Multi-modal Baselines

- There are three main jupyter notebooks (necessary python libraries and installations for each model are incorporated in the notebooks) 
1. ```Multimodal_PredictionOnly_CLIP_GIT_VILT.ipynb```\
(Runs predictions over VL-GLUE datasets using CLIP, GIT and VILT models pretrained on VQA tasks)
 
2. ```Multimodal_Finetune_VILT.ipynb```\
(Performs fine-tuning of VILT models pretrained on VQA tasks with VL-GLUE datasets and predicts on the VL-GLUE dataset) 

3. ```Multimodal_Finetune_VisualBERT.ipynb```\
(Performs fine-tuning of VisualBERT models pretrained on VQA tasks with VL-GLUE datasets and predicts on the VL-GLUE dataset) 

- To evaluate a particular dataset, refer the appropriate path under "home" variable and set the appropriate value of the "tasktype" (from 2way, 4way or 27way- depending on how many number of answer choices the questions in the dataset has)

- For example, to evaluate Winoground dataset in ReadyForFineTuning directory (from the Dropbox link)\
```home = "/path/to/ReadyForFineTuning/Winoground/"```\
```tasktype = "2way"```\
Then execute cells in the jupyter notebook one-by-one.\
Intermediate processed dataset in .jsonl format and final predictions in .csv format will be saved in the working directory. 

- For Finetune notebooks, GPU is recommended. 


