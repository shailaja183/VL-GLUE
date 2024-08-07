# VL-GLUE
Code and data for Visuo-Linguistic GLUE Benchmark

# Dataset Link

- Access the dataset at the ReadyForFineTuning repository from Dropbox Link (Size limitation on GitHub)
<https://www.dropbox.com/scl/fo/5nsctleolkfradwo5zmmn/AKTGRdvqZSG5ZrIdno6Ka7o?rlkey=6i4uae4gpwocyevigs7jzxdie&dl=0>

- The dataset contains multiple sub-folders (CLEVR_HYP, MultimodalQA, MuMuQA, VGSI, VLQAv1 [BlocksWorld, Charts, COCO, NLVR, PIQA, and RecipeQA], WebQA, WinoGround), each being a distinct subset of the VL-GLUE. Baseline models are executed separately for each sub-folder.

# Multi-modal Baselines (Interactive Demo)

- There are three main jupyter notebooks (necessary python libraries and installations for each model are incorporated in the notebooks) 
1. ```Multimodal_PredictionOnly_CLIP_GIT_VILT.ipynb```\
(Runs predictions over VL-GLUE datasets using CLIP, GIT and VILT models pretrained on VQA tasks)
 
2. ```Multimodal_Finetune_VILT.ipynb```\
(Performs fine-tuning of VILT models pretrained on VQA tasks with VL-GLUE datasets and predicts on the VL-GLUE dataset) 

3. ```Multimodal_Finetune_VisualBERT.ipynb```\
(Performs fine-tuning of VisualBERT models pretrained on VQA tasks with VL-GLUE datasets and predicts on the VL-GLUE dataset) 

- To evaluate a particular dataset, configure the appropriate path under "home" variable (the base directory for the individual dataset), "imroot" variable (the directory where images are stored for the dataset) and "tasktype" (from "2way", "4way" or "27way"- depending on the number of answer choices each question in the dataset has)

- For example, to evaluate Winoground dataset in ReadyForFineTuning directory (from the Dropbox link)\
```home = "/path/to/ReadyForFineTuning/Winoground/"```\
```imroot = home+"images_nkmr/merged_images/"```\
```tasktype = "2way"```

- Then execute cells in the jupyter notebook one-by-one. Intermediate processed dataset in .jsonl format and final predictions in .csv format will be saved in the working directory. 

- For Finetune notebooks, GPU is recommended. 

# Batch-Run Results

- The Results/ directory contains files (processed datasets/scripts/prediction results) for baseline models whose which performance is reported with respect to each VL-GLUE dataset in the paper

- Baselines included:
1. Unimodel baselines: QuestionOnly_GPT3, PassageQuestion_RobertaRace, ImageQuestionOnly_BLIP
2. Multimodal baselines (prediction-only): PassageImageQuestion_BLIP, PassageImageQuestion_CLIP
3. Multimodal baselines (fine-tune): PassageImageQuestion_ViLT_Finetune, PassageImageQuestion_VisualBERT_Finetune 

