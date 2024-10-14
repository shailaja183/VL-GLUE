# VL-GLUE
Code and data for Visuo-Linguistic GLUE Benchmark

# Dataset Link

- Access the dataset at the ReadyForFineTuning repository from Dropbox Link (Size limitation on GitHub) 
<https://www.dropbox.com/scl/fo/5nsctleolkfradwo5zmmn/AKTGRdvqZSG5ZrIdno6Ka7o?rlkey=6i4uae4gpwocyevigs7jzxdie&dl=0>

- The easiest way to replicate experiments/notebooks in this repository is to upload the entire ReadyForFineTuning directory to Google drive (maintaining the exact same structure) 

- The dataset contains multiple sub-folders (CLEVR_HYP, MultimodalQA, MuMuQA, VGSI, VLQAv1 [BlocksWorld, Charts, COCO, NLVR, PIQA, and RecipeQA], WebQA, WinoGround), each being a distinct subset of the VL-GLUE. 

# Multi-modal Experiments 

- A separate notebook is provided for each dataset corresponding to different baseline. Then execute cells in the jupyter notebook one-by-one. 

- Intermediate processed dataset in .jsonl format and final predictions in .csv format are provided (which are by default saved in the working directory when notebooks are run). 

- For all Multimodal notebooks, GPU is recommended. 

# Batch-Run Results

- The Results/ directory contains prediction results for different baseline models below
1. Unimodel baselines: QuestionOnly_GPT3, PassageQuestion_RobertaRace, ImageQuestionOnly_BLIP
2. Multimodal baselines (prediction-only): PassageImageQuestion_BLIP
3. Multimodal baselines (fine-tune): PassageImageQuestion_ViLT_Finetune, PassageImageQuestion_VisualBERT_Finetune 

