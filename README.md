# Semi-supervised_learning_for_part_of_speech_tagging
## NLP Capstone Project with Columbia University and J.P. Morgan

**Please check dev branch for the latest version.**

## Preparations
### Dependencies Preparations
0. (Ignore this step) Check the cuda version on GCP as 
```
sudo nvidia-smi
```
1. No need to configure virtual environment for this machine. Just use the base environment and install the dependencies as
```
pip install -r requirements.txt
```    

### Model/Data Preparations
1. Download [model/base_model.pt](https://drive.google.com/drive/u/2/folders/1NC0ZC0t8ncA8KAuZ8igtyQeWA7gVAWCw) from Google Drive into your `model/` directory.
2. Download [data/gweb_sancl/](https://drive.google.com/drive/u/2/folders/1sh9z8TH8Imn1v1NkzCLzLTC4ieQH5ojV) from Google Drive into your `data/` directory.

### Directory Preparation
Make sure you have the following directories in the root before running the script

```bash
.
├── Analysis_int_res.ipynb
├── Analysis_output_Online_fixed_self_learning.ipynb
├── Analysis_output_Online_nonfixed_self_learning.ipynb
├── LICENSE
├── Online_fixed_self_learning_v5.ipynb
├── Online_nonfixed_self_learning_v5.ipynb
├── Online_token_self_learning_v5.ipynb
├── README.md
├── Scratch_fixed_self_learning_v5.ipynb
├── Scratch_nonfixed_self_learning_v5.ipynb
├── Scratch_token_self_learning_v5.ipynb
├── analysis.py
├── build_model.py
├── create_pseudo_data.py
├── create_pseudo_data_by_tokens.py
├── data
│   └── gweb_sancl
│       ├── pos_fine
│       │   ├── answers
│       │   ├── emails
│       │   ├── newsgroups
│       │   ├── reviews
│       │   ├── weblogs
│       │   └── wsj
│       └── unlabeled
│           └── gweb-answers.unlabeled.txt
├── docs
├── intermediate_result
├── metrics
├── model
├── plots_tags
├── requirements.txt
├── result
├── scripts
├── setup.sh
└── utils.py

```


## Directory Structure
Here follows the brief introduction about the specific details for each directory:
1. metrics: store the metrics at each loop after self training including precision, f1 and recall    
2. plots: store the plots for metrics at different parameter settings    
3. model: store the model settings to save time   
4. data: store the data we are gonna use   
5. docs: store the meeting records for the project  
6. pickles: store the serialized python object after self-training for future usages.

## Results
