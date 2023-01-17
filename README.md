# Semi-supervised_learning_for_part_of_speech_tagging
## NLP Capstone Project with Columbia University and J.P. Morgan

## 1 Preparations

---

### 1.1 Dependencies Preparations

- Install the dependencies of the project.

```
pip install -r requirements.txt
```    

---

### 1.2 Data Preparations

- Download **SANCL 2012 dataset (Petrov and McDonald, 2012)** and save data as the following structure in the data directory.

```bash
.
├── data
│   ├── datasets
│   │   └── PennTreebank
│   └── gweb_sancl
│       ├── pos_fine
│       │   ├── answers
│       │   │   ├── gweb-answers-dev.conll
│       │   │   └── gweb-answers-test.conll
│       │   ├── emails
│       │   │   ├── gweb-emails-dev.conll
│       │   │   └── gweb-emails-test.conll
│       │   ├── newsgroups
│       │   │   ├── gweb-newsgroups-dev.conll
│       │   │   └── gweb-newsgroups-test.conll
│       │   ├── reviews
│       │   │   ├── gweb-reviews-dev.conll
│       │   │   └── gweb-reviews-test.conll
│       │   ├── weblogs
│       │   │   ├── gweb-weblogs-dev.conll
│       │   │   └── gweb-weblogs-test.conll
│       │   └── wsj
│       │       ├── gweb-wsj-dev.conll
│       │       ├── gweb-wsj-test.conll
│       │       └── gweb-wsj-train.conll
│       └── unlabeled
│           └── gweb-answers.unlabeled.txt
```

---

### 1.3 Directory Preparation
- Make sure you have the following directories in the root before running the script

```bash
.
├── Analysis_int_res.ipynb
├── Analysis_output_Online_fixed_self_learning.ipynb
├── Analysis_output_Online_nonfixed_self_learning.ipynb
├── Base_model_s1.ipynb
├── LICENSE
├── Online_fixed_self_learning_s1.ipynb
├── Online_nonfixed_self_learning_s1.ipynb
├── Online_token_self_learning_each_tag_s1.ipynb
├── Online_token_self_learning_s1.ipynb
├── README.md
├── Scratch_fixed_self_learning_s1.ipynb
├── Scratch_nonfixed_self_learning_s1.ipynb
├── Supervised_learning_model.ipynb
├── analysis.py
├── build_model.py
├── create_pseudo_data.py
├── create_pseudo_data_by_tokens.py
├── docs
├── intermediate_result
├── metrics
├── model
├── online_fixed_self_learning.py
├── requirements.txt
├── result
├── scripts
├── setup.sh
└── utils.py
```

---

## 2 Directory Structure

Here follows the brief introduction about the specific details for each directory:
1. **data**: store the data we are gonna use
2. **model**: store the models in each iteration   
3. **intermediate_result**: store the intermediate results of self training, like top N sentences, top N tokens and probability list
4. **result**: store the testing results and plots
5. **metrics**: store the metrics at each loop after self training including precision, f1 and recall with different average methods
6. **docs**: slides and docs

---

## 3 How to run this project

- Run **Base_model_s1** to train the base model
- Run **Online_fixed_self_learning_s1** or other notebooks to run the corresponding self-training method (To learn the details for the different self-training methods, please see the slides)

---

## 4 Results and Conclusions

- Please check the slides for detailed results.
- Bert has already performed very well on sequence labeling, even in domain adaptation.
- Self-training hardly improves the model performance on domain adaptation in either sentence-wise or token-wise
- Learning rate is a very important parameter in Semi-supervised learning, and a higher learning rate can easily cause catastrophic forgetting.

