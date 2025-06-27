# Task-Adaptive-Debiasing-with-SCM-for-Sentiment-Analysis

## 1. Repository contents

| Notebook | Purpose |
|----------|---------|
| **Exploratory_Data_Analysis.ipynb** | Quick look at raw data |
| **Data_Preparation_and_Feature_Engineering.ipynb** | Clean text, compute Warmth / Competence scores |
| **Baseline_Model_Training.ipynb** | Train baseline |
| **Debias_Training_(Dynamic_and_Fixed_).ipynb** | Train task-adaptive and Fixed models |
| **Diagnostics_and_Statistical_Tests.ipynb** | Triple-curve plots + DP/KS/Bootstrap/Sign-test |

## 2. Data

This project uses three public sentiment analysis datasets:

**1.Rotten Tomatoes Movie Reviews** (Kaggle)  
  A widely used corpus of movie reviews labeled on a 5-point scale from very negative to very positive.
 
> **Pang, B. & Lee, L. (2005).** Seeing Stars: Exploiting Class Relationships for Sentiment  
> Categorization with Respect to Rating Scales. _ACL 2005_, 115–124.  
> https://aclanthology.org/P05-1015/
 
> **Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A. Y., & Potts, C. (2013).**  
> Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank. _EMNLP 2013_, 1631–1642.  
> https://aclanthology.org/D13-1170/
 
> **Cukierski, W. (2014).** Sentiment Analysis on Movie Reviews [Kaggle competition].  
> https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews 

**2.Amazon Product Reviews**  
  Millions of product reviews (e.g. Appliances, Fashion, Beauty, Pet Supplies) with 1–5 star ratings as sentiment labels, allowing analysis across different product types.
 
> **Hou, Y., Li, J., He, Z., Yan, A., Chen, X., & McAuley, J. (2024).** Bridging Language and Items for Retrieval and Recommendation. _arXiv preprint arXiv:2403.03952_. 
> https://doi.org/10.48550/arXiv.2403.03952

**3.SemEval 2016 Task 4 Twitter Sentiment**  
  A collection of tweets annotated on a 5-class scale (very negative, negative, neutral, positive, very positive).
 
> **Nakov, P., Ritter, A., Rosenthal, S., Stoyanov, V., & Sebastiani, F. (2016).**  
> SemEval-2016 Task 4: Sentiment Analysis in Twitter. In _Proceedings of the 10th International Workshop on Semantic Evaluation_ (SemEval ’16), San Diego, California. Association for Computational Linguistics. 
> https://doi.org/10.48550/arXiv.1912.01973

We also used the comprehensive stereotype content dictionaries developed by Nicolas, Bai & Fiske (2021).
> **Nicolas G, Bai X, Fiske ST**
Comprehensive stereotype content dictionaries using a semi-automated method. Eur J Soc Psychol. 2021; 51: 178–196. 
> https://doi.org/10.1002/ejsp.2724

## 3. Environment setup

Open **`Data_Preparation_and_Feature_Engineering.ipynb`** and locate the first code cell.  
Remove the  **`#`** from the line

```python
!pip install -r requirements.txt -q
```

Run that cell once, then it will install every required library.

## 4. Running experiment

### Execution Order 
01`Data_Preparation_and_Feature_Engineering.ipynb` 

02`Baseline_Model_Training.ipynb `             

03`Debias_Training_(Dynamic_and_Fixed_).ipynb `  

04`Diagnostics_and_Statistical_Tests.ipynb `         

### Inspecting results

All outputs (pickles, figures, CSVs) are written to `output/`.   

### Runtime

On a single NVIDIA A100 (40 GB) GPU in Google Colab, a full pipeline takes approximately:
- **Baseline**: ~ 2 hours  
- **Task-adaptive α grid**: ~ 4-5 hours  
- **Fixed λ**: ~ 4-5 hours  
- **Diagnostics & stats**: < 1 hour 



## 5. Trained checkpoints

If you need them, you can download the trained checkpoints from my public Google Drive folder:

**Google Drive link:**  
<https://drive.google.com/drive/folders/1_Im0EVDWaL6nczWGKsGae7ucszIci-Y4?usp=sharing>


`results_<DATASET>_ep<e>`: Baseline model trained for e epochs. 
`e.g. results_Amazon_ep2`

`results_debiased_<DATASET>_aXXX`: Task-adaptive ( a020 means α = 0.20)
`e.g. results_debiased_Amazon_a020`

`results_debiased_<DATASET>_fixed_aXXX`
Fixed debias  `e.g.results_debiased_Amazon_fixed_a020`

Each of these folders holds one or more sub-directories named  `checkpoint-<step>/`.

After you download a checkpoint, place its folder under `output` (or adjust the path in the notebooks accordingly) and run the notebooks as usual.





