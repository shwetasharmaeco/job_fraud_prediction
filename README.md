# Fraud Detection
images: EDA graphs

dating.ipynb: EDA, Base model, TD-IDF, LSTM analysis

helpers.py: helper functions

# Motivation and Goal
Having fraud jobs on on one's website/app can result in losing users interest or flagging non-fraud jobs as fraud jobs may upset genuine companies and they might using/ posting their openings on the site. Hence, it is really important to correctly predict fraud and not fraud jobs.

# Methods Used

Data Visualization

Natural Language Processing

Vectorization

Logistic Regression, Random Forest, Gradient Boosting

Recurrent Neural Netrwork: Long short-term memory (LSTM)

# Data Pipeline

Download dataset from The Employment Scam Aegean Dataset (EMSCAD).

Select and clean relevant features.

Perfom exploratory data analysis.

Build baseline model.

Build up the model pipelines to train the model.

Choose the best performing model to apply to the jobs dataet, and predict if a job listing is a fraud listing or not.


# Data

The dataset consists of 17,880 real-life job ads that aims at providing a clear picture of the Employment Scam problem including title of the job, location of the job, department, salary range, company, description, benefits, employment_type, required experience, required education', industry. Each row is flagged as fraud or not fraud.

### String Columns
| | | 
|-|-|
|Name| Description 
|Title| The title of the job ad entry.
|Location| Geographical location of the job ad.
|Department| Corporate department (e.g. sales).
|Salary range| Indicative salary range (e.g. $50,000-$60,000)

###  Binary 
| | | 
|-|-|
|Telecommuting| True for telecommuting positions. 
|Company logo| True if company logo is present
|Questions| True if screening questions are present.
|**Fraudulent**| 	Classification attribute.

### HTML fragment
| | | 
|-|-|
|Company profile| A brief company description. 
|Description| The details description of the job ad.
|Requirements| Enlisted requirements for the job opening.
|Benefits| 	Enlisted offered benefits by the employer.

### Nominal
| | | 
|-|-|
|Employment type| Full-type, Part-time, Contract, etc. 
|Required experience| Executive, Entry level, Intern, etc
|Required education| Doctorate, Master’s Degree, Bachelor, etc.
|Industry| Automotive, IT, Health care, Real estate, etc.
|Function| Consulting, Engineering, Research, Sales etc.


