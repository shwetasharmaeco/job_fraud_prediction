import matplotlib.pyplot as plt
import unicodedata
import string
import nltk
from nltk.corpus import stopwords
stopwords_ = set(stopwords.words('english'))
punctuation_ = set(string.punctuation)
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import re
from nltk.stem.snowball import SnowballStemmer
stemmer_snowball = SnowballStemmer('english')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer("english")
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score

def balance_train_data(X, y, method=None):
    '''
    Balances the data passed in according to the specified method.
    '''
    if method == "None":
        return X, y

    elif method == 'undersampling':
        rus = RandomUnderSampler()
        X_train, y_train = rus.fit_resample(X, y)
        return X_train, y_train

    elif method == 'oversampling':
        ros = RandomOverSampler()
        X_train, y_train = ros.fit_resample(X, y)
        return X_train, y_train

    elif method == 'smote':
        smote = SMOTE()
        X_train, y_train = smote.fit_resample(X, y)
        return X_train, y_train

    elif method == 'both':
        smote = SMOTE(sampling_strategy=0.75)
        under = RandomUnderSampler(sampling_strategy=1)
        X_train, y_train = smote.fit_resample(X, y)
        X_train, y_train = under.fit_resample(X_train, y_train)
        return X_train, y_train

    else:
        print('Incorrect balance method')
        return
    
    
def run_model(estimators, X, y, sampling_method,ax, names ):
    
    kf = KFold(n_splits=5, shuffle=True)
    f1 = []
    precisions=[]
    recalls=[]
    
    for i in range(len(estimators)):
        precisions.append([]) 
        recalls.append([])
        f1.append([])
    
    for train_idx, test_idx in kf.split(X):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]
        
        
        
        X_train_new, y_train_new = balance_train_data(X_train, y_train, method=sampling_method)
        
        for i, estimator in enumerate(estimators):
            estimator.fit(X_train_new, y_train_new)
            y_probs = estimator.predict_proba(X_test)
            y_probs = y_probs[:,1]
            y_pred = estimator.predict(X_test)
            
            auc_precision, auc_recall, _ = precision_recall_curve(y_test, y_probs)
            auc_f1, auc_auc = f1_score(y_test, y_pred), auc(auc_recall, auc_precision)
            
            precisions[i].append(precision_score(y_test, y_pred))
            recalls[i].append(recall_score(y_test, y_pred))
            f1[i].append(f1_score(y_test, y_pred))
            
    x = range(0, 5)
    colormap = {0 : 'r',
                1 : 'b',
                2 : 'g', 
                3 : 'c', 
                4 : 'm'}
    
    
    for i in range(len(estimators)):
        ax.plot(x, precisions[i], c=colormap[i], 
                linewidth=1, linestyle='-',
                label='%s Precision' % names[i])
        ax.plot(x, recalls[i], c=colormap[i], 
                linewidth=1, linestyle='--',
                label='%s Recall' % names[i])
        ax.plot(x, f1[i], c=colormap[i],
                linewidth=1, linestyle='-.',
                label='%s f1_score' % names[i])
        

def remove_accents(input_str):
    '''
    This function is to remove the accents
    '''
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii.decode()   

def tokenize_and_stem(text):
    tokens = nltk.tokenize.word_tokenize(text)
    # strip out punctuation and make lowercase
    tokens = [token.lower().strip(string.punctuation)
              for token in tokens if token.isalnum()]

    # now stem the tokens
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    tokens = [stemmer.stem(token) for token in tokens]
    

    return " ".join(tokens)



def clean_cols(data, col):
    data[col] = data[col].str.replace(r"(\s*\<.*?\>\s*)", " ").str.strip()
    data[col] = data[col].str.replace(r"(\s*\#.*?\#\s*)", " ").str.strip()
    data[col] = data[col].str.split().str.join(' ')
    data[col] = data[col].apply(lambda x: " ".join([i for i in x.lower().split() if i not in stopwords_]))
    data[col] = data[col].apply(lambda x: "".join([i for i in x if i not in punctuation_]))
    data[col] = data[col].apply(lambda x: remove_accents(x))
    data[col] = data[col].apply(lambda x: tokenize_and_stem(x))

def fill_nulls(data):
    data["location"] = data["location"].fillna("Not Specified")
    data["department"] = data["department"].fillna("Not Specified")
    data["salary_range"] = data["salary_range"].fillna("Not Specified")
    data["salary_range"] = data["salary_range"]\
    .apply(lambda x: "Specified" if x != "Not Specified" else x)
    data["company_profile"] = data["company_profile"].fillna("")
    data["requirements"] = data["requirements"].fillna("")
    data["benefits"] = data["benefits"].fillna("")
    data["employment_type"] = data["employment_type"].fillna("Other")
    data["required_education"]=data["required_education"].fillna("Unspecified")
    data["required_experience"]=data["required_experience"]\
        .fillna("NotSpecified")
    data["industry"] = data["industry"].fillna("Not Specified")
    data["function"] = data["function"].fillna("Other")
    
    
def plot_bar(data, col):
    
    new_col = data.groupby([col,"fraudulent"]).size().reset_index(name="counts")

    fig = plt.figure()

    ax = new_col.pivot(col, "fraudulent", "counts").plot(kind='bar', figsize=(10,5), title = "Distribution of status per category")

    ax.set_xlabel(col, fontsize=20)
    ax.set_ylabel("Number of Jobs", fontsize=20)

    fig.savefig(f"images/{col}", bbox_inches='tight');
    

def binarize(data, col):
     data[col] = data[col].replace(["f", "t"], [0,1])
        
        
        
        
def plot_text(data, col):
    
    fig = plt.figure()
    fig,(ax1,ax2)= plt.subplots(ncols=2, figsize=(17, 5))
    
    length = data[data["fraudulent"]==1][col].str.len()
    ax1.hist(length,bins = 20,color='red')
    ax1.set_title('Fraud job Post')
    
    length = data[data["fraudulent"]==0][col].str.len()
    ax2.hist(length, bins = 20, color="green")
    ax2.set_title('Real job Post')
    fig.suptitle(f'Characters in {col}')
    fig.savefig(f"images/{col}", bbox_inches='tight')
    plt.show();
    
    
    
    
from sklearn.feature_extraction.text import CountVectorizer

def get_top_n_words(corpus, n=None):
 
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    
    return words_freq[:n]


def plot_wordcloud(data, col):   

    fig = plt.figure()
    fig,(ax1,ax2)= plt.subplots(ncols=2, figsize=(17, 5))
    
    common_words = get_top_n_words(data[data["fraudulent"]==1][col], 20)
    plot_words = ""
    for word, freq in common_words:
        plot_words+=word + " "

        wordcloud = WordCloud(width = 500, height = 500, 
                    background_color ='white', 
                    stopwords = stopwords, 
                    min_font_size = 10).generate(plot_words) 
                       
        ax1.imshow(wordcloud)
        ax1.set_title('Fraud job Post')
    
    common_words = get_top_n_words(data[data["fraudulent"]==0][col], 20)
    plot_words = ""
    for word, freq in common_words:
        plot_words+=word + " "

        wordcloud = WordCloud(width = 500, height = 500, 
                    background_color ='white', 
                    stopwords = stopwords, 
                    min_font_size = 10).generate(plot_words) 
                        
        ax2.imshow(wordcloud) 
        ax2.set_title('Real job Post')
        
    fig.suptitle(f'Most common words in {col}')
    ax1.axis('off')
    ax2.axis('off')
    fig.savefig(f"images/{col}_wordcloud", bbox_inches='tight');
    plt.show()
    
    
def generate_model_report(y_actual, y_predicted):
    print('Accuracy: %.3f' % accuracy_score(y_actual, y_predicted))
    print('Precision: %.3f' % precision_score(y_actual, y_predicted))
    print( 'Recall: %.3f' % recall_score(y_actual, y_predicted))
    print('F1 score: %.3f' % f1_score(y_actual, y_predicted))
    
    
def clean_features(df):
    df["location"] = df["location"].apply(lambda x: x.split(","))
    df["location"] = df["location"].apply(lambda x: str(x[0])) # Only keeping countries from location column
    df.loc[~df.location.isin(["US", "GB", "GR", "CA", "DE", "Not Specified", "NZ","IN", "AU", "PH", "NL","BE", "IE"]), "location"]= "Other"
    df.loc[df['required_education'].str.contains('Vocational'), 'required_education'] = "Vocational"
    df.loc[df['required_education'].str.contains('High School'), 'required_education'] = "High School"
    df.loc[df['required_education'].str.contains('College'), 'required_education'] = "Bachelor's Degree"
    df.loc[df['industry'].str.contains('Computer'), 'industry'] = "IT"
    df.loc[df['industry'].str.contains('Information'), 'industry'] = "IT"
    df.loc[df['industry'].str.contains('Internet'), 'industry'] = "IT"
    df.loc[df['industry'].str.contains('Insurance'), 'industry'] = "Financial Services"
    df.loc[df['industry'].str.contains('Accounting'), 'industry'] = "Financial Services"
    df.loc[df['industry'].str.contains('Health'), 'industry'] = "Health Care"
    df.loc[df['industry'].str.contains('E-Learning'), 'industry'] = "Education"
    df.loc[df['industry'].str.contains('Education'), 'industry'] = "Education"
    df.loc[df['industry'].str.contains('Recruiting'), 'industry'] = "HR"
    df.loc[df['industry'].str.contains('Human Resources'), 'industry'] = "HR"
    
    # value_counts less than 100 set to "Others"
    counts = df["industry"].value_counts()
    idx = counts[counts.lt(100)].index

    df.loc[df["industry"].isin(idx), "industry"] = 'Others'
    