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
%config InlineBackend.figure_format = 'retina'

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
    