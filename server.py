
############# all imports  ###########

from flask import (Flask, render_template, request, flash, session,
                   redirect,jsonify,json)
from jinja2 import StrictUndefined
import urllib.request
import psycopg2
from flask_sqlalchemy import SQLAlchemy
import pickle
import numpy as np
import pandas as pd
from helper_func import *
######################################




########### Flask setup ###############

app = Flask(__name__)
app.secret_key = "dev"
app.jinja_env.undefined = StrictUndefined

######################################


@app.route('/', methods=['POST', 'GET'])
def home():
    return render_template('homepage.html')



@app.route('/get-jobs', methods=['POST', 'GET'])
def get_jobs():

    ########## reading pickled files ###########

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    tfidf = pickle.load(open("tfidf.pkl", 'rb'))


    cat_nlp_test = pd.read_pickle("cat_nlp_test")

    #############################################




    ############ setting up postgres ##############

    db = SQLAlchemy()

    conn = psycopg2.connect(dbname='postgres', user='postgres', host='localhost', port='5432', password='password')

    cur = conn.cursor()
    conn.autocommit = True
    cur.close() # This is optional
    conn.close() # Closing the connection also closes all cursors

    conn = psycopg2.connect(dbname='jobs_fraud', user='postgres', host='localhost', port='5432', password='password')
    cur = conn.cursor()

    query = '''
        SELECT *
        FROM jobs
        ORDER BY random()
        LIMIT 100;

            '''
    cur.execute(query)

    table = cur.fetchall()

    #################################################




    ############### reading postgres table in pandas #######################

    data = pd.DataFrame(table,columns=['title', 'location', 'department', 'salary_range', 'company_profile',
        'description', 'requirements', 'benefits', 'telecommuting',
        'has_company_logo', 'has_questions', 'employment_type',
        'required_experience', 'required_education', 'industry', 'function',
        'fraudulent', 'in_balanced_dataset'])

    data_testing = data.copy()

    ########################################################################




    ######### cleaning columns for making predictions ######################

    data_testing.drop("fraudulent", axis= 1, inplace=True)
    data_testing.drop('in_balanced_dataset', axis= 1, inplace=True)

    binary_cols = ["telecommuting", "has_company_logo","has_questions"]

    for col in binary_cols:
        binarize(data_testing,col)

    fill_nulls(data_testing)

    text_cols = ["title", "company_profile", "description", "requirements", "benefits"]

    for col in text_cols:
        clean_cols(data_testing,col)

    data_testing["text"] = data_testing["title"] + data_testing["company_profile"] + data_testing["description"] + data_testing["requirements"] + data_testing["benefits"]

    clean_features(data_testing)

    rf_test = data_testing[['location', 'industry', 'salary_range', 'telecommuting', 'has_company_logo', 'has_questions', 'employment_type', 'required_experience', 'required_education', 'text']]
    rf_cat_df = rf_test[['location', 'industry', 'salary_range', 'telecommuting', 'has_company_logo', 'has_questions', 'employment_type', 'required_experience', 'required_education']]
    rf_fit_df_dummy = pd.get_dummies(rf_cat_df)

    rf_text_cat = pd.concat([rf_fit_df_dummy, rf_test["text"]], axis = 1)

    tfidf_matrix =  tfidf.transform(rf_text_cat.text.values.astype('U'))

    data_nlp = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names())


    data_nlp.reset_index(drop=True, inplace=True)
    rf_text_cat.reset_index(drop=True, inplace=True)
    df_nlp_cat = pd.concat([rf_text_cat, data_nlp], axis=1)
    df_nlp_cat = df_nlp_cat.drop("text", axis=1)

    df_nlp_cat = df_nlp_cat.fillna(0)

    # Get missing columns in the training test
    missing_cols = set(cat_nlp_test.columns) - set(df_nlp_cat.columns)
    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        df_nlp_cat[c] = 0
    # Ensure the order of column in the test set is in the same order than in train set
    df_nlp_cat = df_nlp_cat[cat_nlp_test.columns]

    ###############################################################



    ############## making predictions ############################

    y_preds = model.predict_proba(df_nlp_cat)[:,1]

    y_preds_test = np.array(["Fraud" if val >=0.25 else "Not fraud" for val in y_preds])

    y_preds_df = pd.DataFrame(y_preds_test)

    data.reset_index(drop=True, inplace=True)
    y_preds_df.reset_index(drop=True, inplace=True)

    final_table = pd.concat([y_preds_df,data], axis = 1).reset_index(drop=True)

    final_table = final_table.rename(columns={0:"Flag"})

    

    ##############################################################


    return render_template('index.html',  tables=[final_table.to_html(index=False, classes='data')], titles=final_table.columns.values)



@app.route('/report', methods=['POST', 'GET'])
def report():
    
    return render_template('report.html', )

if __name__ == '__main__':
    # connect_to_db(app)
    app.run(host='0.0.0.0', debug=True)
