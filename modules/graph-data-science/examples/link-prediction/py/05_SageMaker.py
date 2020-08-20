# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Training a model with Amazon SageMaker Autopilot
#
# In this notebook we're going to learn how to train a model with Amazon's SageMaker Autopilot. Amazon SageMaker Autopilot is an automated machine learning (commonly referred to as AutoML) solution for tabular datasets. 

# tag::pip-install-sagemaker[]
# !pip install sagemaker
# end::pip-install-sagemaker[]

# +
import pandas as pd

# tag::imports[]
import sagemaker
import boto3
import os
# end::imports[]
# -

# We can create our classifier with the following code:

# +
# Load the CSV files saved in the train/test notebook

# tag::load-csv-files[]
df_train_under = pd.read_csv("data/df_train_under_all.csv")
df_test_under = pd.read_csv("data/df_test_under_all.csv")
# end::load-csv-files[]
# -

# tag::train-features[]
df_train_under.drop(columns=["node1", "node2"]).sample(5, random_state=42)
# end::train-features[]

# tag::test-features[]
df_test_under.drop(columns=["node1", "node2"]).sample(5, random_state=42)
# end::test-features[]

# +
# tag::prerequisites[]
boto_session = boto3.Session(aws_access_key_id=os.environ["ACCESS_ID"],
         aws_secret_access_key= os.environ["ACCESS_KEY"])

region = boto_session.region_name

session = sagemaker.Session(boto_session=boto_session)
bucket = session.default_bucket()
prefix = 'sagemaker/autopilot-dm/link-prediction-2020-08-17'

role = os.environ["SAGEMAKER_ROLE"]

sm = boto_session.client(service_name='sagemaker',region_name=region)
# end::prerequisites[]
# -

# ## Upload the dataset to Amazon S3
#
# Copy the file to Amazon Simple Storage Service (Amazon S3) in a .csv format for Amazon SageMaker training to use.

# +
# tag::upload-dataset-s3[]
train_file = 'train_data_link-pred-ordered.csv';
train_data.to_csv(train_file, index=False, header=True)
train_data_s3_path = session.upload_data(path=train_file, key_prefix=prefix + "/train")
print('Train data uploaded to: ' + train_data_s3_path)

test_file = 'test_data_link-pred-ordered.csv';
test_data_no_target.to_csv(test_file, index=False, header=False)
test_data_s3_path = session.upload_data(path=test_file, key_prefix=prefix + "/test")
print('Test data uploaded to: ' + test_data_s3_path)
# end::upload-dataset-s3[]

# +
# tag::evaluation-imports[]
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
# end::evaluation-imports[]

# tag::evaluation-functions[]
def evaluate_model(predictions, actual):
    return pd.DataFrame({
        "Measure": ["Accuracy", "Precision", "Recall"],
        "Score": [accuracy_score(actual, predictions), 
                  precision_score(actual, predictions), 
                  recall_score(actual, predictions)]
    })
# end::evaluation-functions[]

def feature_importance(columns, classifier):        
    display("Feature Importance")
    df = pd.DataFrame({
        "Feature": columns,
        "Importance": classifier.feature_importances_
    })
    df = df.sort_values("Importance", ascending=False)    
    ax = df.plot(kind='bar', x='Feature', y='Importance', legend=None)
    ax.xaxis.set_label_text("")
    plt.tight_layout()
    plt.show()


# +
# tag::test-model[]
predictions = classifier.predict(df_test_under[columns])
y_test = df_test_under["label"]

evaluate_model(predictions, y_test)
# end::test-model[]
# -

evaluate_model(predictions, y_test).to_csv("data/model-eval.csv", index=False)

feature_importance(columns, classifier)
