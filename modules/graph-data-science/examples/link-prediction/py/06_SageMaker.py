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
#
# First let's install the sagemaker library:

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

# Let's first load the features that we engineered in the previous notebook:

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
boto_session = boto3.Session(
    aws_access_key_id=os.environ["ACCESS_ID"],
    aws_secret_access_key= os.environ["ACCESS_KEY"])

region = boto_session.region_name

session = sagemaker.Session(boto_session=boto_session)
bucket = session.default_bucket()
prefix = 'sagemaker/link-prediction-developer-guide'

role = os.environ["SAGEMAKER_ROLE"]

sm = boto_session.client(service_name='sagemaker',region_name=region)
# end::prerequisites[]
# -

# ## Upload the dataset to Amazon S3
#
# Copy the file to Amazon Simple Storage Service (Amazon S3) in a .csv format for Amazon SageMaker training to use.

# +
# tag::upload-dataset-s3[]
train_file = 'data/upload/train_data.csv';
df_train_under.to_csv(train_file, index=False, header=True)
train_data_s3_path = session.upload_data(path=train_file, key_prefix=prefix + "/train")
print('Train data uploaded to: ' + train_data_s3_path)

test_file = 'data/upload/test_data.csv';
df_test_under.to_csv(test_file, index=False, header=False)
test_data_s3_path = session.upload_data(path=test_file, key_prefix=prefix + "/test")
print('Test data uploaded to: ' + test_data_s3_path)
# end::upload-dataset-s3[]
# -

# ## Setting up the SageMaker Autopilot Job
#
# Now that we've uploaded the dataset to S3, we're going to call Autopilot and have it find the best model for the dataset. Autopilot's required inputs are as follows:
#
# * Amazon S3 location for input dataset and for all output artifacts
# * Name of the column of the dataset you want to predict (`label` in this case) 
# * An IAM role
#
# The training CSV file that we send to Autopilot should have a header row.

# +
# tag::autopilot-setup[]
input_data_config = [{
      'DataSource': {
        'S3DataSource': {
          'S3DataType': 'S3Prefix',
          'S3Uri': 's3://{}/{}/train'.format(bucket,prefix)
        }
      },
      'TargetAttributeName': 'label'
    }
  ]

automl_job_config = {
    "CompletionCriteria": {
        "MaxRuntimePerTrainingJobInSeconds": 300,
        "MaxCandidates": 5,
    }
}

output_data_config = {
    'S3OutputPath': 's3://{}/{}/output'.format(bucket,prefix)
  }
# end::autopilot-setup[]
# -

# ## Launching the SageMaker Autopilot Job
#
# We can now launch the Autopilot job by calling the `create_auto_ml_job` API. 

# +
# tag::autopilot-launch[]
from time import gmtime, strftime, sleep

timestamp_suffix = strftime('%Y-%m-%d-%H-%M-%S', gmtime())
auto_ml_job_name = 'automl-link-' + timestamp_suffix
print('AutoMLJobName: ' + auto_ml_job_name)

sm.create_auto_ml_job(AutoMLJobName=auto_ml_job_name,
                      InputDataConfig=input_data_config,
                      OutputDataConfig=output_data_config,
                      AutoMLJobConfig=automl_job_config,
                      RoleArn=role)
# end::autopilot-launch[]
# -

# ### Tracking SageMaker Autopilot job progress
#
# SageMaker Autopilot job consists of the following high-level steps : 
#
# * Analyzing Data, where the dataset is analyzed and Autopilot comes up with a list of ML pipelines that should be tried out on the dataset. The dataset is also split into train and validation sets.
# * Feature Engineering, where Autopilot performs feature transformation on individual features of the dataset as well as at an aggregate level.
# * Model Tuning, where the top performing pipeline is selected along with the optimal hyperparameters for the training algorithm (the last stage of the pipeline). 

# +
print ('JobStatus - Secondary Status')
print('------------------------------')

auto_ml_job_name = "automl-link-2020-08-20-09-25-03"

# tag::autopilot-track-progress[]
describe_response = sm.describe_auto_ml_job(AutoMLJobName=auto_ml_job_name)
print (describe_response['AutoMLJobStatus'] + " - " + describe_response['AutoMLJobSecondaryStatus'])
job_run_status = describe_response['AutoMLJobStatus']
    
while job_run_status not in ('Failed', 'Completed', 'Stopped'):
    describe_response = sm.describe_auto_ml_job(AutoMLJobName=auto_ml_job_name)
    job_run_status = describe_response['AutoMLJobStatus']
    
    print (describe_response['AutoMLJobStatus'] + " - " + describe_response['AutoMLJobSecondaryStatus'])
    sleep(30)
# end::autopilot-track-progress[]

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
