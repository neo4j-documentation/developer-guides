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
from time import gmtime, strftime, sleep
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

timestamp_suffix = strftime('%Y-%m-%d-%H-%M-%S', gmtime())
# timestamp_suffix = "2020-08-20-11-26-33"

prefix = 'sagemaker/link-prediction-developer-guide-' + timestamp_suffix

role = os.environ["SAGEMAKER_ROLE"]

sm = boto_session.client(service_name='sagemaker',region_name=region)
# end::prerequisites[]

print(timestamp_suffix, prefix)
# -

# ## Upload the dataset to Amazon S3
#
# Copy the file to Amazon Simple Storage Service (Amazon S3) in a .csv format for Amazon SageMaker training to use.

# +
# tag::upload-dataset-s3[]
train_columns = [
    "cn", "pa", "tn", "minTriangles", "maxTriangles", "minCoefficient", "maxCoefficient", "sp", "sl", "label"
]
df_train_under = df_train_under[train_columns]

test_columns = [
    "cn", "pa", "tn", "minTriangles", "maxTriangles", "minCoefficient", "maxCoefficient", "sp", "sl"
]
df_test_under = df_test_under.drop(columns=["label"])[test_columns]

train_file = 'data/upload/train_data_binary_classifier.csv';
df_train_under.to_csv(train_file, index=False, header=True)
train_data_s3_path = session.upload_data(path=train_file, key_prefix=prefix + "/train")
print('Train data uploaded to: ' + train_data_s3_path)

test_file = 'data/upload/test_data_binary_classifier.csv';
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
auto_ml_job_name = 'automl-link-' + timestamp_suffix
print('AutoMLJobName: ' + auto_ml_job_name)

sm.create_auto_ml_job(AutoMLJobName=auto_ml_job_name,
                      InputDataConfig=input_data_config,
                      OutputDataConfig=output_data_config,
                      ProblemType="BinaryClassification",
                      AutoMLJobObjective={"MetricName": "Accuracy"},
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

# auto_ml_job_name = "automl-link-2020-08-20-11-26-33"

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
# tag::autopilot-all-candidates[]
candidates = sm.list_candidates_for_auto_ml_job(
    AutoMLJobName=auto_ml_job_name, 
    SortBy='FinalObjectiveMetricValue')['Candidates']

candidates_df = pd.DataFrame({
    "name": [c["CandidateName"] for c in candidates],
    "score": [c["FinalAutoMLJobObjectiveMetric"]["Value"] for c in candidates]
})
candidates_df
# end::autopilot-all-candidates[]

display(candidates_df)
candidates_df.to_csv("data/download/autopilot_candidates.csv", index=False, float_format='%g')

# +
# tag::autopilot-best-candidate[]
best_candidate = sm.describe_auto_ml_job(
    AutoMLJobName=auto_ml_job_name)['BestCandidate']

best_df = pd.DataFrame({
    "name": [best_candidate['CandidateName']],
    "metric": [best_candidate['FinalAutoMLJobObjectiveMetric']['MetricName']],
    "score": [best_candidate['FinalAutoMLJobObjectiveMetric']['Value']]
})
best_df
# end::autopilot-best-candidate[]

display(best_df)
best_df.to_csv("data/download/autopilot_best_candidate.csv", index=False, float_format='%g')
# -

# ### Perform batch inference using the best candidate
#
# Now that we have successfully completed the SageMaker Autopilot job on the dataset, create a model from any of the candidates by using [Inference Pipelines](https://docs.aws.amazon.com/sagemaker/latest/dg/inference-pipelines.html). 

# +
# timestamp_suffix = "automl-link-2020-08-20-09-25-03"

# tag::autopilot-create-model[]
model_name = 'automl-link-pred-model-' + timestamp_suffix

model = sm.create_model(Containers=best_candidate['InferenceContainers'],
                            ModelName=model_name,
                            ExecutionRoleArn=role)

print('Model ARN corresponding to the best candidate is : {}'.format(model['ModelArn']))
# end::autopilot-create-model[]
# -

# ### Evaluating the model
#
# And now we're going to create a transform job based on this model.
# A transform job uses a trained model to get inferences on a dataset and saves these results to S3.

# +
# test_data_s3_path = "s3://sagemaker-us-east-1-715633473519/sagemaker/link-prediction-developer-guide-2020-08-20-11-26-33/train/train_data_copy.csv"
# timestamp_suffix = "2020-08-20-11-26-33"
# timestamp_suffix = strftime('%Y-%m-%d-%H-%M-%S', gmtime())

# tag::autopilot-create-transform-job[]
transform_job_name = 'automl-link-pred-transform-job-' + timestamp_suffix

print(test_data_s3_path, transform_job_name, model_name)

transform_input = {
        'DataSource': {
            'S3DataSource': {
                'S3DataType': 'S3Prefix',
                'S3Uri': test_data_s3_path
            }
        },
        'ContentType': 'text/csv',
        'CompressionType': 'None',
        'SplitType': 'Line'
    }

transform_output = {
        'S3OutputPath': 's3://{}/{}/inference-results'.format(bucket,prefix),
    }

transform_resources = {
        'InstanceType': 'ml.m5.4xlarge',
        'InstanceCount': 1
    }

sm.create_transform_job(TransformJobName = transform_job_name,
                        ModelName = model_name,
                        TransformInput = transform_input,
                        TransformOutput = transform_output,
                        TransformResources = transform_resources
)
# end::autopilot-create-transform-job[]

# +
print ('JobStatus')
print('----------')

# tag::autopilot-track-transform-job[]
describe_response = sm.describe_transform_job(TransformJobName = transform_job_name)
job_run_status = describe_response['TransformJobStatus']
print (job_run_status)

while job_run_status not in ('Failed', 'Completed', 'Stopped'):
    describe_response = sm.describe_transform_job(TransformJobName = transform_job_name)
    job_run_status = describe_response['TransformJobStatus']
    print (job_run_status)
    sleep(30)
# end::autopilot-track-transform-job[]    

print(describe_response)
# -

# Now let's view the results of the transform job:

# +
# tag::autopilot-transform-job-results[]
s3_output_key = '{}/inference-results/test_data_binary_classifier.csv.out'.format(prefix);
local_inference_results_path = 'data/download/inference_results.csv'

inference_results_bucket = boto_session.resource("s3").Bucket(session.default_bucket())
inference_results_bucket.download_file(s3_output_key, local_inference_results_path);

data = pd.read_csv(local_inference_results_path, sep=';', header=None)
data.sample(10, random_state=42)
# end::autopilot-transform-job-results[]

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
df_test_under = pd.read_csv("data/df_test_under_all.csv")

predictions = data[0]
y_test = df_test_under["label"]

evaluate_model(y_test, predictions)
# end::test-model[]
# -

evaluate_model(y_test, predictions).to_csv("data/sagemaker-model-eval.csv", index=False)

feature_importance(columns, classifier)
