# ---
# jupyter:
#   jupytext:
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

# # Setting up your environment
#
# In this notebook we're going setup the environment for the course.
#
# ## Libraries
#
# First let's install some libraries that we'll be using.

# tag::pip-install[]
# !pip install neo4j pandas matplotlib sklearn
# end::pip-install[]

# We'll start by importing py2neo library which we'll use to import the data into Neo4j. py2neo is a client library and toolkit for working with Neo4j from within Python applications. It is well suited for Data Science workflows and has great integration with other Python Data Science tools.

from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://link-prediction-neo4j", auth=("neo4j", "admin"))        
print(driver.address)

# If that works fine, and no exceptions are thrown, we're ready to continue with the rest of the course!
#
# **Keep this notebook open as we'll need to copy the credentials into the other notebooks!**
