# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
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

# # Building a co-author graph
#
# In this notebook we're going to build an inferred graph of co-authors based on people collaborating on the same papers. We're also going to store a property on the relationship indicating the year of their first collaboration.

# tag::imports[]
from neo4j import GraphDatabase
# end::imports[]

# +
# tag::driver[]
bolt_uri = "bolt://link-prediction-neo4j"
driver = GraphDatabase.driver(bolt_uri, auth=("neo4j", "admin"))
# end::driver[]

print(driver.address)
# -

# We can create the co-author graph by running the query below to do this:

# +
# tag::data-import[]
query = """
CALL apoc.periodic.iterate(
  "MATCH (a1)<-[:AUTHOR]-(paper)-[:AUTHOR]->(a2:Author)
   WITH a1, a2, paper
   ORDER BY a1, paper.year
   RETURN a1, a2, collect(paper)[0].year AS year, count(*) AS collaborations",
  "MERGE (a1)-[coauthor:CO_AUTHOR {year: year}]-(a2)
   SET coauthor.collaborations = collaborations",
  {batchSize: 100})
"""

with driver.session(database="neo4j") as session:
    result = session.run(query)
# end::data-import[]
    for row in result:
        print(row)
# -

# Now that we've created our co-author graph, we want to come up with an approach that will allow us to predict future links (relationships) that will be created between people.
