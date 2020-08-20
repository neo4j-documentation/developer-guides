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

# # Citation Dataset Loading
#
# In this notebook we're going to load the citation dataset into Neo4j.
#
# First let's import a couple of Python libraries that will help us with this process.

# We'll start by importing py2neo library which we'll use to import the data into Neo4j. py2neo is a client library and toolkit for working with Neo4j from within Python applications. It is well suited for Data Science workflows and has great integration with other Python Data Science tools.

from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://link-prediction-neo4j", auth=("neo4j", "admin"))        
print(driver.address)

# ## Create Constraints
#
# First let's create some constraints to make sure we don't import duplicate data:

with driver.session(database="neo4j") as session:
    display(session.run("CREATE CONSTRAINT ON (a:Article) ASSERT a.index IS UNIQUE").consume().counters)
    display(session.run("CREATE CONSTRAINT ON (a:Author) ASSERT a.name IS UNIQUE").consume().counters)
    display(session.run("CREATE CONSTRAINT ON (v:Venue) ASSERT v.name IS UNIQUE").consume().counters)

# ## Loading the data
#
# Now let's load the data into the database. We'll create nodes for Articles, Venues, and Authors.
#

# +
query = """
CALL apoc.periodic.iterate(
  'UNWIND ["dblp-ref-0.json", "dblp-ref-1.json", "dblp-ref-2.json", "dblp-ref-3.json"] AS file
   CALL apoc.load.json("https://github.com/mneedham/link-prediction/raw/master/data/" + file)
   YIELD value WITH value
   return value',
  'MERGE (a:Article {index:value.id})
   SET a += apoc.map.clean(value,["id","authors","references", "venue"],[0])
   WITH a, value.authors as authors, value.references AS citations, value.venue AS venue
   MERGE (v:Venue {name: venue})
   MERGE (a)-[:VENUE]->(v)
   FOREACH(author in authors | 
     MERGE (b:Author{name:author})
     MERGE (a)-[:AUTHOR]->(b))
   FOREACH(citation in citations | 
     MERGE (cited:Article {index:citation})
     MERGE (a)-[:CITED]->(cited))', 
   {batchSize: 1000, iterateList: true});
"""

with driver.session(database="neo4j") as session:
    result = session.run(query)
    for row in result:
        print(row)

# +
query = """
MATCH (a:Article) 
WHERE not(exists(a.title))
DETACH DELETE a
"""

with driver.session(database="neo4j") as session:
    result = session.run(query)
    print(result.consume().counters)
# -

# In the next notebook we'll explore the data that we've imported. 
