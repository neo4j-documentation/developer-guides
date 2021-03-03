CALL gds.alpha.degree.write('linkpred',{
    writeProperty: 'degree'
});

CALL gds.graph.create(
  'linkpred', {
      Author: {
          properties: ["degree"]
      }
  }, {
    CO_AUTHOR: {
      orientation: 'UNDIRECTED'
    }
  }
);

//3. Add test train splits to in-memory graph
CALL gds.alpha.ml.splitRelationships.mutate('linkpred', {
  relationshipTypes: ['CO_AUTHOR'],
  remainingRelationshipType: 'CO_AUTHOR_REMAINING',
  holdoutRelationshipType: 'CO_AUTHOR_TESTGRAPH',
  holdoutFraction: 0.2
}) YIELD relationshipsWritten;

CALL gds.alpha.ml.splitRelationships.mutate('linkpred', {
  relationshipTypes: ['CO_AUTHOR_REMAINING'],
  remainingRelationshipType: 'CO_AUTHOR_IGNORED_FOR_TRAINING',
  holdoutRelationshipType: 'CO_AUTHOR_TRAINGRAPH',
  holdoutFraction: 0.2
}) YIELD relationshipsWritten;

// Feature Engineering

CALL gds.pageRank.mutate('linkpred',{
	  maxIterations: 20,
    dampingFactor: 0.05,
    relationshipTypes: ["CO_AUTHOR"],
    mutateProperty: 'pagerank'
});

CALL gds.triangleCount.mutate('linkpred',{
    relationshipTypes: ["CO_AUTHOR"],
    mutateProperty: 'triangles'
});

CALL gds.beta.fastRPExtended.mutate('linkpred',{
    propertyDimension: 45,
    embeddingDimension: 250,
    featureProperties: ["degree", "pagerank", "triangles"],
    relationshipTypes: ["CO_AUTHOR_IGNORED_FOR_TRAINING"],
    iterationWeights: [0, 0, 1.0, 1.0],
    normalizationStrength:0.05,
    mutateProperty: 'fastRP_Extended_Embedding'
});

CALL gds.fastRP.mutate('linkpred',{
    embeddingDimension: 250,
    relationshipTypes: ["CO_AUTHOR_IGNORED_FOR_TRAINING"],
    iterationWeights: [0, 0, 1.0, 1.0],
    normalizationStrength:0.05,
    mutateProperty: 'fastRP_Embedding'
});




//4.  train a link prediction model
CALL gds.alpha.ml.linkPrediction.train('linkpred', {
  trainRelationshipType: 'CO_AUTHOR_TRAINGRAPH',
  testRelationshipType: 'CO_AUTHOR_TESTGRAPH',
  modelName: 'model',
  featureProperties: ['fastRP_Extended_Embedding', 'pagerank'],
  validationFolds: 5,
  classRatio: 1.33,
  randomSeed: 2,
  params: [
    {penalty: 0.24, maxIterations: 1000},
    {penalty: 0.5, maxIterations: 1000},
    {penalty: 1.0, maxIterations: 1000},
    {penalty: 0.0, maxIterations: 1000}
  ]
}) YIELD modelInfo
RETURN
  modelInfo.bestParameters AS winningModel,
  modelInfo.metrics.AUCPR.outerTrain AS trainGraphScore,
  modelInfo.metrics.AUCPR.test AS testGraphScore


CALL gds.beta.model.drop("model");

// Using normal FastRP
CALL gds.alpha.ml.linkPrediction.train('linkpred', {
  trainRelationshipType: 'CO_AUTHOR_TRAINGRAPH',
  testRelationshipType: 'CO_AUTHOR_TESTGRAPH',
  modelName: 'model',
  featureProperties: ['fastRP_Embedding', 'pagerank', 'degree'],
  validationFolds: 5,
  classRatio: 1.33,
  randomSeed: 2,
  params: [
    {penalty: 0.24, maxIterations: 1000},
    {penalty: 0.5, maxIterations: 1000},
    {penalty: 1.0, maxIterations: 1000},
    {penalty: 0.0, maxIterations: 1000}
  ]
}) YIELD modelInfo
RETURN
  modelInfo.bestParameters AS winningModel,
  modelInfo.metrics.AUCPR.outerTrain AS trainGraphScore,
  modelInfo.metrics.AUCPR.test AS testGraphScore