const http = require('https');
const fs = require('fs');

const download = function(url, dest, cb) {
  var file = fs.createWriteStream(dest);
  var request = http.get(url, function(response) {
    response.pipe(file);
    file.on('finish', function() {
      file.close(cb);  // close() is async, call cb after close completes.
    });
  }).on('error', function(err) { // Handle errors
    fs.unlink(dest); // Delete the file async. (But we don't check the result)
    if (cb) cb(err.message);
  });
};

const remotePrefix = "https://raw.githubusercontent.com/neo4j-examples/link-prediction/master/notebooks/"
const localPrefix = "modules/graph-data-science/examples/link-prediction/"

const files = ["03_Train_Test_Split.py", "04_Model_Feature_Engineering.py", "05_Train_Evaluate_Model.py",
               "data/model-eval.csv", "data/df_test_under_sample.csv", "data/df_train_under_sample.csv",
               "data/df_test_under_basic_sample.csv", "data/df_train_under_basic_sample.csv"]
files.forEach(value => download(remotePrefix + value, localPrefix + value))
