const path = require('path')
const hyperlink = require('hyperlink')
const TapRender = require('./tap-render.js');

const root = path.join(__dirname, '..')

;(async () => {
  const tapRender = new TapRender()
  tapRender.pipe(process.stdout)
  try {
    const skipPatterns = [
      // initial redirect
      'load index.html',
      'load ../../../../../index.html',
      // google fonts
      'load https://fonts.googleapis.com/',
      // static resources
      'load static/assets',
      // rate limit on twitter.com (will return 400 code if quota exceeded)
      'external-check https://twitter.com/neo4j'
    ]
    const neo4jRootRelativeUrls = []
    const skipFilter = (report) => {
      return Object.values(report).some((value) => {
          const neo4jRootRelativeUrl = report.at.match(/href="(\/[^"]+)"/)
          const skip = skipPatterns.some((pattern) => String(value).includes(pattern))
          if (!skip) {
            if (neo4jRootRelativeUrl) {
              neo4jRootRelativeUrls.push(neo4jRootRelativeUrl[1])
              return true
            }
          }
          return skip
        }
      )
    };
    await hyperlink({
        root,
        inputUrls: [`build/site/developer/index.html`],
        skipFilter: skipFilter,
        recursive: true,
        internalOnly: true
      },
      tapRender
    )
    // remind: enable once all errors have been fixed!
    /*
    await hyperlink({
      inputUrls: neo4jRootRelativeUrls.map(url => `https://neo4j.com${url}`),
      internalOnly: true
    }, tapRender)
     */
  } catch (err) {
    console.log(err.stack);
    process.exit(1);
  }
  const results = tapRender.close();
  process.exit(results.fail ? 1 : 0);
})()
