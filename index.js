const {
    corpusFolder,
    outputFolder
} = require('./config');
const readCorpus = require('./processCorpus');
const MarkovChain = require('./MarkovModel/MarkovChain');

let CORPUS_DATA = {
    documentsList: [],
    normalizedTokens: [],
    biGrams: [],
    triGrams: []
}

run = async () => {
    try {
        CORPUS_DATA = await readCorpus(corpusFolder);
        printStats();

        let testMarkovChain = new MarkovChain();
        CORPUS_DATA.biGrams.forEach(sentenceBiGrams => {
            testMarkovChain.fit(sentenceBiGrams);
        })
        await testMarkovChain.saveTrainedModel(outputFolder);
    } catch (e) {
        console.error(e);
    }
}

printStats = () => {
    console.log(
        ` - Documents: ${CORPUS_DATA.documentsList.length}\n` +
        ` - Sentences: ${CORPUS_DATA.normalizedTokens.length}\n`
    );
}


run();