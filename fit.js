const { outputFolder } = require('./config');
const { isDirectory } = require('./helpers/filesHelper');
const readCorpus = require('./processCorpus');
const MarkovChain = require('./MarkovModel/MarkovChain');

const args = process.argv.slice(2);
let CORPUS_DATA = {
    documentsList: [],
    normalizedTokens: [],
    biGrams: [],
    triGrams: []
}

run = async () => {
    try {
        if (args.length < 2) {
            console.error(`Error: specify command with all required arguments.`);
            process.exit(0);
        }
        const corpusFolder = args[0];
        if (!await isDirectory(corpusFolder)) {
            console.error(`Error: specified corpus folder is not directory.`);
            process.exit(0);
        }

        const modelName = args[1];
        MarkovChain.setPathToSaveCountModel(
            outputFolder, `${modelName}.count.json`
        );
        MarkovChain.setPathToSaveProbModel(
            outputFolder, `${modelName}.prob.json`
        );

        CORPUS_DATA = await readCorpus(corpusFolder);
        printStats();

        let testMarkovChain = new MarkovChain();

        console.log('Fitting Markov model...');
        CORPUS_DATA.biGrams.forEach(sentenceBiGrams => {
            testMarkovChain.fit(sentenceBiGrams);
        });

        console.log('Saving fitted models...');
        printModelPaths();
        await testMarkovChain.saveTrainedModel(outputFolder);
        console.log('Models saved successfully and are ready to use for future predictions!');
        process.exit(0);
    } catch (e) {
        console.error(e);
        process.exit(0);
    }
}

printStats = () => {
    console.log(
        ` - Documents: ${CORPUS_DATA.documentsList.length}\n` +
        ` - Sentences: ${CORPUS_DATA.normalizedTokens.length}\n`
    );
}

printModelPaths = () => {
    console.log(
        ` - Path to save count model: ${MarkovChain.getPathToSaveCountModel()}\n` +
        ` - Path to save prob model: ${MarkovChain.getPathToSaveProbModel()}\n`
    );
}


run();