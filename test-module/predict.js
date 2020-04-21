const { outputFolder } = require('../config');
const { isDirectory } = require('../helpers/filesHelper');
const readCorpus = require('../processCorpus');
const MarkovChain = require('../MarkovModel/MarkovChain');

const args = process.argv.slice(2);
let CORPUS_DATA = {
    documentsList: [],
    normalizedTokens: [],
    biGrams: [],
    triGrams: []
}

runPrediction = async () => {
    try {
        if (args.length < 2) {
            console.error(`Error: specify command with all required arguments.`);
            process.exit(0);
        }
        const testDataFolder = args[0];
        if (!await isDirectory(testDataFolder)) {
            console.error(`Error: specified test data folder is not directory.`);
            process.exit(0);
        }

        const modelName = args[1];
        MarkovChain.setPathToSaveCountModel(
            outputFolder, `${modelName}.count.json`
        );
        MarkovChain.setPathToSaveProbModel(
            outputFolder, `${modelName}.prob.json`
        );

        console.log(`Loading saved models...`);
        printModelPaths();
        let markovModel = new MarkovChain();
        await markovModel.loadTrainedModel(outputFolder);

        console.log(`Models was loaded. Load data from test corpus...`);
        CORPUS_DATA = await readCorpus(testDataFolder);
        printStats();

        console.log(`Predict data...`);
        let sumProbabilities = 0;
        let countPredictedSentences = 0;
        CORPUS_DATA.biGrams.forEach(sentenceBiGrams => {
            let predictionProb = markovModel.predict(sentenceBiGrams);
            sumProbabilities += predictionProb;
            countPredictedSentences += 1;
        });

        console.log(`Prediction result = ${sumProbabilities / countPredictedSentences}`);
        process.exit(0);
    } catch (err) {
        console.error(err);
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
        ` - Path to load count model: ${MarkovChain.getPathToSaveCountModel()}\n` +
        ` - Path to load prob model: ${MarkovChain.getPathToSaveProbModel()}\n`
    );
}


runPrediction();