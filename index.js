const {
    blrCorpusPath,
    ruCorpusPath,
    ukrCorpusPath,
    trainSize,
    countExperiments
} = require('./config');
const readCorpus = require('./processCorpus');
const shuffle = require('./helpers/shuffle');
const MarkovChain = require('./MarkovModel/MarkovChain');
const { calculateAccuracy } = require('./helpers/metrics');

let BLR = {};
let RU = {};
let UKR = {};

let EXPERIMENT_RESULTS = {};

run = async () => {
    try {
        console.log('Reading corpuses...');
        const blrCorpus = await getCorpusData(blrCorpusPath);
        const ruCorpus = await getCorpusData(ruCorpusPath);
        const ukrCorpus = await getCorpusData(ukrCorpusPath);
        console.log('All corps was read successfully. Running experiments...');

        let sumAccuracy = 0;
        for (let i = 0; i < countExperiments; i++) {
            runInitStage();
            console.log(`\n----- Experiment ${i + 1}:`);
            await runOneExperiment({blrCorpus, ruCorpus, ukrCorpus});
            sumAccuracy += EXPERIMENT_RESULTS.accuracy;
        }

        console.log(
            '\n RESULTS:\n' +
            ` - Count experiments: ${countExperiments}\n` +
            ` - Train set size: ${trainSize}\n` +
            ` - Avg accuracy = ${sumAccuracy / countExperiments}`
        );
        process.exit(0);
    } catch (error) {
        console.error(error);
        process.exit(0);
    }
}

runInitStage = () => {
    BLR = {
        model: new MarkovChain(),
        train: [],
        test: [],
        label: 'blr'
    };
    RU = {
        model: new MarkovChain(),
        train: [],
        test: [],
        label: 'ru'
    };
    UKR = {
        model: new MarkovChain(),
        train: [],
        test: [],
        label: 'ukr'
    };

    EXPERIMENT_RESULTS = {
        positiveAnswers: 0,
        allAnswers: 0,
        accuracy: 0
    };
};

runOneExperiment = async ({blrCorpus, ruCorpus, ukrCorpus}) => {
    try {
        console.log('Make train and test sets...');
        const blrSet = getTrainTestSets(blrCorpus.biGrams);
        BLR.train = blrSet.train;
        BLR.test = blrSet.test;
        const ruSet = getTrainTestSets(ruCorpus.biGrams);
        RU.train = ruSet.train;
        RU.test = ruSet.test;
        const ukrSet = getTrainTestSets(ukrCorpus.biGrams);
        UKR.train = ukrSet.train;
        UKR.test = ukrSet.test;

        console.log('Fitting all models...');
        fitModels();

        console.log('Create test data from sets...');
        const testData = mergeAndShuffleTestData();
        EXPERIMENT_RESULTS.allAnswers = testData.length;

        console.log('Prediction...');
        testData.forEach(labeledSentence => {
            const predictedLabel = getBestPrediction(labeledSentence.sentence).label;
            // console.log(`Predicted: ${predictedLabel}, true: ${labeledSentence.label}`);
            if (predictedLabel === labeledSentence.label)
                EXPERIMENT_RESULTS.positiveAnswers++;
        });

        EXPERIMENT_RESULTS.accuracy = calculateAccuracy(
            EXPERIMENT_RESULTS.positiveAnswers,
            EXPERIMENT_RESULTS.allAnswers
        );

        console.log('\nExperiment results:', EXPERIMENT_RESULTS);
    } catch (error) {
        throw error;
    }
}

getCorpusData = async (corpusPath) => {
    const corpusData = await readCorpus(corpusPath);
    console.log(
        'Corpus stats: \n' +
        ` - Path: ${corpusPath}\n` +
        ` - Sentences: ${corpusData.normalizedSentences.length}`
    );

    return corpusData;
}

getTrainTestSets = (biGramsSentences) => {
    const shuffledSentences = shuffle(biGramsSentences);
    const slicedIndex = Math.ceil(shuffledSentences.length * trainSize);
    return {
        train: shuffledSentences.slice(0, slicedIndex),
        test: shuffledSentences.slice(slicedIndex)
    }
}

fitModels = () => {
    BLR.train.forEach(sentence => {
        BLR.model.fit(sentence);
    });
    RU.train.forEach(sentence => {
        RU.model.fit(sentence);
    });
    UKR.train.forEach(sentence => {
        UKR.model.fit(sentence);
    })
}

mergeAndShuffleTestData = () => {
    const mergedSet = [];
    BLR.test.forEach(sentence => {
        mergedSet.push({sentence, label: BLR.label});
    });
    RU.test.forEach(sentence => {
        mergedSet.push({sentence, label: RU.label});
    });
    UKR.test.forEach(sentence => {
        mergedSet.push({sentence, label: UKR.label});
    });

    return shuffle(mergedSet);
}

getBestPrediction = (sentence) => {
    const predictionResults = [
        {label: BLR.label, probability: BLR.model.predict(sentence)},
        {label: RU.label, probability: RU.model.predict(sentence)},
        {label: UKR.label, probability: UKR.model.predict(sentence)},
    ];

    return predictionResults.sort(probabilitySortRule)[0];
}

probabilitySortRule = (a, b) => {
    return b.probability - a.probability;
}


run();