const {
    blrCorpusPath,
    ruCorpusPath,
    ukrCorpusPath,
    trainSize
} = require('./config');
const readCorpus = require('./processCorpus');
const shuffle = require('./helpers/shuffle');
const MarkovChain = require('./MarkovModel/MarkovChain');
const { calculateAccuracy } = require('./helpers/metrics');

let BLR = {
    model: new MarkovChain(),
    train: [],
    test: [],
    label: 'blr'
};
let RU = {
    model: new MarkovChain(),
    train: [],
    test: [],
    label: 'ru'
};
let UKR = {
    model: new MarkovChain(),
    train: [],
    test: [],
    label: 'ukr'
};

let EXPERIMENT_RESULTS = {
    positiveAnswers: 0,
    allAnswers: 0,
    accuracy: 0
}

run = async () => {
    try {
        console.log('Reading corpuses...');
        const blrCorpus = await getCorpusData(blrCorpusPath);
        const ruCorpus = await getCorpusData(ruCorpusPath);
        const ukrCorpus = await getCorpusData(ukrCorpusPath);

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
            console.log(`Predicted: ${predictedLabel}, true: ${labeledSentence.label}`);
            if (predictedLabel === labeledSentence.label)
                EXPERIMENT_RESULTS.positiveAnswers++;
        });

        EXPERIMENT_RESULTS.accuracy = calculateAccuracy(
            EXPERIMENT_RESULTS.positiveAnswers,
            EXPERIMENT_RESULTS.allAnswers
        );

        console.log('\nExperiment results:');
        console.log(EXPERIMENT_RESULTS);

        process.exit(0);
    } catch (error) {
        console.error(error);
        process.exit(0);
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