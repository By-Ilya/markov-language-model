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
const {
    calculateAccuracy,
    calculatePrecision,
    calculateRecall,
    calculateF1
} = require('./helpers/metrics');

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
        const blrAvgMetrics = { precision: 0, recall: 0, f1: 0 };
        const ruAvgMetrics = { precision: 0, recall: 0, f1: 0 };
        const ukrAvgMetrics = { precision: 0, recall: 0, f1: 0 };
        for (let i = 0; i < countExperiments; i++) {
            runInitStage();
            console.log(`\n----- Experiment ${i + 1}:`);
            await runOneExperiment({blrCorpus, ruCorpus, ukrCorpus});
            sumAccuracy += EXPERIMENT_RESULTS.accuracy;

            blrAvgMetrics.precision += BLR.precision; blrAvgMetrics.recall += BLR.recall; blrAvgMetrics.f1 += BLR.F1;
            ruAvgMetrics.precision += RU.precision; ruAvgMetrics.recall += RU.recall; ruAvgMetrics.f1 += RU.F1;
            ukrAvgMetrics.precision += UKR.precision; ukrAvgMetrics.recall += UKR.recall; ukrAvgMetrics.f1 += UKR.F1;
        }

        blrAvgMetrics.precision /= countExperiments;
        blrAvgMetrics.recall /= countExperiments;
        blrAvgMetrics.f1 /= countExperiments;

        ruAvgMetrics.precision /= countExperiments;
        ruAvgMetrics.recall /= countExperiments;
        ruAvgMetrics.f1 /= countExperiments;

        ukrAvgMetrics.precision /= countExperiments;
        ukrAvgMetrics.recall /= countExperiments;
        ukrAvgMetrics.f1 /= countExperiments;

        console.log(
            '\nRESULTS:\n' +
            ` - Count experiments: ${countExperiments}\n` +
            ` - Train set size: ${trainSize}\n` +
            ` - Avg accuracy = ${sumAccuracy / countExperiments}\n` +
            ' - BLR metrics: ' + JSON.stringify(blrAvgMetrics) + '\n' +
            ' - RU metrics: ' + JSON.stringify(ruAvgMetrics) + '\n' +
            ' - UKR metrics: ' + JSON.stringify(ukrAvgMetrics) + '\n'
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
        label: 'blr',
        TP: 0,
        TN: 0,
        FP: 0,
        FN: 0,
        precision: 0,
        recall: 0,
        F1: 0
    };
    RU = {
        model: new MarkovChain(),
        train: [],
        test: [],
        label: 'ru',
        TP: 0,
        TN: 0,
        FP: 0,
        FN: 0,
        precision: 0,
        recall: 0,
        F1: 0
    };
    UKR = {
        model: new MarkovChain(),
        train: [],
        test: [],
        label: 'ukr',
        TP: 0,
        TN: 0,
        FP: 0,
        FN: 0,
        precision: 0,
        recall: 0,
        F1: 0
    };

    EXPERIMENT_RESULTS = {
        positiveAnswers: 0,
        allAnswers: 0,
        accuracy: 0,
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
            calculateRateValues(labeledSentence.label, predictedLabel);
            // console.log(`Predicted: ${predictedLabel}, true: ${labeledSentence.label}`);
        });

        EXPERIMENT_RESULTS.accuracy = calculateAccuracy(
            EXPERIMENT_RESULTS.positiveAnswers,
            EXPERIMENT_RESULTS.allAnswers
        );

        BLR.precision = calculatePrecision(BLR.TP, BLR.FP);
        BLR.recall = calculateRecall(BLR.TP, BLR.FN);
        BLR.F1 = calculateF1(BLR.precision, BLR.recall);

        RU.precision = calculatePrecision(RU.TP, RU.FP);
        RU.recall = calculateRecall(RU.TP, RU.FN);
        RU.F1 = calculateF1(RU.precision, RU.recall);

        UKR.precision = calculatePrecision(UKR.TP, UKR.FP);
        UKR.recall = calculateRecall(UKR.TP, UKR.FN);
        UKR.F1 = calculateF1(UKR.precision, UKR.recall);

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

calculateRateValues = (trueLabel, predictedLabel) => {
    if (trueLabel === 'blr') {
        if (predictedLabel === trueLabel) {
            EXPERIMENT_RESULTS.positiveAnswers++;
            BLR.TP++;
        } else {
            BLR.TN++;
            if (predictedLabel === 'ru') { RU.FP++; UKR.FN++ }
            if (predictedLabel === 'ukr') { UKR.FP++; RU.FN++ }
        }
    }
    if (trueLabel === 'ru') {
        if (predictedLabel === trueLabel) {
            EXPERIMENT_RESULTS.positiveAnswers++;
            RU.TP++;
        } else {
            RU.TN++;
            if (predictedLabel === 'blr') { BLR.FP++; UKR.FN++ }
            if (predictedLabel === 'ukr') { UKR.FP++; BLR.FN++ }
        }
    }
    if (trueLabel === 'ukr') {
        if (predictedLabel === trueLabel) {
            EXPERIMENT_RESULTS.positiveAnswers++;
            UKR.TP++;
        } else {
            UKR.TN++;
            if (predictedLabel === 'blr') { BLR.FP++; RU.FN++ }
            if (predictedLabel === 'ru') { RU.FP++; BLR.FN++ }
        }
    }
}


run();