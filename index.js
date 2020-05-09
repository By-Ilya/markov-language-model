const {
    blrCorpusPath,
    ruCorpusPath,
    ukrCorpusPath,
    N,
    trainSize,
    countExperiments
} = require('./config');
const {
    readCorpus,
    shuffleSentencesAndGetNGrams
} = require('./processCorpus')
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
            ` - NGrams: ${N}\n` +
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
        const blrSet = getTrainTestSets(
            shuffleSentencesAndGetNGrams(blrCorpus.normalizedSentences)
        );
        BLR.train = blrSet[0].train;
        BLR.test = blrSet[0].test;
        const ruSet = getTrainTestSets(
            shuffleSentencesAndGetNGrams(ruCorpus.normalizedSentences)
        );
        RU.train = ruSet[0].train;
        RU.test = ruSet[0].test;
        const ukrSet = getTrainTestSets(
            shuffleSentencesAndGetNGrams(ukrCorpus.normalizedSentences)
        );
        UKR.train = ukrSet[0].train;
        UKR.test = ukrSet[0].test;

        console.log('Fitting all models...');
        fitModels();

        let backTrackingModels = {
            blr: [],
            ru: [],
            ukr: []
        }
        for (let i = 1; i < blrSet.length; i++) {
            backTrackingModels.blr.push(
                fitAndGetBackTrackingModel(blrSet[i].train, (N - i))
            );
        }
        for (let i = 1; i < ruSet.length; i++) {
            backTrackingModels.ru.push(
                fitAndGetBackTrackingModel(ruSet[i].train, (N - i))
            );
        }
        for (let i = 1; i < ukrSet.length; i++) {
            backTrackingModels.ukr.push(
                fitAndGetBackTrackingModel(ukrSet[i].train, (N - i))
            );
        }

        console.log('Create test data from sets...');
        const testData = mergeAndShuffleTestData();
        EXPERIMENT_RESULTS.allAnswers = testData.length;

        console.log('Prediction...');
        for (let labeledSentence of testData) {
            const predictedLabel = getBestPrediction(
                labeledSentence.sentence, backTrackingModels
            ).label;
            calculateRateValues(labeledSentence.label, predictedLabel);
            // console.log(`Predicted: ${predictedLabel}, true: ${labeledSentence.label}`);
        }

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

getTrainTestSets = (nGramsSentences) => {
    const trainTestSets = [];
    for (let i = 0; i < nGramsSentences.length; i++) {
        const slicedIndex = Math.ceil(nGramsSentences[i].length * trainSize);
        trainTestSets.push({
            train: nGramsSentences[i].slice(0, slicedIndex),
            test: nGramsSentences[i].slice(slicedIndex)
        });
    }

    return trainTestSets;
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
    });
}

fitAndGetBackTrackingModel = (trainSet, n) => {
    let backTrackingModel = new MarkovChain();
    trainSet.forEach(sentence => {
        backTrackingModel.fit(sentence, n);
    });

    return backTrackingModel;
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

getBestPrediction = (sentence, backTrackingModels) => {
    try {
        const predictionResults = [
            {label: BLR.label, probability: BLR.model.predict(sentence, backTrackingModels.blr)},
            {label: RU.label, probability: RU.model.predict(sentence, backTrackingModels.ru)},
            {label: UKR.label, probability: UKR.model.predict(sentence, backTrackingModels.ukr)},
        ];

        return predictionResults.sort(probabilitySortRule)[0];
    } catch (e) {
        throw e;
    }
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
            BLR.FN++;
            if (predictedLabel === 'ru') { RU.FP++; UKR.TN++ }
            if (predictedLabel === 'ukr') { UKR.FP++; RU.TN++ }
        }
    }
    if (trueLabel === 'ru') {
        if (predictedLabel === trueLabel) {
            EXPERIMENT_RESULTS.positiveAnswers++;
            RU.TP++;
        } else {
            RU.FN++;
            if (predictedLabel === 'blr') { BLR.FP++; UKR.TN++ }
            if (predictedLabel === 'ukr') { UKR.FP++; BLR.TN++ }
        }
    }
    if (trueLabel === 'ukr') {
        if (predictedLabel === trueLabel) {
            EXPERIMENT_RESULTS.positiveAnswers++;
            UKR.TP++;
        } else {
            UKR.FN++;
            if (predictedLabel === 'blr') { BLR.FP++; RU.TN++ }
            if (predictedLabel === 'ru') { RU.FP++; BLR.TN++ }
        }
    }
}


run();