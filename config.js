require('dotenv').config();

const blrCorpusPath = process.env.BLR_CORPUS_PATH || './corpus/blr/';
const ruCorpusPath = process.env.RU_CORPUS_PATH || './corpus/ru/';
const ukrCorpusPath = process.env.UKR_CORPUS_PATH || './corpus/ukr/';

const N = parseInt(process.env.N) || 3;
const trainSize = parseFloat(process.env.TRAIN_SIZE) || 0.8;
const countExperiments = parseInt(process.env.COUNT_EXPERIMENTS) || 10;


module.exports = {
    blrCorpusPath,
    ruCorpusPath,
    ukrCorpusPath,
    N,
    trainSize,
    countExperiments
};