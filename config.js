require('dotenv').config();

const blrCorpusPath = process.env.BLR_CORPUS_PATH || './corpus/blr/';
const ruCorpusPath = process.env.RU_CORPUS_PATH || './corpus/ru/';
const ukrCorpusPath = process.env.UKR_CORPUS_PATH || './corpus/ukr/';

const trainSize = parseFloat(process.env.TRAIN_SIZE) || 0.8;


module.exports = {
    blrCorpusPath,
    ruCorpusPath,
    ukrCorpusPath,
    trainSize
};