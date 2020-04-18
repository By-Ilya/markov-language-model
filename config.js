require('dotenv').config();

const corpusFolder = process.env.CORPUS_FOLDER || './corpus/';
const outputFolder = process.env.OUTPUT_FOLDER || './output-data/';


module.exports = {
    corpusFolder,
    outputFolder
};