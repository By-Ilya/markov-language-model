const path = require('path');
const natural = require('natural');
const MyStem = require('mystem3');

const { N } = require('./config');
const {
    getFilesFromDirectory,
    readDataFromFile
} = require('./helpers/filesHelper');

const sentenceTokenizer = new natural.SentenceTokenizer();
const wordTokenizer = new natural.WordTokenizer();
const myStem = new MyStem();
const NGrams = natural.NGrams;

myStem.start();

const NUMBER_REG_EXP = /\d+/g;

const REDUNDANT_FILES_NAMES = ['.DS_Store'];
const START_TOKEN = '__START__';
const NUMBER_TOKEN = '__NUMBER__';
const END_TOKEN = '__END__';

let readCorpus = async (corpusDirectory) => {
    try {
        const documentsList = removeRedundantDocuments(
            await getFilesListFromCorpus(corpusDirectory)
        );
        const sentences = await getSentencesFromDocuments(
            corpusDirectory, documentsList
        );
        const normalizedSentences = await splitSentencesToSymbols(
            sentences
        );

        const nGrams = normalizedSentences.map(tokenizedSentence => {
            return NGrams.ngrams(tokenizedSentence, N);
        });

        return {documentsList, normalizedSentences, nGrams};
    } catch (err) {
        throw err;
    }
};

getFilesListFromCorpus = async (corpusDirectory) => {
    try {
        return await getFilesFromDirectory(
            corpusDirectory
        );
    } catch (err) {
        throw err;
    }
};

removeRedundantDocuments = documentsList => {
    return documentsList.filter(fileName =>
        REDUNDANT_FILES_NAMES.indexOf(fileName) === -1
    );
};

getSentencesFromDocuments = async (corpusDirectory, documentsList) => {
    let sentences = [];
    try {
        for (let fileName of documentsList) {
            const dataFromFile = await readDataFromFile(
                path.resolve(corpusDirectory, fileName)
            );
            sentences = sentences.concat(
                sentenceTokenizer.tokenize(dataFromFile)
            );
        }

        return sentences;
    } catch (err) {
        throw err;
    }
}

splitSentencesToSymbols = async sentences => {
    let normalizedTokens = [];
    for (let sentence of sentences) {
        const tokens = wordTokenizer.tokenize(sentence);
        const lemmas = await getLemmasFromTokens(tokens);

        let symbols = [];
        lemmas.forEach(lemma => {
            if (lemma === NUMBER_TOKEN) {
                symbols = symbols.concat(lemma);
            } else {
                const symbolsFromLemma = lemma.split('');
                symbols = symbols.concat(symbolsFromLemma);
            }
        })

        symbols.unshift(START_TOKEN);
        symbols.push(END_TOKEN);

        normalizedTokens.push(symbols);
    }

    return normalizedTokens;
};

getLemmasFromTokens = async tokens => {
    let lemmas = [];
    for (let token of tokens) {
        if (token.match(NUMBER_REG_EXP)) {
            lemmas.push(NUMBER_TOKEN);
            continue;
        }
        const lemma = await getLemmaInPromise(token);
        lemmas.push(lemma.toLowerCase());
    }

    return lemmas;
};

getLemmaInPromise = async token => {
    return myStem.lemmatize(token);
}


module.exports = readCorpus;