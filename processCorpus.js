const natural = require('natural');

const {
    getFilesFromDirectory,
    readDataFromFile
} = require('./helpers/filesHelper');

const sentenceTokenizer = new natural.SentenceTokenizer();
const wordTokenizer = new natural.WordTokenizer();
const stemmer = natural.PorterStemmerRu;
const NGrams = natural.NGrams;

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
        const normalizedTokens = splitSentencesToTokens(
            sentences
        );

        const biGrams = normalizedTokens.map(tokenizedSentence => {
            return NGrams.bigrams(tokenizedSentence);
        });
        const triGrams = normalizedTokens.map(tokenizedSentence => {
            return NGrams.trigrams(tokenizedSentence);
        });

        return {documentsList, normalizedTokens, biGrams, triGrams};
    } catch (err) {
        throw err;
    }
};

getFilesListFromCorpus = async (corpusDirectory) => {
    console.log('Getting documents from corpus folder...');
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
    console.log('Getting sentences from documents...');
    let sentences = [];
    try {
        for (let fileName of documentsList) {
            const dataFromFile = await readDataFromFile(
                corpusDirectory + fileName
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

splitSentencesToTokens = sentences => {
    console.log('Splitting sentences to tokens...');
    return sentences.map(sentence => {
        const tokens = wordTokenizer.tokenize(sentence);
        const lemmas = getStemsFromTokens(tokens);

        lemmas.unshift(START_TOKEN);
        lemmas.push(END_TOKEN);

        return lemmas;
    })
};

getStemsFromTokens = tokens => {
    return tokens.map(token => {
        if (token.match(NUMBER_REG_EXP)) {
            return NUMBER_TOKEN;
        }

        return stemmer.stem(token);
    })
};


module.exports = readCorpus;