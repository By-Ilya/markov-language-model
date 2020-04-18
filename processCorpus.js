const natural = require('natural');
const Morphy = require('phpmorphy');

const {
    getFilesFromDirectory,
    readDataFromFile
} = require('./helpers/filesHelper');

const tokenizer = new natural.WordTokenizer();
const morphy = new Morphy('ru', {
    storage: Morphy.STORAGE_MEM,
    predict_by_suffix: true,
    predict_by_db: true,
    graminfo_as_text: true,
    use_ancodes_cache: false,
    resolve_ancodes: Morphy.RESOLVE_ANCODES_AS_TEXT,
});

const REDUNDANT_FILES_NAMES = ['.DS_Store'];
const START_TOKEN = '__START__';
const END_TOKEN = '__END__';

readCorpus = async (corpusDirectory) => {
    try {
        const documentsList = removeRedundantDocuments(
            await getFilesListFromCorpus(corpusDirectory)
        );
        const tokens = await getTokensFromDocuments(
            corpusDirectory, documentsList
        );
        const lemmas = getLemmasFromTokens(tokens);
        lemmas.unshift(START_TOKEN);
        lemmas.push(END_TOKEN);

        const biGrams = NGrams.bigrams(lemmas);
        const triGrams = NGrams.trigrams(lemmas);

        return {documentsList, lemmas, biGrams, triGrams};
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

getTokensFromDocuments = async (corpusDirectory, documentsList) => {
    console.log('Getting tokens from documents...');
    let tokens = [];
    try {
        for (let fileName of documentsList) {
            const dataFromFile = await readDataFromFile(
                corpusDirectory + fileName
            );
            tokens = tokens.concat(
                tokenizer.tokenize(dataFromFile)
            );
        }

        return tokens.map(t => {
            return t.toLowerCase();
        });
    } catch (err) {
        throw err;
    }
};

getLemmasFromTokens = tokens => {
    console.log('Getting lemmas from tokens...');
    return tokens
        .map(token => morphy.lemmatize(
            token,
            Morphy.NORMAL
        )[0].toLowerCase())
};


module.exports = readCorpus;