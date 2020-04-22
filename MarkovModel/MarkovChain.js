const path = require('path');

const { N } = require('../config');
const {
    readDataFromFile,
    writeDataToFile
} = require('../helpers/filesHelper');

class MarkovChain {
    static #minProbability = 0.0000000001;
    static #pathToSaveCountModel = './markovModel.count.json';
    static #pathToSaveProbModel = './markovModel.prob.json';

    #countModel;
    #probModel;

    #nGramToKeyValue;
    #addKey;
    #calculateProbabilities;
    #modelToObject;
    #objectToModel;
    #entireObjectsToMaps;

    static setMinProbability(newMinProbability) {
        if (newMinProbability >= 0 && newMinProbability <= 1) {
            MarkovChain.#minProbability = newMinProbability;
        }
    }

    static getMinProbability() {
        return MarkovChain.#minProbability;
    }

    static setPathToSaveCountModel(outFolder, newFileName) {
        if (newFileName)
            MarkovChain.#pathToSaveCountModel = path.resolve(outFolder, newFileName);
    }

    static setPathToSaveProbModel(outFolder, newFileName) {
        if (newFileName)
            MarkovChain.#pathToSaveProbModel = path.resolve(outFolder, newFileName);
    }

    static getPathToSaveCountModel() {
        return MarkovChain.#pathToSaveCountModel;
    }

    static getPathToSaveProbModel() {
        return MarkovChain.#pathToSaveProbModel;
    }


    constructor() {
        this.#countModel = new Map();
        this.#probModel = new Map();

        this.#nGramToKeyValue = (nGram) => {
            const keyArray = nGram.slice(0, N - 1);
            const key = keyArray.join('');

            return {key, value: nGram[N - 1]};
        }

        this.#addKey = (key, value) => {
            if (!this.#countModel.has(key)) {
                this.#countModel.set(key, new Map());
            }

            const countMap = this.#countModel.get(key);
            if (!countMap.has(value)) {
                countMap.set(value, 1);
            } else {
                let count = countMap.get(value);
                countMap.set(value, ++count);
            }
        };

        this.#calculateProbabilities = () => {
            for (let [key, countMap] of this.#countModel) {
                let sumCount = 0;
                for (let count of countMap.values()) {
                    sumCount += count;
                }

                const probMap = new Map();
                for (let [value, count] of countMap) {
                    probMap.set(value, count / sumCount);
                }

                this.#probModel.set(key, probMap);
            }
        };

        this.#modelToObject = (model) => {
            const out = {};
            model.forEach((value, key) => {
                if (value instanceof Map) {
                    out[key] = this.#modelToObject(value);
                } else {
                    out[key] = value;
                }
            });

            return out;
        }

        this.#objectToModel = (inObject) => {
            const mapWithObjects = new Map(Object.entries(JSON.parse(inObject)));
            return this.#entireObjectsToMaps(mapWithObjects);
        }

        this.#entireObjectsToMaps = (inModel) => {
            let inMap = new Map();
            for (let [key, obj] of inModel) {
                const entireMap = new Map(Object.entries(obj));
                inMap.set(key, entireMap);
            }

            return inMap;
        }
    }

    fit(nGrams) {
        nGrams.forEach(nGram => {
            const {key, value} = this.#nGramToKeyValue(nGram);
            this.#addKey(key, value);
        });

        this.#calculateProbabilities();
    }

    addNewKey(key, value) {
        this.#addKey(key, value);
        this.#calculateProbabilities();
    }

    predict(nGrams) {
        let sequenceProb = 1;
        nGrams.forEach(nGram => {
            const {key, value} = this.#nGramToKeyValue(nGram);
            if (this.#probModel.has(key)) {
                const probMap = this.#probModel.get(key);
                if (probMap.has(value)) {
                    sequenceProb *= probMap.get(value);
                } else {
                    sequenceProb *= MarkovChain.getMinProbability();
                }
            } else {
                sequenceProb *= MarkovChain.getMinProbability();
            }
        });

        return sequenceProb;
    }

    getCountModel() {
        return this.#countModel;
    }

    getProbModel() {
        return this.#probModel;
    }

    async saveTrainedModel() {
        try {
            const outCountObject = this.#modelToObject(this.#countModel);
            const outProbObject = this.#modelToObject(this.#probModel);

            await writeDataToFile(
                MarkovChain.#pathToSaveCountModel,
                JSON.stringify(outCountObject)
            );
            await writeDataToFile(
                MarkovChain.#pathToSaveProbModel,
                JSON.stringify(outProbObject)
            );
        } catch (e) {
            throw e;
        }
    }

    async loadTrainedModel() {
        try {
            const countModelData = await readDataFromFile(
                MarkovChain.#pathToSaveCountModel
            );
            const probModelData = await readDataFromFile(
                MarkovChain.#pathToSaveProbModel
            );

            this.#countModel = this.#objectToModel(countModelData);
            this.#probModel = this.#objectToModel(probModelData);
        } catch (e) {
            throw e;
        }
    }
}


module.exports = MarkovChain;