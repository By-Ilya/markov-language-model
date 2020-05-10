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
    #backTracking;
    #findTransitionProb;

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

        this.#nGramToKeyValue = (nGram, n) => {
            const slicedIndex = n === undefined ? N - 1 : n - 1;
            const keyArray = nGram.slice(0, slicedIndex);
            const key = keyArray.join('');

            return {key, value: nGram[slicedIndex]};
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

        this.#backTracking = (key, value, backTrackingModels, index) => {
            const firstLowerKey = key.slice(0, key.length - 1);
            const firstLowerValue = key[key.length - 1];
            const secondLowerKey = key.slice(1);
            const secondLowerValue = value;

            let backTrackingProb = 1;

            let foundProb = this.#findTransitionProb(
                backTrackingModels[index], firstLowerKey, firstLowerValue
            );
            if (foundProb === null) {
                backTrackingProb *= (index === backTrackingModels.length - 1)
                    ? MarkovChain.getMinProbability()
                    : this.#backTracking(firstLowerKey, firstLowerValue, backTrackingModels, index + 1);
            } else backTrackingProb *= foundProb;

            foundProb = this.#findTransitionProb(
                backTrackingModels[index], secondLowerKey, secondLowerValue
            );
            if (foundProb === null) {
                backTrackingProb *= (index === backTrackingModels.length - 1)
                    ? MarkovChain.getMinProbability()
                    : this.#backTracking(secondLowerKey, secondLowerValue, backTrackingModels, index + 1);
            } else backTrackingProb *= foundProb;

            return backTrackingProb;
        }

        this.#findTransitionProb = (markovModel, key, value) => {
            const probModel = markovModel.getProbModel();
            if (probModel.has(key)) {
                const probMap = probModel.get(key);
                if (probMap.has(value)) return probMap.get(value);
                else return null;
            } else return null;
        }
    }

    fit(nGrams, n = undefined) {
        nGrams.forEach(nGram => {
            const {key, value} = this.#nGramToKeyValue(nGram, n);
            this.#addKey(key, value);
        });

        this.#calculateProbabilities();
    }

    addNewKey(key, value) {
        this.#addKey(key, value);
        this.#calculateProbabilities();
    }

    predict(nGrams, backTrackingModels) {
        let sequenceProb = 1;

        for (let nGram of nGrams) {
            const {key, value} = this.#nGramToKeyValue(nGram);
            if (this.#probModel.has(key)) {
                const probMap = this.#probModel.get(key);
                if (probMap.has(value)) sequenceProb *= probMap.get(value);
                else {
                    if (!backTrackingModels && !backTrackingModels.length) {
                        sequenceProb *= MarkovChain.getMinProbability();
                    } else {
                        sequenceProb *= this.#backTracking(
                            key, value, backTrackingModels, 0
                        );
                    }
                }
            } else {
                if (!backTrackingModels && !backTrackingModels.length) {
                    sequenceProb *= MarkovChain.getMinProbability();
                } else {
                    sequenceProb *= this.#backTracking(
                            key, value, backTrackingModels, 0
                    );
                }
            }
        }

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