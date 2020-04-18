const path = require('path');
const {
    readDataFromFile,
    writeDataToFile
} = require('../helpers/filesHelper');

class MarkovChain {
    static #minProbability = 0.0000000001;
    static #countModelFileName = './countMarkovModel.json';
    static #probModelFileName = './probMarkovModel.json';

    #countModel;
    #probModel;

    #addKey;
    #calculateProbabilities;
    #modelToObject;

    static setMinProbability(newMinProbability) {
        if (newMinProbability >= 0 && newMinProbability <= 1) {
            MarkovChain.#minProbability = newMinProbability;
        }
    }

    static getMinProbability() {
        return MarkovChain.#minProbability;
    }

    static setCountModelFileName(newFileName) {
        if (newFileName) MarkovChain.#countModelFileName = newFileName;
    }

    static setProbModelFileName(newFileName) {
        if (newFileName) MarkovChain.#probModelFileName = newFileName;
    }

    static getCountModelFileName() {
        return MarkovChain.#countModelFileName;
    }

    static getProbModelFileName() {
        return MarkovChain.#probModelFileName;
    }


    constructor() {
        this.#countModel = new Map();
        this.#probModel = new Map();

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
    }

    fit(biGrams) {
        biGrams.forEach(biGram => {
            this.#addKey(biGram[0], biGram[1]);
        });

        this.#calculateProbabilities();
    }

    addNewKey(key, value) {
        this.#addKey(key, value);
        this.#calculateProbabilities();
    }

    predict(biGrams) {
        let sequenceProb = 1;
        biGrams.forEach(biGram => {
            if (this.#probModel.has(biGram[0])) {
                const probMap = this.#probModel.get(biGram[0]);
                if (probMap.has(biGram[1])) {
                    sequenceProb *= probMap.get(biGram[1]);
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

    async saveTrainedModel(outDir) {
        const outCountObject = this.#modelToObject(this.#countModel);
        const outProbObject = this.#modelToObject(this.#probModel);

        await writeDataToFile(
            path.resolve(outDir, `./${MarkovChain.getCountModelFileName()}`),
            JSON.stringify(outCountObject)
        );
        await writeDataToFile(
            path.resolve(outDir, `./${MarkovChain.getProbModelFileName()}`),
            JSON.stringify(outProbObject)
        );
    }

    async loadTrainedModel(inDir) {
        const countModelData = await readDataFromFile(
            path.resolve(inDir, `./${MarkovChain.getCountModelFileName()}`)
        );
        const probModelData = await readDataFromFile(
            path.resolve(inDir, `./${MarkovChain.getProbModelFileName()}`)
        );

        this.#countModel = new Map(Object.entries(JSON.parse(countModelData)));
        this.#probModel = new Map(Object.entries(JSON.parse(probModelData)));
    }
}


module.exports = MarkovChain;