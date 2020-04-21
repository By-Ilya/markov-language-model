# markov-language-model
Language prediction model based on Markov chain.

## Description
It is a based implementation of Markov chain that is used to building language models.
The script `index.js` describes the experiment of modeling Markov chains for __belorusian__, __russian__ and __ukrain__ languages
and further prediction of the most probable language for the input text based on the trained models.
The script create train and test sets from read corps, build models and predict languge for test set and calculate average accuracy of built models.

## Requirements
1. `Node JS` library and `NPM` package manager.
2. Libraries installed from `package.json` file.

## Install and configure
1. Go to the project root directory.
2. Run `npm i` or `npm install` command. This command installs necessary libraries.
3. Open `.env` file and configure the following parameters:
- `BLR_CORPUS_PATH`: `string` value, that specifies directory to the corpus with __belorusian__ texts (absolute or relative path).
- `RU_CORPUS_PATH`: `string` value, that specifies directory to the corpus with __russian__ texts (absolute or relative path).
- `UKR_CORPUS_PATH`: `string` value, that specifies directory to the corpus with __ukrain__ texts (absolute or relative path).
- `TRAIN_SIZE`: `float` value in range `[0, 1]`, that specifies the size of the train set.
- `COUNT_EXPERIMENTS`: `integer` value, that specifies count of experiments for calculating average models accuracy.

After that, place into `BLR_CORPUS_PATH`, `RU_CORPUS_PATH`, `UKR_CORPUS_PATH` folders `.txt`-files with corresponding texts.

## Running command
In the project root directory run `npm start` command.
See the result in the system console.

## Used `Node JS` libraries
- `natural` (version `0.6.3`) is used for _tokenizing_ input texts from corpus to _sentences_ and _words_ and creating _bigrams_ from sentences.
- `mystem3` (version `1.2.1`) is used for _creating lemmas_ from words.
