let calculateAccuracy = (positiveAnswers, allAnswers) => {
    return positiveAnswers / allAnswers;
}

let calculatePrecision = (TP, FP) => {
    return TP / (TP + FP);
}

let calculateRecall = (TP, FN) => {
    return TP / (TP + FN);
}

let calculateF1 = (precision, recall) => {
    return 2 * ((precision * recall) / (precision + recall));
}


module.exports = {
    calculateAccuracy,
    calculatePrecision,
    calculateRecall,
    calculateF1
};