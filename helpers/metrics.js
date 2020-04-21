calculateAccuracy = (positiveAnswers, allAnswers) => {
    return positiveAnswers / allAnswers;
}


module.exports = { calculateAccuracy };