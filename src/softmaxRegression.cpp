#include "softmaxRegression.hpp"

#include <algorithm>
#include <cmath>
#include <execution>
#include <stdexcept>

template <typename T>
SoftmaxRegression<T>::SoftmaxRegression(
    double learningRate,
    unsigned int epochs,
    std::size_t numClasses)
    : learningRate_(learningRate), epochs_(epochs), numClasses_(numClasses) {
    if (learningRate <= 0.0) {
        throw std::invalid_argument("Learning rate must be greater than 0");
    }

    if (epochs == 0) {
        throw std::invalid_argument("Epochs must be greater than 0");
    }

    if (numClasses == 0) {
        throw std::invalid_argument("Dataset must have at least one class");
    }
}

template <typename T>
double SoftmaxRegression<T>::getLearningRate() const {
    return this->learningRate_;
}

template <typename T>
unsigned int SoftmaxRegression<T>::getEpochs() const {
    return this->epochs_;
}

template <typename T>
std::size_t SoftmaxRegression<T>::getNumClasses() const {
    return this->numClasses_;
}

template <typename T>
const Matrix& SoftmaxRegression<T>::getWeights() const {
    return this->weights_;
}

template <typename T>
const std::vector<double>& SoftmaxRegression<T>::getBiases() const {
    return this->biases_;
}

template <typename T>
void SoftmaxRegression<T>::setLearningRate(double learningRate) {
    if (learningRate <= 0.0) {
        throw std::invalid_argument("Learning rate must be greater than 0");
    }

    this->learningRate_ = learningRate;
}

template <typename T>
void SoftmaxRegression<T>::setEpochs(unsigned int epochs) {
    if (epochs == 0) {
        throw std::invalid_argument("Epochs must be greater than 0");
    }

    this->epochs_ = epochs;
}

template <typename T>
Matrix SoftmaxRegression<T>::predictLogits(const Matrix& features) const {
    Matrix logits = features * this->getWeights().transpose();

    std::size_t numSamples{logits.rows()};
    std::size_t numClasses{logits.cols()};
    
    std::vector<double> logitElements(logits.elements());

    for (std::size_t i{0}; i < logitElements.size(); ++i) {
        std::size_t classIndex{i % numClasses};
        logitElements[i] += this->getBiases()[classIndex];
    }
    
    logits.setElements(logitElements);
    return logits;
}

template <typename T>
Matrix SoftmaxRegression<T>::predictProbability(const Matrix& features) const {
    Matrix logits = predictLogits(features);

    std::size_t numSamples{logits.rows()};
    std::size_t numClasses{logits.cols()};

    Matrix probs(numSamples, numClasses);

    std::vector<double> probDists;

    for (std::size_t i{0}; i < numSamples; ++i) {
        std::size_t offset{i * numClasses};

        std::vector<double> sampleLogits(
            logits.elements().begin() + offset,
            logits.elements().begin() + offset + numClasses);

        const std::vector<double> sampleProbDistr = this->softmax(sampleLogits);

        probDists.insert(probDists.end(), sampleProbDistr.begin(), sampleProbDistr.end());
    }

    probs.setElements(probDists);
    return probs;
}

template <typename T>
std::vector<std::size_t> SoftmaxRegression<T>::predictClass(const Matrix& features) const {
    Matrix probs = predictProbability(features);

    std::size_t numSamples{probs.rows()};
    std::size_t numClasses{probs.cols()};

    std::vector<std::size_t> classes(numSamples);
    const std::vector<double>& probElements = probs.elements();

    for (std::size_t i{0}; i < numSamples; ++i) {
        std::size_t offset{i * numClasses};

        auto maxIter = std::max_element(
            probElements.begin() + offset,
            probElements.begin() + offset + numClasses);

        classes[i] = std::distance(probElements.begin() + offset, maxIter);
    }

    return classes;
}

template <typename T>
void SoftmaxRegression<T>::train(const Dataset<T>& trainingData) {
    std::size_t numSamples = trainingData.countSamples();
    std::size_t numFeatures = trainingData.countFeatures();

    if (numSamples == 0 || numFeatures == 0) {
        throw std::invalid_argument("Cannot train with empty dataset");
    }

    const Matrix& features = trainingData.getFeatures();
    const std::vector<std::size_t>& enumTargets = trainingData.getEnumTargets();

    // Initialize weights and biases
    std::vector<double> initWeights(numFeatures * this->getNumClasses(), 0.01);
    weights_ = Matrix(this->getNumClasses(), numFeatures, initWeights);
    biases_ = std::vector<double>(this->getNumClasses(), 0.0);
    
    for (unsigned int epoch{0}; epoch < epochs_; ++epoch) {
        // Get probability distribution for each sample
        Matrix probs = predictProbability(features);
        const std::vector<double>& probElements = probs.elements();

        // Calculate prediction error (Delta = P - Y)
        std::vector<double> deltaElements = probElements;

        for (std::size_t i{0}; i < numSamples; ++i) {
            std::size_t trueClass = enumTargets[i];

            if (trueClass >= this->getNumClasses()) {
                throw std::out_of_range("Target class index exceeds numClasses");
            }
            // Increase probability of true class
            deltaElements[(i * this->getNumClasses()) + trueClass] -= 1.0;
        }

        Matrix delta(numSamples, this->getNumClasses(), deltaElements);

        // Compute gradients
        Matrix weightGradients = (delta.transpose() * features) * (1.0 / numSamples);
        
        std::vector<double> biasGradients(this->getNumClasses(), 0.0);
        for (std::size_t i{0}; i < numSamples; ++i) {
            for (std::size_t c{0}; c < numClasses_; ++c) {
                biasGradients[c] += delta.elementAt(i, c);
            }
        }

        for (std::size_t c{0}; c < numClasses_; ++c) {
            biasGradients[c] /= numSamples;
        }

        // Update parameters
        weights_ = weights_ - (weightGradients * this->getLearningRate());

        for (std::size_t c{0}; c < numClasses_; ++c) {
            biases_[c] -= this->getLearningRate() * biasGradients[c];
        }
    }
}

template <typename T>
double SoftmaxRegression<T>::validate(
    std::function<double(const std::vector<T>&, const Matrix&)> lossFunction,
    const Dataset<T>& validationData) const {
    std::size_t numSamples = validationData.countSamples();
    std::size_t numFeatures = validationData.countFeatures();

    if (numSamples == 0 || numFeatures == 0) {
        throw std::invalid_argument("Cannot validate with empty dataset");
    }

    const Matrix& features = validationData.getFeatures();
    const std::vector<std::size_t>& enumTargets = validationData.getEnumTargets();

    Matrix probs = this->predictProbability(features);

    return lossFunction(enumTargets, probs.elements());
}

template <typename T>
double SoftmaxRegression<T>::test(
    std::function<double(const std::vector<T>&, const Matrix&)> lossFunction,
    const Dataset<T>& testData) const {
    
    std::size_t numSamples = testData.countSamples();
    std::size_t numFeatures = testData.countFeatures();

    if (numSamples == 0 || numFeatures == 0) {
        throw std::invalid_argument("Cannot test with empty dataset");
    }

    const Matrix& features = testData.getFeatures();
    const std::vector<std::size_t>& enumTargets = testData.getEnumTargets();

    Matrix probs = this->predictProbability(features);

    return lossFunction(enumTargets, probs.elements());
}

template <typename T>
double SoftmaxRegression<T>::categoricalCrossEntropyLoss(
    const std::vector<std::size_t>& enumTargets,
    const std::vector<double>& predictedProbabilities) {
    if (enumTargets.size() == 0 || predictedProbabilities.size() == 0) {
        throw std::invalid_argument("Targets and probabilities cannot be empty");
    }

    double sumLoss{0.0};
    const double epsilon = 1e-15;

    for (std::size_t i{0}; i < enumTargets.size(); ++i) {
        std::size_t offset{i * this->getNumClasses()};

        double predProb{predictedProbabilities[enumTargets[i] + offset]};

        if (predProb < epsilon) {
            predProb = epsilon;
        }

        sumLoss -= std::log(predProb);
    }

    return sumLoss / enumTargets.size();
}

template <typename T>
std::vector<double> SoftmaxRegression<T>::softmax(
    const std::vector<double>& scores) const {
    if (scores.empty()) {
        return {};
    }

    double maxScore = scores[0];

    for (double score : scores) {
        if (score > maxScore) {
            maxScore = score;
        }
    }

    double sumExpScores = std::transform_reduce(
        std::execution::par,
        scores.begin(), scores.end(),
        0.0,
        std::plus<>(),
        [maxScore](double score) {
            return std::exp(score - maxScore);
        }
    );

    std::vector<double> output(scores.size());
    for (std::size_t i{0}; i < scores.size(); ++i) {
        output[i] = std::exp(scores[i] - maxScore) / sumExpScores;
    }

    return output;
}