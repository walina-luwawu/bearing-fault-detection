#include "dataset.hpp"

#include <algorithm>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>

template <typename T>
Dataset<T>::Dataset(const Matrix& features, const std::vector<T>& targets) :
    features_(features), targets_(targets) {
    enumTargets_ = std::vector<std::size_t>(targets.size());

    std::size_t id{0};
    std::unordered_map<T, std::size_t> valToId;

    for (std::size_t i{0}; i < targets.size(); ++i) {
        auto [iter, inserted] = valToId.insert({targets[i], id});
        
        if (inserted) {
            enumTargets_[i] = id;
            ++id;
        } else {
            enumTargets_[i] = iter->second;
        }
    }
}

template <typename T>
Dataset<T>::Dataset(std::size_t numSamples, std::size_t numFeatures) :
    features_(Matrix(numSamples, numFeatures)),
    targets_(std::vector<T>(numSamples)) {}

template <typename T>
Dataset<T>::Dataset(const Dataset<T>& original) :
    features_(Matrix(original.getFeatures())),
    targets_(original.getTargets()),
    enumTargets_(original.getEnumTargets()) {}

template <typename T>
Dataset<T>& Dataset<T>::operator=(const Dataset<T>& other) {
    if (this != &other) {
        this->features_ = other.features_;
        this->targets_ = other.targets_;
        this->enumTargets_ = other.enumTargets_;
    }

    return *this;
}

template <typename T>
void Dataset<T>::setFeatures(const Matrix& features) {
    this->features_ = features;
}

template <typename T>
void Dataset<T>::setTargets(const std::vector<T>& targets) {
    this->targets_ = targets;
    setEnumTargets();
}

template <typename T>
void Dataset<T>::setEnumTargets() {
    this->enumTargets_ = std::vector<std::size_t>(this->getTargets().size());

    std::size_t id{0};
    std::unordered_map<T, std::size_t> valToId;

    const std::vector<T>& targets = this->getTargets();

    for (std::size_t i{0}; i < targets.size(); ++i) {
        auto [iter, inserted] = valToId.insert({targets[i], id});
        
        if (inserted) {
            this->enumTargets_[i] = id;
            ++id;
        } else {
            this->enumTargets_[i] = iter->second;
        }
    }
}

template <typename T>
const Matrix& Dataset<T>::getFeatures() const {
    return this->features_;
}

template <typename T>
const std::vector<T>& Dataset<T>::getTargets() const {
    return this->targets_;
}

template <typename T>
const std::vector<std::size_t>& Dataset<T>::getEnumTargets() const {
    return this->enumTargets_;
}

template <typename T>
std::size_t Dataset<T>::countFeatures() const {
    return this->features_.cols();
}

template <typename T>
std::size_t Dataset<T>::countSamples() const {
    return this->features_.rows();
}

template <typename T>
std::tuple<Dataset<T>, Dataset<T>, Dataset<T>> Dataset<T>::split(
    uint8_t trainingPct,
    uint8_t testPct,
    bool shuffle
) const {
    if (testPct == 0 && trainingPct == 0) {
        throw std::invalid_argument("Cannot split dataset into zero-sized slices");
    }

    if (testPct + trainingPct > 100) {
        throw std::out_of_range("Dataset slices cannot be larger than 100%");
    }

    Dataset<T> workDataset(*this);

    if (shuffle) {
        workDataset.shuffle();
    }

    const std::vector<double>& totalFeats = workDataset.getFeatures().elements();
    const std::vector<T>& totalTargets = workDataset.getTargets();

    std::size_t totalSamples = this->countSamples();
    std::size_t trainingEndSample = (totalSamples * trainingPct) / 100;
    std::size_t testEndSample = (totalSamples * (testPct + trainingPct)) / 100;

    /* training Data */
    std::vector<T> trainingTargets(
        totalTargets.begin(),
        totalTargets.begin() + trainingEndSample
    );
    std::vector<double> trainingFeats(
        totalFeats.begin(),
        totalFeats.begin() + (trainingEndSample * this->countFeatures())
    );
    Dataset<T> training(
        Matrix(trainingTargets.size(), this->countFeatures(), trainingFeats),
        trainingTargets
    );
    /* Test Data */
    std::vector<T> testTargets(
        totalTargets.begin() + trainingEndSample,
        totalTargets.begin() + testEndSample
    );
    std::vector<double> testFeats(
        totalFeats.begin() + (trainingEndSample * this->countFeatures()),
        totalFeats.begin() + (testEndSample * this->countFeatures())
    );
    Dataset<T> test(
        Matrix(testTargets.size(), this->countFeatures(), testFeats),
        testTargets
    );
    /* Validation Data */
    std::vector<T> valTargets(
        totalTargets.begin() + testEndSample,
        totalTargets.end()
    );
    std::vector<double> valFeats(
        totalFeats.begin() + (testEndSample * this->countFeatures()),
        totalFeats.end()
    );
    Dataset<T> validation(
        Matrix(valTargets.size(), this->countFeatures(), valFeats),
        valTargets
    );

    return {training, test, validation};
}

template <typename T>
void Dataset<T>::shuffle() {
    std::vector<std::size_t> indices(this->countSamples());
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device seed;
    std::mt19937 engine(seed());
    std::shuffle(indices.begin(), indices.end(), engine);

    std::vector<T> tempTargets(this->countSamples());

    for (std::size_t i{0}; i < this->countSamples(); ++i) {
        tempTargets[i] = this->targets_[indices[i]];
    }

    this->targets_ = tempTargets;

    std::vector<double> shuffledFeats(features_.size());
    const std::vector<double>& origFeats = this->getFeatures().elements();

    for (std::size_t i{0}; i < this->countSamples(); ++i) {
        std::size_t srcStart = indices[i] * this->countFeatures();
        std::size_t dstStart = i * this->countFeatures();

        std::copy(
            origFeats.begin() + srcStart,
            origFeats.begin() + srcStart + this->countFeatures(),
            shuffledFeats.begin() + dstStart);
    }

    this->features_.setElements(shuffledFeats);
}