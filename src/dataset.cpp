#include "dataset.hpp"

template <typename T>
Dataset<T>::Dataset(const Matrix &features, const std::vector<T> &target) :
    features_(features), target_(target) {}

template <typename T>
Dataset<T>::Dataset(std::size_t numSamples, std::size_t numFeatures) :
    features_(Matrix(numSamples, numFeatures)),
    target_(std::vector<T>(numSamples)) {}

template <typename T>
Dataset<T>::Dataset(const Dataset<T>& original) :
    features_(Matrix(original.getFeatures())),
    target_(original.getTarget()) {}

template <typename T>
Dataset<T>& Dataset<T>::operator=(const Dataset<T> &other) {
    if (this != &other) {
        features_ = other.features_;
        target_ = other.target_;
    }

    return *this;
}

template <typename T>
void Dataset<T>::setFeatures(const Matrix &features) {
    features_ = features;
}

template <typename T>
void Dataset<T>::setTarget(const std::vector<T> &target) {
    target_ = target;
}

template <typename T>
const Matrix& Dataset<T>::getFeatures() const {
    return features_;
}

template <typename T>
const std::vector<T>& Dataset<T>::getTarget() const {
    return target_;
}

template <typename T>
std::size_t Dataset<T>::countFeatures() const {
    return features_.cols();
}

template <typename T>
std::size_t Dataset<T>::countSamples() const {
    return features_.rows();
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

    const std::vector<double> &totalFeats = workDataset.getFeatures().elements();
    const std::vector<T> &totalTarget = workDataset.getTarget();

    std::size_t totalSamples = this->countSamples();
    std::size_t trainingEndSample = (totalSamples * trainingPct) / 100;
    std::size_t testEndSample = (totalSamples * (testPct + trainingPct)) / 100;

    /* traininging Data */
    std::vector<T> trainingTarget(
        totalTarget.begin(),
        totalTarget.begin() + trainingEndSample
    );
    std::vector<double> trainingFeats(
        totalFeats.begin(),
        totalFeats.begin() + (trainingEndSample * this->countFeatures())
    );
    Dataset<T> training(
        Matrix(trainingTarget.size(), this->countFeatures(), trainingFeats),
        trainingTarget
    );
    /* Test Data */
    std::vector<T> testTarget(
        totalTarget.begin() + trainingEndSample,
        totalTarget.begin() + testEndSample
    );
    std::vector<double> testFeats(
        totalFeats.begin() + (trainingEndSample * this->countFeatures()),
        totalFeats.begin() + (testEndSample * this->countFeatures())
    );
    Dataset<T> test(
        Matrix(testTarget.size(), this->countFeatures(), testFeats),
        testTarget
    );
    /* Validation Data */
    std::vector<T> valTarget(
        totalTarget.begin() + testEndSample,
        totalTarget.end()
    );
    std::vector<double> valFeats(
        totalFeats.begin() + (testEndSample * this->countFeatures()),
        totalFeats.end()
    );
    Dataset<T> validation(
        Matrix(valTarget.size(), this->countFeatures(), valFeats),
        valTarget
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

    std::vector<T> tempTarget(this->countSamples());

    for (std::size_t i{0}; i < this->countSamples(); ++i) {
        tempTarget[i] = this->target_[indices[i]];
    }

    this->target_ = tempTarget;

    std::vector<double> shuffledFeats(features_.size());
    const std::vector<double> &origFeats = this->features_.elements();

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