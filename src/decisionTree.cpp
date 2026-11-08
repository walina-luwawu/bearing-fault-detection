#include "decisionTree.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

template <typename T>
DecisionTree<T>::DecisionTree(
    std::size_t maxDepth,
    std::size_t minSamplesSplit) :
    root_(nullptr), maxDepth_(maxDepth), minSamplesSplit_(minSamplesSplit) {
    if (maxDepth == 0) {
        throw std::invalid_argument("Max tree depth must be greater than 0");
    }

    if (minSamplesSplit == 0) {
        throw std::invalid_argument("Branches must have at least one sample");
    }
}

template <typename T>
std::size_t DecisionTree<T>::getMaxDepth() const {
    return this->maxDepth_;
}

template <typename T>
std::size_t DecisionTree<T>::getMinSamplesSplit() const {
    return this->minSamplesSplit_;
}

template <typename T>
const typename DecisionTree<T>::Node* DecisionTree<T>::getRoot() const {
    return this->root_;
}

template <typename T>
void DecisionTree<T>::fit(const Dataset<T>& dataset) {
    if (dataset.countFeatures() == 0 || dataset.countSamples == 0) {
        throw std::invalid_argument("Cannot fit empty dataset");
    }

    root_ = buildTree(dataset.getFeatures(), dataset.getTargets(), 0);
}

template <typename T>
T DecisionTree<T>::predict(const std::vector<double>& sample) const {
    const Node* ptr_node= this->getRoot();

    while (!ptr_node->isLeaf) {
        if (sample[ptr_node->featureIndex] < ptr_node->threshold) {
            ptr_node = ptr_node->left.get();
        }
        else {
            ptr_node = ptr_node->right.get();
        }
    }

    return ptr_node->prediction;
}

template <typename T>
std::vector<T> DecisionTree<T>::predict(const Matrix& features) const {
    std::vector<T> predictions(features.rows());

    for (std::size_t i{0}; i < features.rows(); ++i) {
        std::vector<double> sample(
            features.elements().begin() + i * features.cols(),
            features.elements().begin() + (i + 1) * features.cols());

        predictions[i] = predict(sample);
    }

    return predictions;
}

template <typename T>
Matrix DecisionTree<T>::getConfusionMatrix(const Dataset<T>& dataset) const {
    const std::vector<T>& targets = dataset.getTargets();
    if (targets.empty()) {
        return Matrix(0, 0);
    }

    std::vector<T> predictions = this->predict(dataset.getFeatures());

    std::unordered_set<T> uniqueClassesSet;
    for (const auto& t : targets) uniqueClassesSet.insert(t);
    for (const auto& p : predictions) uniqueClassesSet.insert(p);

    std::vector<T> uniqueClasses(uniqueClassesSet.begin(), uniqueClassesSet.end());
    std::sort(uniqueClasses.begin(), uniqueClasses.end());

    std::size_t numClasses = uniqueClasses.size();

    std::unordered_map<T, std::size_t> classToIndex;
    for (std::size_t i{0}; i < numClasses; ++i) {
        classToIndex[uniqueClasses[i]] = i;
    }

    Matrix confMatrix(numClasses, numClasses);

    std::vector<double> counts(numClasses * numClasses, 0.0);

    for (std::size_t i{0}; i < targets.size(); ++i) {
        std::size_t trueIndex = classToIndex[targets[i]];
        std::size_t predIndex = classToIndex[predictions[i]];

        counts[trueIndex * numClasses + predIndex] += 1.0;
    }

    confMatrix.setElements(counts);
    return confMatrix;
}

template <typename T>
std::unordered_map<T, typename DecisionTree<T>::ClassMetrics> DecisionTree<T>::getClassMetrics(
    const Dataset<T>& dataset) const {
    const std::vector<T>& targets = dataset.getTargets();
    if (targets.empty()) {
        return {};
    }

    std::vector<T> predictions = this->predict(dataset.getFeatures());

    std::unordered_set<T> uniqueClassesSet;
    for (const auto& t : targets) {
        uniqueClassesSet.insert(t);
    }
    for (const auto& p : predictions) {
        uniqueClassesSet.insert(p);
    }
    
    std::vector<T> uniqueClasses(uniqueClassesSet.begin(), uniqueClassesSet.end());
    std::sort(uniqueClasses.begin(), uniqueClasses.end());

    std::size_t numClasses = uniqueClasses.size();
    std::unordered_map<T, std::size_t> classToIndex;
    for (std::size_t i{0}; i < numClasses; ++i) {
        classToIndex[uniqueClasses[i]] = i;
    }

    Matrix confMat = this->getConfusionMatrix(dataset);

    std::unordered_map<T, typename DecisionTree<T>::ClassMetrics> metricsMap;

    for (std::size_t i{0}; i < numClasses; ++i) {
        T uniqueClass = uniqueClasses[i];
        
        double truePositives{confMat.elementAt(i, i)};
        
        double falsePositives{0.0};
        for (std::size_t r{0}; r < numClasses; ++r) {
            if (r != i) {
                falsePositives += confMat.elementAt(r, i);
            }
        }

        double falseNegatives{0.0};
        for (std::size_t c{0}; c < numClasses; ++c) {
            if (c != i) {
                falseNegatives += confMat.elementAt(i, c);
            }
        }

        double precision{0.0};
        double recall{0.0};
        double f1{0.0};

        if (truePositives + falsePositives > 0.0) {
            precision = truePositives / (truePositives + falsePositives);
        }

        if (truePositives + falseNegatives > 0.0) {
            recall = truePositives / (truePositives + falseNegatives);
        }

        if (precision + recall > 0.0) {
            f1 = 2.0 * precision * recall / (precision + recall);
        }

        metricsMap[uniqueClass] = {precision, recall, f1};
    }

    return metricsMap;
}

template <typename T>
std::unique_ptr<typename DecisionTree<T>::Node> DecisionTree<T>::buildTree(
    const Matrix& features,
    const std::vector<T>& targets,
    std::size_t currentDepth) {
    if (currentDepth == this->getMaxDepth() ||
        targets.size() < this->getMinSamplesSplit() ||
        this->calculateGini(targets) == 0.0) {
        return std::make_unique<Node>(this->getMostCommonClass(targets));
    }

    auto [featIndex, threshold, gain] = this->findBestSplit(features, targets);

    if (gain <= 0.0) {
        return std::make_unique<Node>(this->getMostCommonClass(targets));
    }

    std::vector<double> leftElements, rightElements;
    std::vector<T> leftTargets, rightTargets;

    for (std::size_t r{0}; r < features.rows(); ++r) {
        if (features.elementAt(r, featIndex) < threshold) {
            for (std::size_t c{0}; c < features.cols(); ++c) {
                leftElements.push_back(features.elementAt(r, c));
            }
            leftTargets.push_back(targets[r]);
        } else {
            for (std::size_t c{0}; c < features.cols(); ++c) {
                rightElements.push_back(features.elementAt(r, c));
            }
            rightTargets.push_back(targets[r]);
        }
    }

    if (leftTargets.empty() || rightTargets.empty()) {
        return std::make_unique<Node>(this->getMostCommonClass(targets));
    }

    Matrix leftFeatures(
        leftElements.size() / features.cols(),
        features.cols(),
        leftElements);
    std::unique_ptr<Node> leftChild = buildTree(leftFeatures, leftTargets, currentDepth + 1);

    Matrix rightFeatures(
        rightElements.size() / features.cols(),
        features.cols(),
        rightElements);
    std::unique_ptr<Node> rightChild = buildTree(rightFeatures, rightTargets, currentDepth + 1);

    return std::make_unique<Node>(
        featIndex,
        threshold,
        std::move(leftChild),
        std::move(rightChild));
}

template <typename T>
std::tuple<std::size_t, double, double> DecisionTree<T>::findBestSplit(
    const Matrix& features,
    const std::vector<T>& targets) {
    double bestGain{-1.0};
    double bestThreshold{0.0};
    std::size_t bestFeatureIndex{0};

    double parentGini{calculateGini(targets)};
    std::unordered_set<double> testThresholds;

    for (std::size_t i{0}; i < features.cols(); ++i) {
        std::vector<double> featVals(features.rows());

        for (std::size_t r{0}; r < features.rows(); ++r) {
            featVals[r] = features.elementAt(r, i);
        }

        std::vector<std::size_t> indices(featVals.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::sort(
            indices.begin(), indices.end(),
            [&](std::size_t m, std::size_t n){
                return featVals[m] < featVals[n];
            });

        std::vector<double> sortedFeatVals(featVals.size());
        std::vector<T> sortedTargets(featVals.size());

        for (std::size_t i{0}; i < featVals.size(); ++i) {
            sortedFeatVals[i] = featVals[indices[i]];
            sortedTargets[i] = targets[indices[i]];
        }

        for (std::size_t j{0}; j < sortedFeatVals.size() - 1; ++j) {
            if (sortedFeatVals[j] == sortedFeatVals[j + 1]) continue;

            double midpoint{(sortedFeatVals[j] + sortedFeatVals[j + 1]) / 2.0};

            auto iter = std::lower_bound(
                sortedFeatVals.begin(),
                sortedFeatVals.end(),
                midpoint);

            std::size_t splitIndex = std::distance(sortedFeatVals.begin(), iter);

            if (splitIndex == 0 || splitIndex == sortedFeatVals.size()) continue;

            std::vector<T> left(
                sortedTargets.begin(),
                sortedTargets.begin() + splitIndex);
            std::vector<T> right(
                sortedTargets.begin() + splitIndex,
                sortedTargets.end());

            double numSamples = static_cast<double>(sortedFeatVals.size());

            double weightedLeftGini{
                (static_cast<double>(left.size())/numSamples) * calculateGini(left)};
            double weightedRightGini{
                (static_cast<double>(right.size())/numSamples) * calculateGini(right)};

            double weightedGini{weightedLeftGini + weightedRightGini};

            double gain{parentGini - weightedGini};

            if (gain > bestGain) {
                bestGain = gain;
                bestThreshold = midpoint;
                bestFeatureIndex = i;
            }
        }
    }

    return {bestFeatureIndex, bestThreshold, bestGain};
}

template <typename T>
T DecisionTree<T>::getMostCommonClass(const std::vector<T>& targets) const {
    if (targets.empty()) {
        throw std::invalid_argument("Tree must have at least one target");
    }

    std::unordered_map<T, std::size_t> classToFreq;

    for (std::size_t i{0}; i < targets.size(); ++i) {
        ++classToFreq[targets[i]];
    }

    auto maxIter = std::max_element(
        classToFreq.begin(), classToFreq.end(),
        [](const auto& keyVal1, const auto& keyVal2) {
            return keyVal1.second < keyVal2.second;
        }
    );

    return maxIter->first;
}

template <typename T>
double DecisionTree<T>::calculateGini(const std::vector<T>& targets) const {
    if (targets.empty()) {
        throw std::invalid_argument("Tree must have at least one target");
    }

    std::unordered_map<T, std::size_t> classToFreq;

    for (std::size_t i{0}; i < targets.size(); ++i) {
        ++classToFreq[targets[i]];
    }

    double sumSquareProbs{0};
    double numSamples = static_cast<double>(targets.size());

    for (const auto& [key, val] : classToFreq) {
        double prob{static_cast<double>(val) / numSamples};
        sumSquareProbs += prob * prob;
    }

    return 1.0 - sumSquareProbs;
}