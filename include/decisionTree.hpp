#ifndef DECISION_TREE_HPP
#define DECISION_TREE_HPP

#include "dataset.hpp"
#include "matrix.hpp"
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

template <typename T>
class DecisionTree {
private:
    struct Node {
        bool isLeaf;
        T prediction;
        std::size_t featureIndex;
        double threshold;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;

        // Constructor for a leaf node
        Node(T pred) : isLeaf(true), prediction(pred), featureIndex(0),
            threshold(0.0), left(nullptr), right(nullptr) {}

        // Constructor for a decision node
        Node(std::size_t featIndex, double thresh,
            std::unique_ptr<Node> l, std::unique_ptr<Node> r) :
            isLeaf(false), prediction(), featureIndex(featIndex),
            threshold(thresh), left(std::move(l)), right(std::move(r)) {}
    };

    std::unique_ptr<Node> root_;
    std::size_t maxDepth_;
    std::size_t minSamplesSplit_;

    std::unique_ptr<Node> buildTree(
        const Matrix& features,
        const std::vector<T>& targets,
        std::size_t currentDepth);
    
    std::tuple<std::size_t, double, double> findBestSplit(
        const Matrix& features,
        const std::vector<T>& targets);
    T getMostCommonClass(const std::vector<T>& targets) const;
    double calculateGini(const std::vector<T>& targets) const;

public:
    struct ClassMetrics {
        double precision;
        double recall;
        double f1Score;
    };

    DecisionTree(std::size_t maxDepth, std::size_t minSamplesSplit);

    std::size_t getMaxDepth() const;
    std::size_t getMinSamplesSplit() const;
    const Node* getRoot() const;

    void fit(const Dataset<T>& dataset);
    T predict(const std::vector<double>& sample) const;
    std::vector<T> predict(const Matrix& features) const;

    Matrix getConfusionMatrix(const Dataset<T>& dataset) const;
    std::unordered_map<T, ClassMetrics> getClassMetrics(const Dataset<T>& dataset) const;
};

#endif