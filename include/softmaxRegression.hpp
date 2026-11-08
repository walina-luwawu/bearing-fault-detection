#ifndef SOFTMAX_REGRESSION_HPP
#define SOFTMAX_REGRESSION_HPP

#include "dataset.hpp"
#include <functional>
#include <unordered_map>
#include <vector>

template <typename T>
class SoftmaxRegression {
    private:
        double learningRate_;
        unsigned int epochs_;
        std::size_t numClasses_;
        Matrix weights_;
        std::vector<double> biases_;

        std::vector<double> softmax(const std::vector<double>& scores) const;

    public:
        struct ClassMetrics {
            double precision;
            double recall;
            double f1Score;
        };
        SoftmaxRegression(double learningRate, unsigned int epochs, std::size_t numClasses);

        double getLearningRate() const;
        unsigned int getEpochs() const;
        std::size_t getNumClasses() const;
        const Matrix& getWeights() const;
        const std::vector<double>& getBiases() const;

        void setLearningRate(double learningRate);
        void setEpochs(unsigned int epochs);

        Matrix predictLogits(const Matrix& features) const;
        Matrix predictProbability(const Matrix& features) const;
        std::vector<std::size_t> predictClass(const Matrix& features) const;
        void train(const Dataset<T>& trainingData);

        double validate(
            std::function<double(const std::vector<T>&, const Matrix&)> lossFunction,
            const Dataset<T>& validationData) const;

        double test(
            std::function<double(const std::vector<T>&, const Matrix&)> lossFunction,
            const Dataset<T>& testData) const;

        Matrix getConfusionMatrix(const Dataset<T>& dataset) const;

        std::unordered_map<T, ClassMetrics> getClassMetrics(const Dataset<T>& dataset) const;

        double categoricalCrossEntropyLoss(
            const std::vector<std::size_t>& enumTargets,
            const std::vector<double>& predictedProbabilities);
};

#endif