#include "matrix.hpp"
#include <random>
#include <vector>
#include <utility>
#include <stdexcept>

template <typename T>
class Dataset {
    private:
        Matrix features_;
        std::vector<T> target_;
    public:
        Dataset(const Matrix &features, const std::vector<T> &targets);
        Dataset(std::size_t numSamples, std::size_t numFeatures);
        Dataset(const Dataset<T> &original);
        
        Dataset<T>& operator=(const Dataset<T> &other);

        void setFeatures(const Matrix &features);
        void setTarget(const std::vector<T> &target);

        const Matrix& getFeatures() const;
        const std::vector<T>& getTarget() const;
        std::size_t countFeatures() const;
        std::size_t countSamples() const;

        std::tuple<Dataset<T>, Dataset<T>, Dataset<T>> split(
            uint8_t trainingPct,
            uint8_t testPct,
            bool shuffle
        ) const;
        void shuffle();
};