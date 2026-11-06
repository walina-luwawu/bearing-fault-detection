#include "matrix.hpp"
#include <cstdint>
#include <vector>

template <typename T>
class Dataset {
    private:
        Matrix features_;
        std::vector<T> targets_;
    public:
        Dataset(const Matrix &features, const std::vector<T> &targets);
        Dataset(std::size_t numSamples, std::size_t numFeatures);
        Dataset(const Dataset<T> &original);
        
        Dataset<T>& operator=(const Dataset<T> &other);

        void setFeatures(const Matrix &features);
        void setTargets(const std::vector<T> &targets);

        const Matrix& getFeatures() const;
        const std::vector<T>& getTargets() const;
        std::size_t countFeatures() const;
        std::size_t countSamples() const;

        std::tuple<Dataset<T>, Dataset<T>, Dataset<T>> split(
            uint8_t trainingPct,
            uint8_t testPct,
            bool shuffle
        ) const;
        void shuffle();
};