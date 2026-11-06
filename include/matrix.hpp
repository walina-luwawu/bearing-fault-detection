#include <vector>
#include <iostream>
#include <stdexcept>

class Matrix {
    private:
        std::size_t rows_;
        std::size_t cols_;
        std::vector<double> elements_;

    public:
        Matrix(std::size_t rows, std::size_t cols, std::vector<double> elements);
        Matrix(std::size_t rows, std::size_t cols);
        Matrix(const Matrix &other);

        Matrix& operator=(const Matrix &other);
        Matrix& operator+=(const Matrix &other);
        Matrix& operator-=(const Matrix & other);
        Matrix& operator*=(const Matrix &other);
        Matrix& operator*=(double scalar);

        Matrix operator+(const Matrix &other) const;
        Matrix operator-(const Matrix &other) const;
        Matrix operator*(const Matrix &other) const;
        Matrix operator*(double scalar) const;
        std::vector<double> operator*(const std::vector<double> &x) const;

        void setElements(const std::vector<double> &elements);
        void setDimensions(const std::size_t rows, const std::size_t cols);

        std::size_t rows() const;
        std::size_t cols() const;
        std::size_t size() const;
        const std::vector<double>& elements() const;
        double elementAt(std::size_t row, std::size_t col) const;

        std::vector<double> multiply(const std::vector<double> &x) const;
        Matrix multiply(const Matrix &other) const;
        Matrix add(const Matrix &other) const;
        Matrix subtract(const Matrix &other) const;
        Matrix transpose() const;
        void scale(double scalar);
    
};

Matrix operator*(double scalar, const Matrix &A);
std::vector<double> operator*(const std::vector<double> &x, const Matrix &A);
std::ostream& operator<<(std::ostream& os, const Matrix& matrix);