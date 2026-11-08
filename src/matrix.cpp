#include "matrix.hpp"
#include "cblas.h"
#include <stdexcept>

Matrix::Matrix(std::size_t rows, std::size_t cols, std::vector<double> elements) :
    rows_(rows), cols_(cols), elements_(std::move(elements))
{
    if (rows_ * cols_ != elements_.size()) {
        throw std::invalid_argument("Matrix dimensions must match size");
    }
}

Matrix::Matrix(std::size_t rows, std::size_t cols) :
    rows_(rows), cols_(cols), elements_(rows*cols, 0.0) {}
    
Matrix::Matrix(const Matrix& other) :
    rows_(other.rows()),
    cols_(other.cols()),
    elements_(other.elements_) {}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        this->rows_ = other.rows();
        this->cols_ = other.cols();
        this->elements_ = other.elements_;
    }

    return *this;
}

Matrix Matrix::operator+(const Matrix& other) const {
    return this->add(other);
}

Matrix Matrix::operator-(const Matrix& other) const {
    return this->subtract(other);
}

Matrix Matrix::operator*(const Matrix& other) const {
    return this->multiply(other);
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result = *this;
    result.scale(scalar);
    return result;
}

std::vector<double> Matrix::operator*(const std::vector<double>& x) const {
    return this->multiply(x);
}

Matrix& Matrix::operator+=(const Matrix& other) {
    *this = *this + other;
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    *this = *this - other;
    return *this;
}

Matrix& Matrix::operator*=(const Matrix& other) {
    *this = *this * other;
    return *this;
}

Matrix& Matrix::operator*=(double scalar) {
    *this = *this *scalar;
    return *this;
}

std::size_t Matrix::rows() const {
    return this->rows_;
}

std::size_t Matrix::cols() const {
    return this->cols_;
}

std::size_t Matrix::size() const {
    return this->elements_.size();
}

const std::vector<double>& Matrix::elements() const {
    return this->elements_;
}

double Matrix::elementAt(const std::size_t row, const std::size_t col) const {
    if (row >= this->rows() || col >= this->cols()) {
        throw std::out_of_range("Row/Col is out of range of matrix");
    }
    return this->elements().at((this->cols() * row) + col);
}

void Matrix::setElements(const std::vector<double>& elements) {
    if (elements.size() != this->size()) {
        throw std::invalid_argument("Number of elements must match matrix size");
    }

    this->elements_ = elements;
}

void Matrix::setDimensions(const std::size_t rows, const std::size_t cols) {
    if (rows*cols != this->size()) {
        throw std::invalid_argument("New dimensions must fit matrix size");
    }

    this->rows_ = rows;
    this->cols_ = cols;
 }

std::vector<double> Matrix::multiply(const std::vector<double>& x) const {
    if (x.size() != cols()) {
        throw std::invalid_argument("Vector must be length n to multiply by mxn matrix");
    }

    // Note: Make sure result vector has the right size before calling cblas_dgemv!
    std::vector<double> result(rows(), 0.0);

    cblas_dgemv(
        CblasRowMajor,
        CblasNoTrans,
        this->rows(),
        this->cols(),
        1.0,
        this->elements().data(),
        this->cols(),
        x.data(),
        1,
        0.0,
        result.data(),
        1
    );

    return result;
}

Matrix Matrix::multiply(const Matrix& other) const {
    if (other.rows() != this->cols()) {
        throw std::invalid_argument("Matrix inner dimensions must match for multiplication");
    }

    Matrix result = Matrix(this->rows(), other.cols());
    
    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        this->rows(),
        other.cols(),
        this->cols(),
        1.0,
        this->elements().data(),
        this->cols(),
        other.elements().data(),
        other.cols(),
        0.0,
        result.elements_.data(),
        result.cols()
    );

    return result;
}

Matrix Matrix::add(const Matrix& other) const {
    if (this->rows() != other.rows() || this->cols() != other.cols()) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    Matrix result = Matrix(*this);

    cblas_daxpy(
        this->size(),
        1.0,
        other.elements().data(),
        1,
        result.elements_.data(),
        1
    );

    return result;
}

Matrix Matrix::subtract(const Matrix& other) const {
    if (this->rows() != other.rows() || this->cols() != other.cols()) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }
    Matrix copy = Matrix(other);
    copy.scale(-1.0);

    return this->add(copy);
}

Matrix Matrix::transpose() const {
    Matrix transpose = Matrix(this->cols(), this->rows());

    for (std::size_t i{0}; i < this->size(); ++i) {
        std::size_t row{i / this->cols()};
        std::size_t col{i % this->cols()};

        transpose.elements_.at(transpose.cols() * col + row) = this->elementAt(row, col);
    }

    return transpose;
}

void Matrix::scale(double scalar) {
    cblas_dscal(
        this->size(),
        scalar,
        this->elements_.data(),
        1
    );
}

std::vector<double> operator*(
    const std::vector<double> &x,
    const Matrix &A) {
    if (x.size() != A.rows()) {
        throw std::invalid_argument("Vector must be length m to multiply by mxn matrix");
    }

    std::vector<double> result(A.cols(), 0.0);

    cblas_dgemv(
        CblasRowMajor,
        CblasTrans,
        A.rows(),
        A.cols(),
        1.0,
        A.elements().data(),
        A.cols(),
        x.data(),
        1,
        0.0,
        result.data(),
        1
    );

    return result;
}

Matrix operator*(double scalar, const Matrix& A) {
    Matrix result = A;
    result.scale(scalar);
    return result;
}

std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
    if (matrix.size() == 0) {
        os << "[]" << std::endl;
        return os;
    }

    for (std::size_t i{0}; i < matrix.size(); ++i) {
        std::size_t row{i / matrix.cols()};
        std::size_t col{i % matrix.cols()};

        os << matrix.elementAt(row, col);

        if (col + 1 < matrix.cols()) {
            os << ",";
        }
        else {
            os << std::endl;
        }
    }
    
    return os;
}