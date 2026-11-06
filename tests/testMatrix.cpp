#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <sstream>
#include <vector>

#include "matrix.hpp"

/* Constructor(rows, cols, elements) */

TEST(MatrixConstructorWithElements, MbyN) {
    Matrix A(2, 3, {1, 2, 3, 4, 5, 6});

    EXPECT_EQ(A.rows(), 2);
    EXPECT_EQ(A.cols(), 3);
    EXPECT_THAT(A.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{1, 2, 3, 4, 5, 6}));
}

TEST(MatrixConstructorWithElements, NbyN) {
    Matrix A(2, 2, {1, 2, 3, 4});

    EXPECT_EQ(A.rows(), 2);
    EXPECT_EQ(A.cols(), 2);
    EXPECT_THAT(A.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{1, 2, 3, 4}));
}

TEST(MatrixConstructorWithElements, ZeroRowsByN) {
    Matrix A(0, 3, {});

    EXPECT_EQ(A.rows(), 0);
    EXPECT_EQ(A.cols(), 3);
    EXPECT_THAT(A.elements(), testing::IsEmpty());
}

TEST(MatrixConstructorWithElements, MbyZeroCols) {
    Matrix A(3, 0, {});

    EXPECT_EQ(A.rows(), 3);
    EXPECT_EQ(A.cols(), 0);
    EXPECT_THAT(A.elements(), testing::IsEmpty());
}

TEST(MatrixConstructorWithElements, ZeroByZero) {
    Matrix A(0, 0, {});

    EXPECT_EQ(A.rows(), 0);
    EXPECT_EQ(A.cols(), 0);
    EXPECT_THAT(A.elements(), testing::IsEmpty());
}

TEST(MatrixConstructorWithElements, TooFewElementsThrows) {
    EXPECT_THROW(Matrix(2, 3, {1, 2, 3, 4, 5}), std::invalid_argument);
}

TEST(MatrixConstructorWithElements, TooManyElementsThrows) {
    EXPECT_THROW(Matrix(2, 3, {1, 2, 3, 4, 5, 6, 7}), std::invalid_argument);
}

TEST(MatrixConstructorWithElements, ElementsForZeroRowMatrixThrows) {
    EXPECT_THROW(Matrix(0, 3, {1}), std::invalid_argument);
}

TEST(MatrixConstructorWithElements, ElementsForZeroColMatrixThrows) {
    EXPECT_THROW(Matrix(3, 0, {1}), std::invalid_argument);
}

/* Constructor(rows, cols) */

TEST(MatrixConstructorDimensionsOnly, MbyNIsZeroFilled) {
    Matrix A(2, 3);

    EXPECT_EQ(A.rows(), 2);
    EXPECT_EQ(A.cols(), 3);
    EXPECT_THAT(A.elements(), testing::AllOf(
            testing::SizeIs(6), testing::Each(testing::DoubleEq(0.0))));
}

TEST(MatrixConstructorDimensionsOnly, NbyNIsZeroFilled) {
    Matrix A(2, 2);

    EXPECT_EQ(A.rows(), 2);
    EXPECT_EQ(A.cols(), 2);
    EXPECT_THAT(A.elements(), testing::AllOf(
        testing::SizeIs(4), testing::Each(testing::DoubleEq(0.0))));
}

TEST(MatrixConstructorDimensionsOnly, ZeroRowsByNIsEmpty) {
    Matrix A(0, 3);

    EXPECT_EQ(A.rows(), 0);
    EXPECT_EQ(A.cols(), 3);
    EXPECT_THAT(A.elements(), testing::IsEmpty());
}

TEST(MatrixConstructorDimensionsOnly, MbyZeroColsIsEmpty) {
    Matrix A(3, 0);

    EXPECT_EQ(A.rows(), 3);
    EXPECT_EQ(A.cols(), 0);
    EXPECT_THAT(A.elements(), testing::IsEmpty());
}

TEST(MatrixConstructorDimensionsOnly, ZeroByZeroIsEmpty) {
    Matrix A(0, 0);

    EXPECT_EQ(A.rows(), 0);
    EXPECT_EQ(A.cols(), 0);
    EXPECT_THAT(A.elements(), testing::IsEmpty());
}

/* Copy Constructor */

TEST(MatrixCopyConstructor, CopiesDimensionsAndElements) {
    Matrix A(2, 3, {1, 2, 3, 4, 5, 6});

    Matrix B(A);

    EXPECT_EQ(B.rows(), 2);
    EXPECT_EQ(B.cols(), 3);
    EXPECT_THAT(B.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{1, 2, 3, 4, 5, 6}));
}

TEST(MatrixCopyConstructor, CopyIsIndependentOfOriginal) {
    Matrix A(2, 2, {1, 2, 3, 4});
    Matrix B(A);

    B.setElements({9, 9, 9, 9});

    EXPECT_THAT(A.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{1, 2, 3, 4}));
}

TEST(MatrixCopyConstructor, ZeroRowsByN) {
    Matrix A(0, 3);

    Matrix B(A);

    EXPECT_EQ(B.rows(), 0);
    EXPECT_EQ(B.cols(), 3);
    EXPECT_THAT(B.elements(), testing::IsEmpty());
}

TEST(MatrixCopyConstructor, MbyZeroCols) {
    Matrix A(3, 0);

    Matrix B(A);

    EXPECT_EQ(B.rows(), 3);
    EXPECT_EQ(B.cols(), 0);
    EXPECT_THAT(B.elements(), testing::IsEmpty());
}

TEST(MatrixCopyConstructor, ZeroByZero) {
    Matrix A(0, 0);

    Matrix B(A);

    EXPECT_EQ(B.rows(), 0);
    EXPECT_EQ(B.cols(), 0);
    EXPECT_THAT(B.elements(), testing::IsEmpty());
}

/* operator= */

TEST(MatrixAssignment, CopiesDimensionsAndElements) {
    Matrix A(2, 3, {1, 2, 3, 4, 5, 6});
    Matrix B(2, 2);

    B = A;

    EXPECT_EQ(B.rows(), 2);
    EXPECT_EQ(B.cols(), 3);
    EXPECT_THAT(B.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{1, 2, 3, 4, 5, 6}));
}

TEST(MatrixAssignment, AssignedCopyIsIndependentOfOriginal) {
    Matrix A(2, 2, {1, 2, 3, 4});
    Matrix B(2, 2);
    B = A;

    B.setElements({9, 9, 9, 9});

    EXPECT_THAT(A.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{1, 2, 3, 4}));
}

TEST(MatrixAssignment, ReturnsReferenceToLeftHandSide) {
    Matrix A(2, 2, {1, 2, 3, 4});
    Matrix B(2, 2);

    Matrix &result = (B = A);

    EXPECT_EQ(&result, &B);
}

TEST(MatrixAssignment, SelfAssignmentLeavesMatrixUnchanged) {
    Matrix A(2, 2, {1, 2, 3, 4});
    Matrix &alias = A;

    A = alias;

    EXPECT_EQ(A.rows(), 2);
    EXPECT_EQ(A.cols(), 2);
    EXPECT_THAT(A.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{1, 2, 3, 4}));
}

TEST(MatrixAssignment, EmptyMatrixOverNonEmptyMatrix) {
    Matrix A(0, 0);
    Matrix B(2, 3, {1, 2, 3, 4, 5, 6});

    B = A;

    EXPECT_EQ(B.rows(), 0);
    EXPECT_EQ(B.cols(), 0);
    EXPECT_THAT(B.elements(), testing::IsEmpty());
}

TEST(MatrixAssignment, NonEmptyMatrixOverEmptyMatrix) {
    Matrix A(2, 3, {1, 2, 3, 4, 5, 6});
    Matrix B(0, 0);

    B = A;

    EXPECT_EQ(B.rows(), 2);
    EXPECT_EQ(B.cols(), 3);
    EXPECT_THAT(B.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{1, 2, 3, 4, 5, 6}));
}

/* rows(), cols(), size(), elements() */

TEST(MatrixRows, ReturnsNumberOfRows) {
    Matrix A(2, 3);

    EXPECT_EQ(A.rows(), 2);
}

TEST(MatrixCols, ReturnsNumberOfColumns) {
    Matrix A(2, 3);

    EXPECT_EQ(A.cols(), 3);
}

TEST(MatrixSize, MbyNIsRowsTimesCols) {
    Matrix A(2, 3);

    EXPECT_EQ(A.size(), 6);
}

TEST(MatrixSize, ZeroRowsByNIsZero) {
    Matrix A(0, 3);

    EXPECT_EQ(A.size(), 0);
}

TEST(MatrixSize, MbyZeroColsIsZero) {
    Matrix A(3, 0);

    EXPECT_EQ(A.size(), 0);
}

TEST(MatrixSize, ZeroByZeroIsZero) {
    Matrix A(0, 0);

    EXPECT_EQ(A.size(), 0);
}

TEST(MatrixElements, ReturnsElementsInRowMajorOrder) {
    Matrix A(2, 3, {1, 2, 3, 4, 5, 6});

    EXPECT_THAT(A.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{1, 2, 3, 4, 5, 6}));
}

/* elementAt() */

TEST(MatrixElementAt, ReturnsElementAtGivenPositionInRowMajorOrder) {
    Matrix A(2, 3, {1, 2, 3, 4, 5, 6});

    EXPECT_DOUBLE_EQ(A.elementAt(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(A.elementAt(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(A.elementAt(0, 2), 3.0);
    EXPECT_DOUBLE_EQ(A.elementAt(1, 0), 4.0);
    EXPECT_DOUBLE_EQ(A.elementAt(1, 1), 5.0);
    EXPECT_DOUBLE_EQ(A.elementAt(1, 2), 6.0);
}

TEST(MatrixElementAt, RowEqualToRowCountThrows) {
    Matrix A(2, 3, {1, 2, 3, 4, 5, 6});

    EXPECT_ANY_THROW(A.elementAt(2, 0));
}

TEST(MatrixElementAt, ColEqualToColCountThrows) {
    Matrix A(2, 3, {1, 2, 3, 4, 5, 6});

    EXPECT_THROW(A.elementAt(0, 3), std::out_of_range);
}

TEST(MatrixElementAt, ZeroRowsByNThrows) {
    Matrix A(0, 3);

    EXPECT_THROW(A.elementAt(0, 0), std::out_of_range);
}

TEST(MatrixElementAt, MbyZeroColsThrows) {
    Matrix A(3, 0);

    EXPECT_THROW(A.elementAt(0, 0), std::out_of_range);
}

TEST(MatrixElementAt, ZeroByZeroThrows) {
    Matrix A(0, 0);

    EXPECT_THROW(A.elementAt(0, 0), std::out_of_range);
}

/* setElements() */

TEST(MatrixSetElements, ReplacesElementsAndKeepsDimensions) {
    Matrix A(2, 3);

    A.setElements({1, 2, 3, 4, 5, 6});

    EXPECT_EQ(A.rows(), 2);
    EXPECT_EQ(A.cols(), 3);
    EXPECT_THAT(A.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{1, 2, 3, 4, 5, 6}));
}

TEST(MatrixSetElements, TooFewElementsThrows) {
    Matrix A(2, 3);

    EXPECT_THROW(A.setElements({1, 2, 3, 4, 5}), std::invalid_argument);
}

TEST(MatrixSetElements, TooManyElementsThrows) {
    Matrix A(2, 3);

    EXPECT_THROW(A.setElements({1, 2, 3, 4, 5, 6, 7}), std::invalid_argument);
}

TEST(MatrixSetElements, EmptyElementsOnZeroRowsByN) {
    Matrix A(0, 3);

    A.setElements({});

    EXPECT_THAT(A.elements(), testing::IsEmpty());
}

TEST(MatrixSetElements, EmptyElementsOnMbyZeroCols) {
    Matrix A(3, 0);

    A.setElements({});

    EXPECT_THAT(A.elements(), testing::IsEmpty());
}

TEST(MatrixSetElements, EmptyElementsOnZeroByZero) {
    Matrix A(0, 0);

    A.setElements({});

    EXPECT_THAT(A.elements(), testing::IsEmpty());
}

TEST(MatrixSetElements, NonEmptyElementsOnEmptyMatrixThrows) {
    Matrix A(0, 0);

    EXPECT_THROW(A.setElements({1}), std::invalid_argument);
}

/* setDimensions() */

TEST(MatrixSetDimensions, ReshapeKeepsElementsInRowMajorOrder) {
    Matrix A(2, 3, {1, 2, 3, 4, 5, 6});

    A.setDimensions(3, 2);

    EXPECT_EQ(A.rows(), 3);
    EXPECT_EQ(A.cols(), 2);
    EXPECT_THAT(A.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{1, 2, 3, 4, 5, 6}));
}

TEST(MatrixSetDimensions, ReshapeToDifferentElementCountThrows) {
    Matrix A(2, 3, {1, 2, 3, 4, 5, 6});

    EXPECT_THROW(A.setDimensions(2, 2), std::invalid_argument);
}

TEST(MatrixSetDimensions, ZeroRowsByNToMbyZeroCols) {
    Matrix A(0, 3);

    A.setDimensions(3, 0);

    EXPECT_EQ(A.rows(), 3);
    EXPECT_EQ(A.cols(), 0);
    EXPECT_THAT(A.elements(), testing::IsEmpty());
}

TEST(MatrixSetDimensions, ZeroByZeroToZeroRowsByN) {
    Matrix A(0, 0);

    A.setDimensions(0, 3);

    EXPECT_EQ(A.rows(), 0);
    EXPECT_EQ(A.cols(), 3);
    EXPECT_THAT(A.elements(), testing::IsEmpty());
}

/* add() / operator+() / operator+=() */

TEST(MatrixAdd, MbyN) {
    Matrix A(2, 3, {1, 2, 3, 4, 5, 6});
    Matrix B(2, 3, {6, 5, 4, 3, 2, 1});

    Matrix C = A.add(B);

    EXPECT_EQ(C.rows(), 2);
    EXPECT_EQ(C.cols(), 3);
    EXPECT_THAT(C.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{7, 7, 7, 7, 7, 7}));
}

TEST(MatrixAdd, NbyN) {
    Matrix A(2, 2, {1, 2, 3, 4});
    Matrix B(2, 2, {5, 6, 7, 8});

    Matrix C = A.add(B);

    EXPECT_THAT(C.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{6, 8, 10, 12}));
}

TEST(MatrixAdd, ZeroRowsByN) {
    Matrix A(0, 3);
    Matrix B(0, 3);

    Matrix C = A.add(B);

    EXPECT_EQ(C.rows(), 0);
    EXPECT_EQ(C.cols(), 3);
    EXPECT_THAT(C.elements(), testing::IsEmpty());
}

TEST(MatrixAdd, MbyZeroCols) {
    Matrix A(3, 0);
    Matrix B(3, 0);

    Matrix C = A.add(B);

    EXPECT_EQ(C.rows(), 3);
    EXPECT_EQ(C.cols(), 0);
    EXPECT_THAT(C.elements(), testing::IsEmpty());
}

TEST(MatrixAdd, ZeroByZero) {
    Matrix A(0, 0);
    Matrix B(0, 0);

    Matrix C = A.add(B);

    EXPECT_EQ(C.rows(), 0);
    EXPECT_EQ(C.cols(), 0);
    EXPECT_THAT(C.elements(), testing::IsEmpty());
}

TEST(MatrixAdd, DifferentRowCountsThrows) {
    Matrix A(2, 3);
    Matrix B(3, 3);

    EXPECT_THROW(A.add(B), std::invalid_argument);
}

TEST(MatrixAdd, DifferentColCountsThrows) {
    Matrix A(2, 3);
    Matrix B(2, 2);

    EXPECT_THROW(A.add(B), std::invalid_argument);
}

TEST(MatrixAdd, DoesNotModifyOperands) {
    Matrix A(2, 2, {1, 2, 3, 4});
    Matrix B(2, 2, {5, 6, 7, 8});

    Matrix C = A.add(B);

    EXPECT_THAT(A.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{1, 2, 3, 4}));
    EXPECT_THAT(B.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{5, 6, 7, 8}));
}

TEST(MatrixOperatorPlus, MbyN) {
    Matrix A(2, 3, {1, 2, 3, 4, 5, 6});
    Matrix B(2, 3, {6, 5, 4, 3, 2, 1});

    Matrix C = A + B;

    EXPECT_EQ(C.rows(), 2);
    EXPECT_EQ(C.cols(), 3);
    EXPECT_THAT(C.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{7, 7, 7, 7, 7, 7}));
}

TEST(MatrixOperatorPlus, ZeroByZero) {
    Matrix A(0, 0);
    Matrix B(0, 0);

    Matrix C = A + B;

    EXPECT_EQ(C.rows(), 0);
    EXPECT_EQ(C.cols(), 0);
    EXPECT_THAT(C.elements(), testing::IsEmpty());
}

TEST(MatrixOperatorPlus, MismatchedDimensionsThrows) {
    Matrix A(2, 3);
    Matrix B(3, 2);

    EXPECT_THROW(A + B, std::invalid_argument);
}

TEST(MatrixOperatorPlusEquals, MbyN) {
    Matrix A(2, 3, {1, 2, 3, 4, 5, 6});
    Matrix B(2, 3, {6, 5, 4, 3, 2, 1});

    A += B;

    EXPECT_EQ(A.rows(), 2);
    EXPECT_EQ(A.cols(), 3);
    EXPECT_THAT(A.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{7, 7, 7, 7, 7, 7}));
}

TEST(MatrixOperatorPlusEquals, ReturnsReferenceToLeftHandSide) {
    Matrix A(2, 2, {1, 2, 3, 4});
    Matrix B(2, 2, {5, 6, 7, 8});

    Matrix &result = (A += B);

    EXPECT_EQ(&result, &A);
}

TEST(MatrixOperatorPlusEquals, AddingMatrixToItself) {
    Matrix A(2, 2, {1, 2, 3, 4});

    A += A;

    EXPECT_THAT(A.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{2, 4, 6, 8}));
}

TEST(MatrixOperatorPlusEquals, ZeroByZero) {
    Matrix A(0, 0);
    Matrix B(0, 0);

    A += B;

    EXPECT_EQ(A.rows(), 0);
    EXPECT_EQ(A.cols(), 0);
    EXPECT_THAT(A.elements(), testing::IsEmpty());
}

TEST(MatrixOperatorPlusEquals, MismatchedDimensionsThrows) {
    Matrix A(2, 3);
    Matrix B(3, 2);

    EXPECT_THROW(A += B, std::invalid_argument);
}

/* subtract() / operator-() / operator-=() */

TEST(MatrixSubtract, MbyN) {
    Matrix A(2, 3, {6, 5, 4, 3, 2, 1});
    Matrix B(2, 3, {1, 2, 3, 4, 5, 6});

    Matrix C = A.subtract(B);

    EXPECT_EQ(C.rows(), 2);
    EXPECT_EQ(C.cols(), 3);
    EXPECT_THAT(C.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{5, 3, 1, -1, -3, -5}));
}

TEST(MatrixSubtract, NbyN) {
    Matrix A(2, 2, {5, 6, 7, 8});
    Matrix B(2, 2, {1, 2, 3, 4});

    Matrix C = A.subtract(B);

    EXPECT_THAT(C.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{4, 4, 4, 4}));
}

TEST(MatrixSubtract, ZeroRowsByN) {
    Matrix A(0, 3);
    Matrix B(0, 3);

    Matrix C = A.subtract(B);

    EXPECT_EQ(C.rows(), 0);
    EXPECT_EQ(C.cols(), 3);
    EXPECT_THAT(C.elements(), testing::IsEmpty());
}

TEST(MatrixSubtract, MbyZeroCols) {
    Matrix A(3, 0);
    Matrix B(3, 0);

    Matrix C = A.subtract(B);

    EXPECT_EQ(C.rows(), 3);
    EXPECT_EQ(C.cols(), 0);
    EXPECT_THAT(C.elements(), testing::IsEmpty());
}

TEST(MatrixSubtract, ZeroByZero) {
    Matrix A(0, 0);
    Matrix B(0, 0);

    Matrix C = A.subtract(B);

    EXPECT_EQ(C.rows(), 0);
    EXPECT_EQ(C.cols(), 0);
    EXPECT_THAT(C.elements(), testing::IsEmpty());
}

TEST(MatrixSubtract, DifferentRowCountsThrows) {
    Matrix A(2, 3);
    Matrix B(3, 3);

    EXPECT_THROW(A.subtract(B), std::invalid_argument);
}

TEST(MatrixSubtract, DifferentColCountsThrows) {
    Matrix A(2, 3);
    Matrix B(2, 2);

    EXPECT_THROW(A.subtract(B), std::invalid_argument);
}

TEST(MatrixSubtract, DoesNotModifyOperands) {
    Matrix A(2, 2, {5, 6, 7, 8});
    Matrix B(2, 2, {1, 2, 3, 4});

    Matrix C = A.subtract(B);

    EXPECT_THAT(A.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{5, 6, 7, 8}));
    EXPECT_THAT(B.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{1, 2, 3, 4}));
}

TEST(MatrixOperatorMinus, MbyN) {
    Matrix A(2, 3, {6, 5, 4, 3, 2, 1});
    Matrix B(2, 3, {1, 2, 3, 4, 5, 6});

    Matrix C = A - B;

    EXPECT_EQ(C.rows(), 2);
    EXPECT_EQ(C.cols(), 3);
    EXPECT_THAT(C.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{5, 3, 1, -1, -3, -5}));
}

TEST(MatrixOperatorMinus, ZeroByZero) {
    Matrix A(0, 0);
    Matrix B(0, 0);

    Matrix C = A - B;

    EXPECT_EQ(C.rows(), 0);
    EXPECT_EQ(C.cols(), 0);
    EXPECT_THAT(C.elements(), testing::IsEmpty());
}

TEST(MatrixOperatorMinus, MismatchedDimensionsThrows) {
    Matrix A(2, 3);
    Matrix B(3, 2);

    EXPECT_THROW((A - B), std::invalid_argument);
}

TEST(MatrixOperatorMinusEquals, MbyN) {
    Matrix A(2, 3, {6, 5, 4, 3, 2, 1});
    Matrix B(2, 3, {1, 2, 3, 4, 5, 6});

    A -= B;

    EXPECT_EQ(A.rows(), 2);
    EXPECT_EQ(A.cols(), 3);
    EXPECT_THAT(A.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{5, 3, 1, -1, -3, -5}));
}

TEST(MatrixOperatorMinusEquals, ReturnsReferenceToLeftHandSide) {
    Matrix A(2, 2, {5, 6, 7, 8});
    Matrix B(2, 2, {1, 2, 3, 4});

    Matrix &result = (A -= B);

    EXPECT_EQ(&result, &A);
}

TEST(MatrixOperatorMinusEquals, SubtractingMatrixFromItself) {
    Matrix A(2, 2, {1, 2, 3, 4});

    A -= A;

    EXPECT_THAT(A.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{0, 0, 0, 0}));
}

TEST(MatrixOperatorMinusEquals, ZeroByZero) {
    Matrix A(0, 0);
    Matrix B(0, 0);

    A -= B;

    EXPECT_EQ(A.rows(), 0);
    EXPECT_EQ(A.cols(), 0);
    EXPECT_THAT(A.elements(), testing::IsEmpty());
}

TEST(MatrixOperatorMinusEquals, MismatchedDimensionsThrows) {
    Matrix A(2, 3);
    Matrix B(3, 2);

    EXPECT_THROW(A -= B, std::invalid_argument);
}

/* Matrix-Matrix: multiply() / operator* / operator*= */

TEST(MatrixMultiplyMatrix, MbyKTimesKbyN) {
    Matrix A(2, 3, {1, 2, 3, 4, 5, 6});
    Matrix B(3, 2, {7, 8, 9, 10, 11, 12});

    Matrix C = A.multiply(B);

    EXPECT_EQ(C.rows(), 2u);
    EXPECT_EQ(C.cols(), 2u);
    EXPECT_THAT(C.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{58, 64, 139, 154}));
}

TEST(MatrixMultiplyMatrix, ResultIsRectangular) {
    Matrix A(3, 2, {7, 8, 9, 10, 11, 12});
    Matrix B(2, 3, {1, 2, 3, 4, 5, 6});

    Matrix C = A.multiply(B);

    EXPECT_EQ(C.rows(), 3);
    EXPECT_EQ(C.cols(), 3);
    EXPECT_THAT(C.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{39, 54, 69, 49, 68, 87, 59, 82, 105}));
}

TEST(MatrixMultiplyMatrix, NbyNTimesNbyN) {
    Matrix A(2, 2, {1, 2, 3, 4});
    Matrix B(2, 2, {5, 6, 7, 8});

    Matrix C = A.multiply(B);

    EXPECT_EQ(C.rows(), 2);
    EXPECT_EQ(C.cols(), 2);
    EXPECT_THAT(C.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{19, 22, 43, 50}));
}

TEST(MatrixMultiplyMatrix, IsNotCommutative) {
    Matrix A(2, 2, {1, 2, 3, 4});
    Matrix B(2, 2, {5, 6, 7, 8});

    Matrix C = B.multiply(A);

    EXPECT_THAT(C.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{23, 34, 31, 46}));
}

TEST(MatrixMultiplyMatrix, IdentityLeavesMatrixUnchanged) {
    Matrix I(2, 2, {1, 0, 0, 1});
    Matrix A(2, 3, {1, 2, 3, 4, 5, 6});

    Matrix C = I.multiply(A);

    EXPECT_THAT(C.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{1, 2, 3, 4, 5, 6}));
}

TEST(MatrixMultiplyMatrix, ZeroRowsByKTimesKbyN) {
    Matrix A(0, 3);
    Matrix B(3, 2, {7, 8, 9, 10, 11, 12});

    Matrix C = A.multiply(B);

    EXPECT_EQ(C.rows(), 0);
    EXPECT_EQ(C.cols(), 2);
    EXPECT_THAT(C.elements(), testing::IsEmpty());
}

TEST(MatrixMultiplyMatrix, MbyKTimesKbyZeroCols) {
    Matrix A(2, 3, {1, 2, 3, 4, 5, 6});
    Matrix B(3, 0);

    Matrix C = A.multiply(B);

    EXPECT_EQ(C.rows(), 2);
    EXPECT_EQ(C.cols(), 0);
    EXPECT_THAT(C.elements(), testing::IsEmpty());
}

TEST(MatrixMultiplyMatrix, MbyZeroTimesZeroByNIsZeroFilled) {
    Matrix A(2, 0);
    Matrix B(0, 3);

    Matrix C = A.multiply(B);

    EXPECT_EQ(C.rows(), 2);
    EXPECT_EQ(C.cols(), 3);
    EXPECT_THAT(C.elements(), testing::AllOf(
        testing::SizeIs(6), testing::Each(testing::DoubleEq(0.0))));
}

TEST(MatrixMultiplyMatrix, ZeroByZeroTimesZeroByZero) {
    Matrix A(0, 0);
    Matrix B(0, 0);

    Matrix C = A.multiply(B);

    EXPECT_EQ(C.rows(), 0);
    EXPECT_EQ(C.cols(), 0);
    EXPECT_THAT(C.elements(), testing::IsEmpty());
}

TEST(MatrixMultiplyMatrix, InnerDimensionMismatchThrows) {
    Matrix A(2, 3);
    Matrix B(2, 3);

    EXPECT_THROW(A.multiply(B), std::invalid_argument);
}

TEST(MatrixMultiplyMatrix, EmptyInnerDimensionMismatchThrows) {
    Matrix A(2, 0);
    Matrix B(3, 2);

    EXPECT_THROW(A.multiply(B), std::invalid_argument);
}

TEST(MatrixMultiplyMatrix, DoesNotModifyOperands) {
    Matrix A(2, 2, {1, 2, 3, 4});
    Matrix B(2, 2, {5, 6, 7, 8});

    Matrix C = A.multiply(B);

    EXPECT_THAT(A.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{1, 2, 3, 4}));
    EXPECT_THAT(B.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{5, 6, 7, 8}));
}

TEST(MatrixOperatorTimesMatrix, MbyKTimesKbyN) {
    Matrix A(2, 3, {1, 2, 3, 4, 5, 6});
    Matrix B(3, 2, {7, 8, 9, 10, 11, 12});

    Matrix C = A * B;

    EXPECT_EQ(C.rows(), 2);
    EXPECT_EQ(C.cols(), 2);
    EXPECT_THAT(C.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{58, 64, 139, 154}));
}

TEST(MatrixOperatorTimesMatrix, MbyZeroTimesZeroByNIsZeroFilled) {
    Matrix A(2, 0);
    Matrix B(0, 3);

    Matrix C = A * B;

    EXPECT_EQ(C.rows(), 2);
    EXPECT_EQ(C.cols(), 3);
    EXPECT_THAT(C.elements(), testing::AllOf(
        testing::SizeIs(6), testing::Each(testing::DoubleEq(0.0))));
}

TEST(MatrixOperatorTimesMatrix, InnerDimensionMismatchThrows) {
    Matrix A(2, 3);
    Matrix B(2, 3);

    EXPECT_THROW(A * B, std::invalid_argument);
}

TEST(MatrixOperatorTimesEqualsMatrix, MbyKTimesKbyNChangesDimensions) {
    Matrix A(2, 3, {1, 2, 3, 4, 5, 6});
    Matrix B(3, 2, {7, 8, 9, 10, 11, 12});

    A *= B;

    EXPECT_EQ(A.rows(), 2);
    EXPECT_EQ(A.cols(), 2);
    EXPECT_THAT(A.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{58, 64, 139, 154}));
}

TEST(MatrixOperatorTimesEqualsMatrix, ReturnsReferenceToLeftHandSide) {
    Matrix A(2, 2, {1, 2, 3, 4});
    Matrix B(2, 2, {5, 6, 7, 8});

    Matrix &result = (A *= B);

    EXPECT_EQ(&result, &A);
}

TEST(MatrixOperatorTimesEqualsMatrix, MultiplyingMatrixByItself) {
    Matrix A(2, 2, {1, 2, 3, 4});

    A *= A;

    EXPECT_THAT(A.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{7, 10, 15, 22}));
}

TEST(MatrixOperatorTimesEqualsMatrix, ZeroByZero) {
    Matrix A(0, 0);
    Matrix B(0, 0);

    A *= B;

    EXPECT_EQ(A.rows(), 0);
    EXPECT_EQ(A.cols(), 0);
    EXPECT_THAT(A.elements(), testing::IsEmpty());
}

TEST(MatrixOperatorTimesEqualsMatrix, InnerDimensionMismatchThrows) {
    Matrix A(2, 3);
    Matrix B(2, 3);

    EXPECT_THROW(A *= B, std::invalid_argument);
}

/* Matrix-Vector: multiply() / operator*() / operator*=() */

TEST(MatrixMultiplyVector, MbyNTimesVector) {
    Matrix A(2, 3, {1, 2, 3, 4, 5, 6});
    std::vector<double> x = {1, 2, 3};

    std::vector<double> y = A.multiply(x);

    EXPECT_THAT(y, testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{14, 32}));
}

TEST(MatrixMultiplyVector, NbyNTimesVector) {
    Matrix A(2, 2, {1, 2, 3, 4});
    std::vector<double> x = {5, 6};

    std::vector<double> y = A.multiply(x);

    EXPECT_THAT(y, testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{17, 39}));
}

TEST(MatrixMultiplyVector, ZeroRowsByNTimesVectorReturnsEmptyVector) {
    Matrix A(0, 3);
    std::vector<double> x = {1, 2, 3};

    std::vector<double> y = A.multiply(x);

    EXPECT_THAT(y, testing::IsEmpty());
}

TEST(MatrixMultiplyVector, MbyZeroColsTimesEmptyVectorReturnsZeroFilledVector) {
    Matrix A(2, 0);
    std::vector<double> x;

    std::vector<double> y = A.multiply(x);

    EXPECT_THAT(y, testing::AllOf(
        testing::SizeIs(2), testing::Each(testing::DoubleEq(0.0))));
}

TEST(MatrixMultiplyVector, ZeroByZeroTimesEmptyVectorReturnsEmptyVector) {
    Matrix A(0, 0);
    std::vector<double> x;

    std::vector<double> y = A.multiply(x);

    EXPECT_THAT(y, testing::IsEmpty());
}

TEST(MatrixMultiplyVector, VectorTooShortThrows) {
    Matrix A(2, 3);
    std::vector<double> x = {1, 2};

    EXPECT_THROW(A.multiply(x), std::invalid_argument);
}

TEST(MatrixMultiplyVector, VectorTooLongThrows) {
    Matrix A(2, 3);
    std::vector<double> x = {1, 2, 3, 4};

    EXPECT_THROW(A.multiply(x), std::invalid_argument);
}

TEST(MatrixMultiplyVector, NonEmptyVectorWithMbyZeroColsThrows) {
    Matrix A(2, 0);
    std::vector<double> x = {1};

    EXPECT_THROW(A.multiply(x), std::invalid_argument);
}

TEST(MatrixOperatorTimesVector, MbyNTimesVector) {
    Matrix A(2, 3, {1, 2, 3, 4, 5, 6});
    std::vector<double> x = {1, 2, 3};

    std::vector<double> y = A * x;

    EXPECT_THAT(y, testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{14, 32}));
}

TEST(MatrixOperatorTimesVector, MbyZeroColsTimesEmptyVectorReturnsZeroFilledVector) {
    Matrix A(2, 0);
    std::vector<double> x;

    std::vector<double> y = A * x;

    EXPECT_THAT(y, testing::AllOf(
        testing::SizeIs(2), testing::Each(testing::DoubleEq(0.0))));
}

TEST(MatrixOperatorTimesVector, SizeMismatchThrows) {
    Matrix A(2, 3);
    std::vector<double> x = {1, 2};

    EXPECT_THROW(A * x, std::invalid_argument);
}

TEST(VectorTimesMatrix, VectorTimesMbyN) {
    std::vector<double> x = {1, 2};
    Matrix A(2, 3, {1, 2, 3, 4, 5, 6});

    std::vector<double> y = x * A;

    EXPECT_THAT(y, testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{9, 12, 15}));
}

TEST(VectorTimesMatrix, VectorTimesNbyN) {
    std::vector<double> x = {5, 6};
    Matrix A(2, 2, {1, 2, 3, 4});

    std::vector<double> y = x * A;

    EXPECT_THAT(y, testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{23, 34}));
}

TEST(VectorTimesMatrix, EmptyVectorTimesZeroRowsByNReturnsZeroFilledVector) {
    std::vector<double> x;
    Matrix A(0, 3);

    std::vector<double> y = x * A;

    EXPECT_THAT(y, testing::AllOf(
        testing::SizeIs(3), testing::Each(testing::DoubleEq(0.0))));
}

TEST(VectorTimesMatrix, VectorTimesMbyZeroColsReturnsEmptyVector) {
    std::vector<double> x = {1, 2, 3};
    Matrix A(3, 0);

    std::vector<double> y = x * A;

    EXPECT_THAT(y, testing::IsEmpty());
}

TEST(VectorTimesMatrix, EmptyVectorTimesZeroByZeroReturnsEmptyVector) {
    std::vector<double> x;
    Matrix A(0, 0);

    std::vector<double> y = x * A;

    EXPECT_THAT(y, testing::IsEmpty());
}

TEST(VectorTimesMatrix, SizeMismatchThrows) {
    std::vector<double> x = {1, 2, 3};
    Matrix A(2, 3);

    EXPECT_THROW(x * A, std::invalid_argument);
}

/* Matrix-Scalar: operator*(), operator*=(), scale() */

TEST(MatrixOperatorTimesScalar, MbyN) {
    Matrix A(2, 3, {1, 2, 3, 4, 5, 6});

    Matrix B = A * 2.0;

    EXPECT_EQ(B.rows(), 2);
    EXPECT_EQ(B.cols(), 3);
    EXPECT_THAT(B.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{2, 4, 6, 8, 10, 12}));
}

TEST(MatrixOperatorTimesScalar, NegativeScalar) {
    Matrix A(2, 2, {1, -2, 3, -4});

    Matrix B = A * -2.0;

    EXPECT_THAT(B.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{-2, 4, -6, 8}));
}

TEST(MatrixOperatorTimesScalar, ZeroScalarGivesZeroMatrix) {
    Matrix A(2, 2, {1, 2, 3, 4});

    Matrix B = A * 0.0;

    EXPECT_THAT(B.elements(), testing::AllOf(
        testing::SizeIs(4), testing::Each(testing::DoubleEq(0.0))));
}

TEST(MatrixOperatorTimesScalar, ScalarOfOneLeavesMatrixUnchanged) {
    Matrix A(2, 2, {1, 2, 3, 4});

    Matrix B = A * 1.0;

    EXPECT_THAT(B.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{1, 2, 3, 4}));
}

TEST(MatrixOperatorTimesScalar, ZeroRowsByN) {
    Matrix A(0, 3);

    Matrix B = A * 2.0;

    EXPECT_EQ(B.rows(), 0);
    EXPECT_EQ(B.cols(), 3);
    EXPECT_THAT(B.elements(), testing::IsEmpty());
}

TEST(MatrixOperatorTimesScalar, MbyZeroCols) {
    Matrix A(3, 0);

    Matrix B = A * 2.0;

    EXPECT_EQ(B.rows(), 3);
    EXPECT_EQ(B.cols(), 0);
    EXPECT_THAT(B.elements(), testing::IsEmpty());
}

TEST(MatrixOperatorTimesScalar, ZeroByZero) {
    Matrix A(0, 0);

    Matrix B = A * 2.0;

    EXPECT_EQ(B.rows(), 0);
    EXPECT_EQ(B.cols(), 0);
    EXPECT_THAT(B.elements(), testing::IsEmpty());
}

TEST(MatrixOperatorTimesScalar, DoesNotModifyOriginal) {
    Matrix A(2, 2, {1, 2, 3, 4});

    Matrix B = A * 2.0;

    EXPECT_THAT(A.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{1, 2, 3, 4}));
}

TEST(ScalarTimesMatrix, MbyN) {
    Matrix A(2, 3, {1, 2, 3, 4, 5, 6});

    Matrix B = 2.0 * A;

    EXPECT_EQ(B.rows(), 2);
    EXPECT_EQ(B.cols(), 3);
    EXPECT_THAT(B.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{2, 4, 6, 8, 10, 12}));
}

TEST(ScalarTimesMatrix, NegativeScalar) {
    Matrix A(2, 2, {1, -2, 3, -4});

    Matrix B = -2.0 * A;

    EXPECT_THAT(B.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{-2, 4, -6, 8}));
}

TEST(ScalarTimesMatrix, ZeroRowsByN) {
    Matrix A(0, 3);

    Matrix B = 2.0 * A;

    EXPECT_EQ(B.rows(), 0);
    EXPECT_EQ(B.cols(), 3);
    EXPECT_THAT(B.elements(), testing::IsEmpty());
}

TEST(ScalarTimesMatrix, MbyZeroCols) {
    Matrix A(3, 0);

    Matrix B = 2.0 * A;

    EXPECT_EQ(B.rows(), 3);
    EXPECT_EQ(B.cols(), 0);
    EXPECT_THAT(B.elements(), testing::IsEmpty());
}

TEST(ScalarTimesMatrix, ZeroByZero) {
    Matrix A(0, 0);

    Matrix B = 2.0 * A;

    EXPECT_EQ(B.rows(), 0);
    EXPECT_EQ(B.cols(), 0);
    EXPECT_THAT(B.elements(), testing::IsEmpty());
}

TEST(MatrixOperatorTimesEqualsScalar, MbyN) {
    Matrix A(2, 3, {1, 2, 3, 4, 5, 6});

    A *= 2.0;

    EXPECT_EQ(A.rows(), 2u);
    EXPECT_EQ(A.cols(), 3u);
    EXPECT_THAT(A.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{2, 4, 6, 8, 10, 12}));
}

TEST(MatrixOperatorTimesEqualsScalar, ReturnsReferenceToLeftHandSide) {
    Matrix A(2, 2, {1, 2, 3, 4});

    Matrix &result = (A *= 2.0);

    EXPECT_EQ(&result, &A);
}

TEST(MatrixOperatorTimesEqualsScalar, NegativeScalar) {
    Matrix A(2, 2, {1, -2, 3, -4});

    A *= -2.0;

    EXPECT_THAT(A.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{-2, 4, -6, 8}));
}

TEST(MatrixOperatorTimesEqualsScalar, ZeroScalarGivesZeroMatrix) {
    Matrix A(2, 2, {1, 2, 3, 4});

    A *= 0.0;

    EXPECT_THAT(A.elements(), testing::AllOf(
        testing::SizeIs(4), testing::Each(testing::DoubleEq(0.0))));
}

TEST(MatrixOperatorTimesEqualsScalar, ZeroRowsByN) {
    Matrix A(0, 3);

    A *= 2.0;

    EXPECT_EQ(A.rows(), 0);
    EXPECT_EQ(A.cols(), 3);
    EXPECT_THAT(A.elements(), testing::IsEmpty());
}

TEST(MatrixOperatorTimesEqualsScalar, MbyZeroCols) {
    Matrix A(3, 0);

    A *= 2.0;

    EXPECT_EQ(A.rows(), 3);
    EXPECT_EQ(A.cols(), 0);
    EXPECT_THAT(A.elements(), testing::IsEmpty());
}

TEST(MatrixOperatorTimesEqualsScalar, ZeroByZero) {
    Matrix A(0, 0);

    A *= 2.0;

    EXPECT_EQ(A.rows(), 0);
    EXPECT_EQ(A.cols(), 0);
    EXPECT_THAT(A.elements(), testing::IsEmpty());
}

TEST(MatrixScale, MbyN) {
    Matrix A(2, 3, {1, 2, 3, 4, 5, 6});

    A.scale(2.0);

    EXPECT_EQ(A.rows(), 2);
    EXPECT_EQ(A.cols(), 3);
    EXPECT_THAT(A.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{2, 4, 6, 8, 10, 12}));
}

TEST(MatrixScale, NegativeScalar) {
    Matrix A(2, 2, {1, -2, 3, -4});

    A.scale(-2.0);

    EXPECT_THAT(A.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{-2, 4, -6, 8}));
}

TEST(MatrixScale, ZeroScalarGivesZeroMatrix) {
    Matrix A(2, 2, {1, 2, 3, 4});

    A.scale(0.0);

    EXPECT_THAT(A.elements(), testing::AllOf(
        testing::SizeIs(4), testing::Each(testing::DoubleEq(0.0))));
}

TEST(MatrixScale, ZeroRowsByN) {
    Matrix A(0, 3);

    A.scale(2.0);

    EXPECT_EQ(A.rows(), 0);
    EXPECT_EQ(A.cols(), 3);
    EXPECT_THAT(A.elements(), testing::IsEmpty());
}

TEST(MatrixScale, MbyZeroCols) {
    Matrix A(3, 0);

    A.scale(2.0);

    EXPECT_EQ(A.rows(), 3);
    EXPECT_EQ(A.cols(), 0);
    EXPECT_THAT(A.elements(), testing::IsEmpty());
}

TEST(MatrixScale, ZeroByZero) {
    Matrix A(0, 0);

    A.scale(2.0);

    EXPECT_EQ(A.rows(), 0);
    EXPECT_EQ(A.cols(), 0);
    EXPECT_THAT(A.elements(), testing::IsEmpty());
}

/* transpose() */

TEST(MatrixTranspose, MbyN) {
    Matrix A(2, 3, {1, 2, 3, 4, 5, 6});

    Matrix T = A.transpose();

    EXPECT_EQ(T.rows(), 3);
    EXPECT_EQ(T.cols(), 2);
    EXPECT_THAT(T.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{1, 4, 2, 5, 3, 6}));
}

TEST(MatrixTranspose, NbyN) {
    Matrix A(2, 2, {1, 2, 3, 4});

    Matrix T = A.transpose();

    EXPECT_EQ(T.rows(), 2);
    EXPECT_EQ(T.cols(), 2);
    EXPECT_THAT(T.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{1, 3, 2, 4}));
}

TEST(MatrixTranspose, SymmetricMatrixIsUnchanged) {
    Matrix A(2, 2, {1, 2, 2, 1});

    Matrix T = A.transpose();

    EXPECT_THAT(T.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{1, 2, 2, 1}));
}

TEST(MatrixTranspose, TransposingTwiceReturnsOriginal) {
    Matrix A(2, 3, {1, 2, 3, 4, 5, 6});

    Matrix T = A.transpose().transpose();

    EXPECT_EQ(T.rows(), 2);
    EXPECT_EQ(T.cols(), 3);
    EXPECT_THAT(T.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{1, 2, 3, 4, 5, 6}));
}

TEST(MatrixTranspose, ZeroRowsByNBecomesNbyZeroCols) {
    Matrix A(0, 3);

    Matrix T = A.transpose();

    EXPECT_EQ(T.rows(), 3);
    EXPECT_EQ(T.cols(), 0);
    EXPECT_THAT(T.elements(), testing::IsEmpty());
}

TEST(MatrixTranspose, MbyZeroColsBecomesZeroRowsByM) {
    Matrix A(3, 0);

    Matrix T = A.transpose();

    EXPECT_EQ(T.rows(), 0);
    EXPECT_EQ(T.cols(), 3);
    EXPECT_THAT(T.elements(), testing::IsEmpty());
}

TEST(MatrixTranspose, ZeroByZero) {
    Matrix A(0, 0);

    Matrix T = A.transpose();

    EXPECT_EQ(T.rows(), 0);
    EXPECT_EQ(T.cols(), 0);
    EXPECT_THAT(T.elements(), testing::IsEmpty());
}

TEST(MatrixTranspose, DoesNotModifyOriginal) {
    Matrix A(2, 3, {1, 2, 3, 4, 5, 6});

    Matrix T = A.transpose();

    EXPECT_EQ(A.rows(), 2);
    EXPECT_EQ(A.cols(), 3);
    EXPECT_THAT(A.elements(), testing::Pointwise(
        testing::DoubleEq(), std::vector<double>{1, 2, 3, 4, 5, 6}));
}

/* operator<< */

TEST(MatrixOutputStream, ContainsEveryElement) {
    Matrix A(2, 2, {11, 22, 33, 44});
    std::ostringstream os;

    os << A;

    EXPECT_THAT(os.str(), testing::AllOf(
        testing::HasSubstr("11"), testing::HasSubstr("22"),
        testing::HasSubstr("33"), testing::HasSubstr("44")));
}

TEST(MatrixOutputStream, ReturnsTheStreamForChaining) {
    Matrix A(2, 2, {1, 2, 3, 4});
    std::ostringstream os;

    std::ostream &result = (os << A);

    EXPECT_EQ(&result, &os);
}

TEST(MatrixOutputStream, ZeroByZeroDoesNotThrow) {
    Matrix A(0, 0);
    std::ostringstream os;

    EXPECT_NO_THROW(os << A);
}

TEST(MatrixOutputStream, ZeroRowsByNDoesNotThrow) {
    Matrix A(0, 3);
    std::ostringstream os;

    EXPECT_NO_THROW(os << A);
}

TEST(MatrixOutputStream, MbyZeroColsDoesNotThrow) {
    Matrix A(3, 0);
    std::ostringstream os;

    EXPECT_NO_THROW(os << A);
}