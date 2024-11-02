package org.example;

import java.util.Arrays;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.io.IOException;

import static java.lang.Math.abs;

public final class Main {
    public static void main(String[] args) throws IOException {
        Input input = new Input(System.in, System.out);
        double[] C = input.inputObjectiveFunction();
        double[][] A = input.inputConstrainFunction();
        double[] B = input.inputRightHandSide();
        InteriorPoint interiorPointClass = new InteriorPoint(C, A);
//        double[] X = Matrix.solveNonNegativeLeastSquares(A,B);
        double[] X = input.inputInitialValues();
        interiorPointClass.interiorPoint(X, 0);

//        interiorPointClass.interiorPoint(X, 0);

//        Algorithm algorithm = new Algorithm(c, a, b);
//        SimplexResult result = algorithm.simplex(a, c, b);
//        Output output = new Output(result, System.out);
//        output.printResult();
    }
}


class InteriorPoint {
    double a = 0.5;

    double[][] A;

    double[] C;

    public InteriorPoint(double[] C, double[][] A) {
        this.A = A;
        this.C = C;
    }


    public void interiorPoint(double[] X, double normX) {
        int count = 0;
        double[] result = new double[X.length];
        while(true) {
            count++;
            double[] Xprev = X;
            normX = Matrix.norm(X);
            double[][] D = Matrix.createDiagonal(X);
            double[][] ATilde = Matrix.multiplyMatrix(A, D);
            double[] Ctilde = Matrix.multiplyMatrix(D, C);
            double[][] ATildeTranspose = Matrix.Transpose(ATilde);
            double[][] ATilxATT = Matrix.multiplyMatrix(ATilde, ATildeTranspose);
            double[][] inverseMatrix = Matrix.inverseMatrix(Matrix.multiplyMatrix(ATilde, ATildeTranspose));
            double[][] calculateA = Matrix.multiplyMatrix(ATildeTranspose, Matrix.multiplyMatrix(inverseMatrix, ATilde));
            double[][] P = Matrix.subtractFromIdentity(calculateA);
            double[] Cp = Matrix.multiplyMatrix(P, Ctilde);
            double[] Xunit = new double[X.length];
            Arrays.fill(Xunit, 1);
            double[] XTilde = Matrix.sumMatrix(Xunit, Matrix.multiplyOnConstant(Cp, a / findV(Cp)));
            X = Matrix.multiplyMatrix(D, XTilde);
            result = X;
            if (Matrix.findP2Norm(X, Xprev) < 0.1 || count>5) {
                break;
            }

        }
        double value = 0;
        for (int i = 0; i < result.length; i++) {
            value += C[i] * result[i];
        }
        System.out.println("Result: " + Arrays.toString(result));
        System.out.print("Approximate value: " + value);
    }

    public double findV(double[] Cp) {
        double v = 1;
        for (double value : Cp) {
            if (value < 0) {
                v = Math.min(v, value);
            }
        }
        return v * (-1);
    }
}


class Input {
    private double[] c;
    private BufferedReader reader;
    private PrintStream out;

    public Input(InputStream in, PrintStream out) {
        this.reader = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8));
        this.out = out;
    }

    public double[] inputObjectiveFunction() throws IOException {
        out.print("All vector input in equation form");
        out.print("A vector of coefficients of objective function - C: ");
        String[] coeffStrings = reader.readLine().split(" ");
        double[] c = new double[coeffStrings.length];
        for (int i = 0; i < coeffStrings.length; i++) {
            c[i] = Double.parseDouble(coeffStrings[i]);
        }
        this.c = c;
        return c;
    }
    public double[] inputInitialValues() throws IOException {
        out.print("Input coordinates ");
        out.print("A vector of coefficients - X: ");
        String[] coeffStrings = reader.readLine().split(" ");
        double[] c = new double[coeffStrings.length];
        for (int i = 0; i < coeffStrings.length; i++) {
            c[i] = Double.parseDouble(coeffStrings[i]);
        }
        this.c = c;
        return c;
    }

    public double[][] inputConstrainFunction() throws IOException {
        out.print("Amount of rows in a matrix of coefficients of constrain function - A: ");
        int m = Integer.parseInt(reader.readLine());
        double[][] a = new double[m][this.c.length];
        for (int i = 0; i < m; i++) {
            out.print("Enter coefficients for constraint " + (i + 1) + ": ");
            String[] termStrings = reader.readLine().split(" ");
            for (int j = 0; j < termStrings.length; j++) {
                a[i][j] = Double.parseDouble(termStrings[j]);
            }
        }
        return a;
    }


    public double[] inputRightHandSide() throws IOException {
        out.print("A vector of right-hand side numbers - b: ");
        String[] sumStrings = reader.readLine().split(" ");
        double[] b = new double[sumStrings.length];
        for (int i = 0; i < sumStrings.length; i++) {
            b[i] = Double.parseDouble(sumStrings[i]);
        }
        return b;
    }
}

class Matrix {

    public static double[][] multiplyMatrix(double[][] A, double[][] D) {
        double[][] result = new double[A.length][D[0].length];
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < D[0].length; j++) {
                for (int k = 0; k < A[0].length; k++) {
                    result[i][j] += A[i][k] * D[k][j];
                }
            }
        }
        return result;
    }

    public static double findP2Norm(double[] A, double[] B) {
        double res = 0.0;
        for (int i = 0; i < A.length; i++) {
            res += Math.pow(A[i] - B[i], 2);
        }
        return Math.sqrt(res);
    }
//переделать

    public static double[] solveNonNegativeLeastSquares(double[][] A, double[] B) {
        int n = A[0].length;
        double[] x = new double[n];  // Initialize x with zeros
        double learningRate = 0.001;
        int maxIterations = 10000;

        for (int iter = 0; iter < maxIterations; iter++) {
            double[] gradient = computeGradient(A, B, x);

            // Update step with projection to ensure non-negativity
            for (int i = 0; i < n; i++) {
                x[i] = x[i] - learningRate * gradient[i];
                if (x[i] < 0) x[i] = 0;  // Projection step
            }

            // Check for convergence (small gradient)
            if (norm(gradient) < 1e-6) break;
        }

        return x;
    }

    public static double[] sumMatrix(double[] A, double[] B) {
        double[] result = new double[A.length];
        for (int i = 0; i < A.length; i++) {
            result[i] = A[i] + B[i];
        }
        return result;
    }

    public static double[] computeGradient(double[][] A, double[] B, double[] x) {
        int m = A.length;
        int n = A[0].length;
        double[] gradient = new double[n];
        double[] Ax = new double[m];

        // Compute Ax
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                Ax[i] += A[i][j] * x[j];
            }
        }

        // Compute the gradient as A^T * (Ax - B)
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                gradient[j] += A[i][j] * (Ax[i] - B[i]);
            }
        }

        return gradient;
    }


    public static double[] multiplyMatrix(double[][] A, double[] D) {
        double[] result = new double[A.length];
        for (int i = 0; i < A.length; i++) {
            for (int k = 0; k < A[0].length; k++) {
                result[i] += A[i][k] * D[k];
            }
        }
        return result;
    }

    public static double[][] Transpose(double[][] A) {
        double[][] result = new double[A[0].length][A.length];
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A[0].length; j++) {
                result[j][i] = A[i][j];
            }
        }
        return result;
    }


    public static double[][] inverseMatrix(double[][] A) {
        double determinate = determinate(A);
        int size = A.length;
        double[][] minors = minors(A);

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                minors[i][j] = minors[i][j] / determinate;
            }
        }
        return minors;
    }

    public static double determinate(double[][] A) {
        int size = A.length;
        if (size == 1) {
            return A[0][0];
        }

        if (size == 2) {
            return (int) (A[0][0] * A[1][1] - A[0][1] * A[1][0]);
        } else {
            double det = 0;
            for (int i = 0; i < size; i++) {
                double[][] subMatrix = new double[size - 1][size - 1];
                for (int j = 1; j < size; j++) {
                    for (int k = 0; k < size; k++) {
                        if (k != i) {
                            subMatrix[j - 1][k - (k > i ? 1 : 0)] = A[j][k];
                        }
                    }
                }
                det += Math.pow(-1, i) * A[0][i] * determinate(subMatrix);
            }
            return det;
        }
    }

    public static double[][] createDiagonal(double[] X) {
        int size = X.length;
        double[][] D = new double[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (i == j) {
                    D[i][j] = X[i];
                } else {
                    D[i][j] = 0;
                }
            }
        }
        return D;
    }

    public static double[][] minors(double[][] A) {
        double[][] result = new double[A.length][A.length];
        int size = A.length;

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                double[][] subMatrix = new double[size - 1][size - 1];
                for (int k = 0; k < size; k++) {
                    for (int l = 0; l < size; l++) {
                        if (k != i && l != j) {
                            subMatrix[k - (k > i ? 1 : 0)][l - (l > j ? 1 : 0)] = A[k][l];
                        }
                    }
                }
                result[i][j] = Math.pow(-1, i + j) * determinate(subMatrix);
            }
        }
        return Transpose(result);
    }

    public static double[][] createIdentityMatrix(int size) {
        double[][] identityMatrix = new double[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (i == j) {
                    identityMatrix[i][j] = 1;
                } else {
                    identityMatrix[i][j] = 0;
                }
            }
        }
        return identityMatrix;
    }

    public static double[] multiplyOnConstant(double[] A, double constant) {
        double[] result = new double[A.length];
        for (int i = 0; i < A.length; i++) {
            result[i] = A[i] * constant;
        }
        return result;
    }

    public static double[][] subtractMatrix(double[][] B, double[][] A) {
        double[][] result = new double[A.length][A[0].length];
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A[0].length; j++) {
                result[i][j] = B[i][j] - A[i][j];
            }
        }
        return result;
    }

    public static double norm(double[] A) {
        double res = 0.0;
        double maxAbsValue = 0.0;

        for (double v : A) {
            maxAbsValue = Math.max(maxAbsValue, Math.abs(v));
        }

        if (maxAbsValue > 0) {
            for (double v : A) {
                double normalizedValue = v / maxAbsValue;
                res += normalizedValue * normalizedValue;
            }
            res = maxAbsValue * Math.sqrt(res);
        }

        return res;
    }


    public static double[][] subtractFromIdentity(double[][] A) {
        double[][] result = new double[A.length][A[0].length];
        double[][] Identity = createIdentityMatrix(A.length);
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A[0].length; j++) {
                result[i][j] = Identity[i][j] - A[i][j];
            }
        }
        return result;
    }

}