/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// CHECKSTYLE: stop all
package org.apache.commons.math4.legacy.optim.nonlinear.scalar.noderiv;

import org.apache.commons.math4.legacy.exception.MathIllegalStateException;
import org.apache.commons.math4.legacy.exception.NumberIsTooSmallException;
import org.apache.commons.math4.legacy.exception.OutOfRangeException;
import org.apache.commons.math4.legacy.exception.util.LocalizedFormats;
import org.apache.commons.math4.legacy.linear.Array2DRowRealMatrix;
import org.apache.commons.math4.legacy.linear.ArrayRealVector;
import org.apache.commons.math4.legacy.linear.RealMatrix;
import org.apache.commons.math4.legacy.linear.RealVector;
import org.apache.commons.math4.legacy.optim.PointValuePair;
import org.apache.commons.math4.legacy.optim.nonlinear.scalar.GoalType;
import org.apache.commons.math4.legacy.optim.nonlinear.scalar.MultivariateOptimizer;
import org.apache.commons.math4.core.jdkmath.JdkMath;

/**
 * Powell's BOBYQA algorithm. This implementation is translated and
 * adapted from the Fortran version available
 * <a href="http://plato.asu.edu/ftp/other_software/bobyqa.zip">here</a>.
 * See <a href="http://www.optimization-online.org/DB_HTML/2010/05/2616.html">
 * this paper</a> for an introduction.
 * <br>
 * BOBYQA is particularly well suited for high dimensional problems
 * where derivatives are not available. In most cases it outperforms the
 * {@link PowellOptimizer} significantly. Stochastic algorithms like
 * {@link CMAESOptimizer} succeed more often than BOBYQA, but are more
 * expensive. BOBYQA could also be considered as a replacement of any
 * derivative-based optimizer when the derivatives are approximated by
 * finite differences.
 *
 * @since 3.0
 */
public class BOBYQAOptimizer
    extends MultivariateOptimizer {
    /** Minimum dimension of the problem: {@value}. */
    public static final int MINIMUM_PROBLEM_DIMENSION = 2;
    /** Default value for {@link #initialTrustRegionRadius}: {@value}. */
    public static final double DEFAULT_INITIAL_RADIUS = 10.0;
    /** Default value for {@link #stoppingTrustRegionRadius}: {@value}. */
    public static final double DEFAULT_STOPPING_RADIUS = 1E-8;
    /** Constant 0. */
    private static final double ZERO = 0d;
    /** Constant 1. */
    private static final double ONE = 1d;
    /** Constant 2. */
    private static final double TWO = 2d;
    /** Constant 10. */
    private static final double TEN = 10d;
    /** Constant 16. */
    private static final double SIXTEEN = 16d;
    /** Constant 250. */
    private static final double TWO_HUNDRED_FIFTY = 250d;
    /** Constant -1. */
    private static final double MINUS_ONE = -ONE;
    /** Constant 1/2. */
    private static final double HALF = ONE / 2;
    /** Constant 1/4. */
    private static final double ONE_OVER_FOUR = ONE / 4;
    /** Constant 1/8. */
    private static final double ONE_OVER_EIGHT = ONE / 8;
    /** Constant 1/10. */
    private static final double ONE_OVER_TEN = ONE / 10;
    /** Constant 1/1000. */
    private static final double ONE_OVER_A_THOUSAND = ONE / 1000;

    /**
     * numberOfInterpolationPoints XXX.
     */
    private final int numberOfInterpolationPoints;
    /**
     * initialTrustRegionRadius XXX.
     */
    private double initialTrustRegionRadius;
    /**
     * stoppingTrustRegionRadius XXX.
     */
    private final double stoppingTrustRegionRadius;
    /**
     * Current best values for the variables to be optimized.
     * The vector will be changed in-place to contain the values of the least
     * calculated objective function values.
     */
    private ArrayRealVector currentBest;
    /**
     * Index of the interpolation point at the trust region center.
     */
    private int trustRegionCenterInterpolationPointIndex;
    /**
     * Last <em>n</em> columns of matrix H (where <em>n</em> is the dimension
     * of the problem).
     * The first m rows of bMatrix are ΞT without its first column,
     * the last n rows of bMatrix are Υ without its first row and column,
     * the submatrices ΞT and Υ being taken from expression (2.7)
     * TODO CYRIL split in 2 matrices
     */
    private Array2DRowRealMatrix bMatrix;
    /**
     * Factorization of the leading <em>npt</em> square submatrix of H, this
     * factorization being Z Z<sup>T</sup>, which provides both the correct
     * rank and positive semi-definiteness.
     * XXX "zmat" in the original code.
     */
    private Array2DRowRealMatrix zMatrix;
    /**
     * Coordinates of the interpolation points relative to {@link #originShift}.
     */
    private Array2DRowRealMatrix interpolationPoints;
    /**
     * Shift of origin that should reduce the contributions from rounding
     * errors to values of the model and Lagrange functions.
     * XXX "xbase" in the original code.
     */
    private ArrayRealVector originShift;
    /**
     * Values of the objective function at the interpolation points.
     * XXX "fval" in the original code.
     */
    private ArrayRealVector fAtInterpolationPoints;
    /**
     * Displacement from {@link #originShift} of the trust region center.
     * XXX "xopt" in the original code.
     */
    private ArrayRealVector trustRegionCenterOffset;
    /**
     * Gradient of the quadratic model at {@link #originShift} +
     * {@link #trustRegionCenterOffset}.
     */
    private ArrayRealVector gradientAtTrustRegionCenter;
    /**
     * Differences {@link #getLowerBound()} - {@link #originShift}.
     * All the components of every {@link #trustRegionCenterOffset} are going
     * to satisfy the bounds<br>
     * {@link #getLowerBound() lowerBound}<sub>i</sub> &le;
     * {@link #trustRegionCenterOffset}<sub>i</sub>,<br>
     * with appropriate equalities when {@link #trustRegionCenterOffset} is
     * on a constraint boundary.
     * XXX "sl" in the original code.
     */
    private ArrayRealVector lowerDifference;
    /**
     * Differences {@link #getUpperBound()} - {@link #originShift}
     * All the components of every {@link #trustRegionCenterOffset} are going
     * to satisfy the bounds<br>
     *  {@link #trustRegionCenterOffset}<sub>i</sub> &le;
     *  {@link #getUpperBound() upperBound}<sub>i</sub>,<br>
     * with appropriate equalities when {@link #trustRegionCenterOffset} is
     * on a constraint boundary.
     * XXX "su" in the original code.
     */
    private ArrayRealVector upperDifference;
    /**
     * Parameters of the implicit second derivatives of the quadratic model.
     * XXX "pq" in the original code.
     */
    private ArrayRealVector modelSecondDerivativesParameters;
    /**
     * Point chosen by function {@link #trsbox(double) trsbox}
     * or {@link #altmov(int,double) altmov}.
     * Usually {@link #originShift} + {@link #newPoint} is the vector of
     * variables for the next evaluation of the objective function.
     * It also satisfies the constraints indicated in {@link #lowerDifference}
     * and {@link #upperDifference}.
     * XXX "xnew" in the original code.
     */
    private ArrayRealVector newPoint;
    /**
     * Alternative to {@link #newPoint}, chosen by
     * {@link #altmov(int,double) altmov}.
     * It may replace {@link #newPoint} in order to increase the denominator
     * in the {@link #update(double, double, int) updating procedure}.
     * XXX "xalt" in the original code.
     */
    private ArrayRealVector alternativeNewPoint;
    /**
     * Trial step from {@link #trustRegionCenterOffset} which is usually
     * {@link #newPoint} - {@link #trustRegionCenterOffset}.
     * XXX "d__" in the original code.
     */
    private ArrayRealVector trialStepPoint;
    /**
     * Explicit second derivatives of the quadratic model.
     * XXX "hq" in the original code.
     */
    private ArrayRealVector modelSecondDerivativesValues;

    /**
     * Lower bounds (constraints) of the objective variables.
     */
    private ArrayRealVector lowerBounds;

    /**
     * Upper bounds (constraints) of the objective variables.
     */
    private ArrayRealVector upperBounds;

    private int dimension;

    /**
     * @param numberOfInterpolationPoints Number of interpolation conditions.
     * For a problem of dimension {@code n}, its value must be in the interval
     * {@code [n+2, (n+1)(n+2)/2]}.
     * Choices that exceed {@code 2n+1} are not recommended.
     */
    public BOBYQAOptimizer(int numberOfInterpolationPoints) {
        this(numberOfInterpolationPoints,
             DEFAULT_INITIAL_RADIUS,
             DEFAULT_STOPPING_RADIUS);
    }

    /**
     * @param numberOfInterpolationPoints Number of interpolation conditions.
     * For a problem of dimension {@code n}, its value must be in the interval
     * {@code [n+2, (n+1)(n+2)/2]}.
     * Choices that exceed {@code 2n+1} are not recommended.
     * @param initialTrustRegionRadius Initial trust region radius.
     * @param stoppingTrustRegionRadius Stopping trust region radius.
     */
    public BOBYQAOptimizer(int numberOfInterpolationPoints,
                           double initialTrustRegionRadius,
                           double stoppingTrustRegionRadius) {
        super(null); // No custom convergence criterion.
        this.numberOfInterpolationPoints = numberOfInterpolationPoints;
        this.initialTrustRegionRadius = initialTrustRegionRadius;
        this.stoppingTrustRegionRadius = stoppingTrustRegionRadius;
    }

    // This subroutine seeks the least value of a function of many variables,
    // by applying a trust region method that forms quadratic models by
    // interpolation. There is usually some freedom in the interpolation
    // conditions, which is taken up by minimizing the Frobenius norm of
    // the change to the second derivative of the model, beginning with the
    // zero matrix. The values of the variables are constrained by upper and
    // lower bounds. The arguments of the subroutine are as follows.
    //
    // N must be set to the number of variables and must be at least two.
    // numberOfInterpolationPoints is the number of interpolation conditions.
    // Its value must be in the interval [N+2,(N+1)(N+2)/2].
    // Choices that exceed 2*N+1 are not recommended.
    // Initial values of the variables must be set in X(1),X(2),...,X(N). They
    //   will be changed to the values that give the least calculated F.
    // For I=1,2,...,N, XL(I) and XU(I) must provide the lower and upper
    //   bounds, respectively, on X(I). The construction of quadratic models
    //   requires XL(I) to be strictly less than XU(I) for each I. Further,
    //   the contribution to a model from changes to the I-th variable is
    //   damaged severely by rounding errors if XU(I)-XL(I) is too small.
    // RHOBEG and RHOEND must be set to the initial and final values of a trust
    //   region radius, so both must be positive with RHOEND no greater than
    //   RHOBEG. Typically, RHOBEG should be about one tenth of the greatest
    //   expected change to a variable, while RHOEND should indicate the
    //   accuracy that is required in the final values of the variables. An
    //   error return occurs if any of the differences XU(I)-XL(I), I=1,...,N,
    //   is less than 2*RHOBEG.
    // MAXFUN must be set to an upper bound on the number of calls of CALFUN.
    // The array W will be used for working space. Its length must be at least
    //   (numberOfInterpolationPoints+5)*(numberOfInterpolationPoints+N)+3*N*(N+5)/2.
    /** {@inheritDoc} */
    @Override
    protected PointValuePair doOptimize() {
        init();

        return bobyqb();
    }

    // ----------------------------------------------------------------------------------------

    /**
     *     The arguments N, numberOfInterpolationPoints, X, XL, XU, RHOBEG, RHOEND,
     *     IPRINT and MAXFUN are identical to the corresponding arguments of BOBYQA.
     *     XBASE holds a shift of origin that should reduce the contributions
     *       from rounding errors to values of the model and Lagrange functions.
     *     interpolationPoints is a matrix that holds the coordinates of the
     *       interpolation points relative to XBASE.
     *     FVAL holds the values of F at the interpolation points.
     *     XOPT is set to the displacement from XBASE of the trust region centre.
     *     gradientAtTrustRegionCenter holds the gradient of the quadratic model at XBASE+XOPT.
     *     HQ holds the explicit second derivatives of the quadratic model.
     *     PQ contains the parameters of the implicit second derivatives of the
     *       quadratic model.
     *     BMAT holds the last N columns of H.
     *     ZMAT holds the factorization of the leading numberOfInterpolationPoints by
     *       numberOfInterpolationPoints submatrix of H,
     *       this factorization being ZMAT times ZMAT^T, which provides both the
     *       correct rank and positive semi-definiteness.
     *     NDIM is the first dimension of BMAT and has the value numberOfInterpolationPoints+N.
     *     SL and SU hold the differences XL-XBASE and XU-XBASE, respectively.
     *       All the components of every XOPT are going to satisfy the bounds
     *       SL(I) .LEQ. XOPT(I) .LEQ. SU(I), with appropriate equalities when
     *       XOPT is on a constraint boundary.
     *     XNEW is chosen by SUBROUTINE TRSBOX or ALTMOV. Usually XBASE+XNEW is the
     *       vector of variables for the next call of CALFUN. XNEW also satisfies
     *       the SL and SU constraints in the way that has just been mentioned.
     *     XALT is an alternative to XNEW, chosen by ALTMOV, that may replace XNEW
     *       in order to increase the denominator in the updating of UPDATE.
     *     D is reserved for a trial step from XOPT, which is usually XNEW-XOPT.
     *     VLAG contains the values of the Lagrange functions at a new point X.
     *       They are part of a product that requires VLAG to be of length NDIM.
     *     W is a one-dimensional array that is used for working space. Its length
     *       must be at least 3*NDIM = 3*(numberOfInterpolationPoints+N).
     *
     * @return the value of the objective at the optimum.
     */
    private PointValuePair bobyqb() {
        final ArrayRealVector work1 = new ArrayRealVector(dimension);
        double dsq = Double.NaN;

        setInPlace(trustRegionCenterOffset, interpolationPoints.getRowVectorRef(trustRegionCenterInterpolationPointIndex));
        double xoptsq =  getSquaredNorm(trustRegionCenterOffset);
        double fsave = fAtInterpolationPoints.getEntry(0);
        int trustRegionIterations = 0;
        int itest = 0;
        int kNew = 0;
        int nfsav = getEvaluations();
        double rho = initialTrustRegionRadius;
        double delta = rho;
        double diffa = ZERO;
        double diffb = ZERO;
        double diffc = ZERO;
        double f = ZERO;
        double adelt = ZERO;
        double ratio = ZERO;
        double dnorm = ZERO;
        double distsq = ZERO;

        initializeGradient();

        int state = 60;
        goto_for: for(;;) {
        goto_switch: switch (state) {
        case 60: {
            // Generate the next point in the trust region that provides a small value
            // of the quadratic model subject to the constraints on the variables.
            // The integer trustRegionIterations is set to the number "trust region" iterations that
            // have occurred since the last "alternative" iteration. If the length
            // of XNEW-XOPT is less than HALF*RHO, however, then there is a branch to
            // label 650 or 680 with trustRegionIterations=-1, instead of calculating F at XNEW.
            final double[] dsqCrvmin = trsbox(delta);
            dsq = dsqCrvmin[0];
            final double crvmin = dsqCrvmin[1];

            dnorm = JdkMath.min(delta, JdkMath.sqrt(dsq));
            if (dnorm < HALF * rho) {
                trustRegionIterations = -1;
                distsq = power2(TEN * rho);
                if (getEvaluations() <= nfsav + 2) {
                    state = 650; break goto_switch;
                }

                // The following choice between labels 650 and 680 depends on whether or
                // not our work with the current RHO seems to be complete. Either RHO is
                // decreased or termination occurs if the errors in the quadratic model at
                // the last three interpolation points compare favourably with predictions
                // of likely improvements to the model within distance HALF*RHO of XOPT.

                // Computing MAX
                final double errbig = JdkMath.max(JdkMath.max(diffa, diffb), diffc);
                final double frhosq = rho * ONE_OVER_EIGHT * rho;
                if (crvmin > ZERO &&
                    errbig > frhosq * crvmin) {
                    state = 650; break goto_switch;
                }
                final double bdtol = errbig / rho;
                for (int j = 0; j < dimension; j++) {
                    double bdtest = bdtol;
                    if (newPoint.getEntry(j) == lowerDifference.getEntry(j)) {
                        bdtest = work1.getEntry(j);
                    }
                    if (newPoint.getEntry(j) == upperDifference.getEntry(j)) {
                        bdtest = -work1.getEntry(j);
                    }
                    if (bdtest < bdtol) {
                        double curv = modelSecondDerivativesValues.getEntry((j + 1 +  (j+1)*(j+1)) / 2 - 1);
                        for (int k = 0; k < numberOfInterpolationPoints; k++) {
                            curv += modelSecondDerivativesParameters.getEntry(k) * power2(interpolationPoints.getEntry(k, j));;
                        }
                        bdtest += HALF * curv * rho;
                        if (bdtest < bdtol) {
                            state = 650; break goto_switch;
                        }
                    }
                }
                state = 680; break goto_switch;
            }
            ++trustRegionIterations;
        }
        case 90: {
            // Severe cancellation is likely to occur if XOPT is too far from XBASE.
            // If the following test holds, then XBASE is shifted so that XOPT becomes
            // zero. The appropriate changes are made to BMAT and to the second
            // derivatives of the current model, beginning with the changes to BMAT
            // that do not depend on ZMAT. VLAG is used temporarily for working space.
            if (dsq <= xoptsq * ONE_OVER_A_THOUSAND) {
                final double fracsq = xoptsq * ONE_OVER_FOUR;
                final RealVector workVector = interpolationPoints.operate(trustRegionCenterOffset).mapSubtractToSelf(HALF * xoptsq);

                for (int k = 0; k < numberOfInterpolationPoints; k++) {
                    final double sum = workVector.getEntry(k);
                    final double temp = fracsq - HALF * sum;
                    final ArrayRealVector work0 = new ArrayRealVector(dimension);
                    final RealVector work1Bis = new ArrayRealVector(dimension);
                    for (int i = 0; i < dimension; i++) {
                        work1Bis.setEntry(i, bMatrix.getEntry(k, i));
                        work0.setEntry(i, sum * interpolationPoints.getEntry(k, i) + temp * trustRegionCenterOffset.getEntry(i));
                        final int ip = numberOfInterpolationPoints + i;
                        for (int j = 0; j <= i; j++) {
                            bMatrix.setEntry(ip, j,
                                          bMatrix.getEntry(ip, j)
                                          + work1Bis.getEntry(i) * work0.getEntry(j)
                                          + work0.getEntry(i) * work1Bis.getEntry(j));
                        }
                    }
                }

                // Then the revisions of BMAT that depend on ZMAT are calculated.
                for (int m = 0; m < zMatrix.getColumnDimension(); m++) {
                    final double sumZ = getSum(zMatrix.transpose().getRowVector(m));
                    final RealVector work0 = zMatrix.transpose().getRowVector(m).ebeMultiply(workVector);
                    final double sumW = getSum(work0);
                    final RealVector work1Bis = trustRegionCenterOffset.mapMultiply(fracsq * sumZ - HALF * sumW)
                        .add(interpolationPoints.preMultiply(work0));
                    for (int j = 0; j < dimension; j++) {
                        final double sum = work1Bis.getEntry(j);
                        for (int k = 0; k < numberOfInterpolationPoints; k++) {
                            bMatrix.setEntry(k, j, bMatrix.getEntry(k, j) + sum * zMatrix.getEntry(k, m));
                        }
                    }
                    for (int i = 0; i < dimension; i++) {
                        final int ip = i + numberOfInterpolationPoints;
                        final double temp = work1Bis.getEntry(i);
                        for (int j = 0; j <= i; j++) {
                            bMatrix.setEntry(ip, j, bMatrix.getEntry(ip, j) + temp * work1Bis.getEntry(j));
                        }
                    }
                }

                // The following instructions complete the shift, including the changes
                // to the second derivative parameters of the quadratic model.

                final double sumpq = getSum(modelSecondDerivativesParameters);
                int ih = 0;
                final RealVector work1Quatro = trustRegionCenterOffset.mapMultiply(-HALF * sumpq)
                    .add(interpolationPoints.preMultiply(modelSecondDerivativesParameters));
                subtractInPlace(interpolationPoints, trustRegionCenterOffset);
                for (int j = 0; j < dimension; j++) {
                    for (int i = 0; i <= j; i++) {
                         modelSecondDerivativesValues.setEntry(ih,
                                    modelSecondDerivativesValues.getEntry(ih)
                                    + work1Quatro.getEntry(i) * trustRegionCenterOffset.getEntry(j)
                                    + trustRegionCenterOffset.getEntry(i) * work1Quatro.getEntry(j));
                        bMatrix.setEntry(numberOfInterpolationPoints + i, j, bMatrix.getEntry(
                            numberOfInterpolationPoints + j, i));
                        ih++;
                    }
                }
                addInPlace(originShift, trustRegionCenterOffset);
                subtractInPlace(newPoint, trustRegionCenterOffset);
                subtractInPlace(lowerDifference, trustRegionCenterOffset);
                subtractInPlace(upperDifference, trustRegionCenterOffset);
                setZeroInPlace(trustRegionCenterOffset);
                xoptsq = ZERO;
            }

            double cauchy = Double.NaN;
            double alpha = Double.NaN;
            if (trustRegionIterations == 0) {
                // Pick two alternative vectors of variables, relative to XBASE, that
                // are suitable as new positions of the KNEW-th interpolation point.
                // Firstly, XNEW is set to the point on a line through XOPT and another
                // interpolation point that minimizes the predicted value of the next
                // denominator, subject to ||XNEW - XOPT|| .LEQ. ADELT and to the SL
                // and SU bounds. Secondly, XALT is set to the best feasible point on
                // a constrained version of the Cauchy step of the KNEW-th Lagrange
                // function, the corresponding value of the square of this function
                // being returned in CAUCHY. The choice between these alternatives is
                // going to be made when the denominator is calculated.
                final double[] alphaCauchy = altmov(kNew, adelt);
                alpha = alphaCauchy[0];
                cauchy = alphaCauchy[1];
                setInPlace(trialStepPoint, newPoint.subtract(trustRegionCenterOffset));
            }
            // Calculate Hw components and BETA for the current choice of D. The scalar
            // product of D with interpolationPoints(K,.) is going to be held in W(numberOfInterpolationPoints+K) for
            // use when VQUAD is calculated.
            // see (4.9) to (4.11)
            RealVector work2 = null;
            double beta = 0;
            RealVector startOfHw= null;
            RealVector tailOfHw = null;
            double denominator = 0;
            boolean denominatorOk = false;
            while (!denominatorOk) {
                work2 = interpolationPoints.operate(trialStepPoint);
                final RealVector workb = interpolationPoints.operate(trustRegionCenterOffset);
                final RealVector work3 = new ArrayRealVector(work2).mapMultiplyToSelf(HALF).add(workb).ebeMultiply(work2);
                final RealVector work4 = zMatrix.preMultiply(work3);

                final RealVector work5 = zMatrix.transpose().preMultiply(work4);
                startOfHw = bMatrix.operate(trialStepPoint).getSubVector(0, numberOfInterpolationPoints).add(work5);

                final RealVector work6 = bMatrix
                    .getSubMatrix(0, numberOfInterpolationPoints - 1, 0, dimension - 1)
                    .preMultiply(work3);
                tailOfHw = work6
                    .add(bMatrix
                        .getSubMatrix(numberOfInterpolationPoints, numberOfInterpolationPoints+dimension-1, 0, dimension -1).operate(trialStepPoint)
                    );
                final double bsum =  work6.add(tailOfHw).dotProduct(trialStepPoint);
                final double dx = trialStepPoint.dotProduct(trustRegionCenterOffset);
                final double stepL2Squared = getSquaredNorm(trialStepPoint);

                beta = dx * dx + stepL2Squared * (xoptsq + dx + dx + HALF * stepL2Squared) - work4.dotProduct(work4) - bsum;

                startOfHw.setEntry(trustRegionCenterInterpolationPointIndex,
                    startOfHw.getEntry(trustRegionCenterInterpolationPointIndex) + ONE);

                if (trustRegionIterations == 0) {
                    // If trustRegionIterations is zero, the denominator may be increased by replacing
                    // the step D of ALTMOV by a Cauchy step. Then RESCUE may be called if
                    // rounding errors have damaged the chosen denominator.
                    denominator = power2(startOfHw.getEntry(kNew)) + alpha * beta;
                    if (denominator < cauchy && cauchy > ZERO) {
                        setInPlace(newPoint, alternativeNewPoint);
                        setInPlace(trialStepPoint, newPoint.subtract(trustRegionCenterOffset));
                        cauchy = ZERO;
                        continue;
                    }
                }
                denominatorOk = true;
            }

            if (trustRegionIterations != 0) {
                // Alternatively, if trustRegionIterations is positive, then set KNEW to the index of
                // the next interpolation point to be deleted to make room for a trust
                // region step.
                final double deltaSquared = delta * delta;
                double maximumWeightedDenominator = ZERO;
                double rescueThreshold = ZERO;
                kNew = 0;
                for (int k = 0; k < numberOfInterpolationPoints; k++) {
                    if (k == trustRegionCenterInterpolationPointIndex) {
                        continue;
                    }
                    // apply (6.1)
                    final double hdiag = getSquaredNorm(zMatrix.getRowVectorRef(k));
                    final double sigma = beta * hdiag + power2(startOfHw.getEntry(k));
                    double distanceSquared = squaredL2Distance(interpolationPoints.getRowVectorRef(k), trustRegionCenterOffset);
                    // NOTE: it is not clear in paper that a power2 is necessary here - but there is one in the implem, and it works better
                    final double weightedDenominator = JdkMath.max(ONE,power2(distanceSquared / deltaSquared)) * sigma;
                    if (weightedDenominator > maximumWeightedDenominator) {
                        maximumWeightedDenominator = weightedDenominator;
                        kNew = k;
                        denominator = sigma;
                    }

                    // RESCUE may be called if rounding errors have damaged_
                    // the chosen denominator, which is the reason for attempting to select
                    // KNEW before calculating the next value of the objective function.
                    // biglsq is used to determine of RESCUE should be run
                    // RESCUE is not implemented - dead code
                    rescueThreshold = JdkMath.max(rescueThreshold, JdkMath.max(ONE,distanceSquared / deltaSquared) * power2(startOfHw.getEntry(k)));
                }
            }

            // Put the variables for the next calculation of the objective function
            //   in XNEW, with any adjustments for the bounds.
            // Calculate the value of the objective function at XBASE+XNEW, unless
            //   the limit on the number of calculations of F has been reached.
            setInPlace(currentBest, clipSelf(originShift.add(newPoint), lowerBounds, upperBounds));
            f = computeF(currentBest);

            // Use the quadratic model to predict the change in F due to the step D,
            //   and set DIFF to the error of this prediction.

            final double fopt = fAtInterpolationPoints.getEntry(trustRegionCenterInterpolationPointIndex);
            double vquad = trialStepPoint.dotProduct(gradientAtTrustRegionCenter);
            int ih1 = 0;
            for (int j = 0; j < dimension; j++) {
                for (int i = 0; i <= j; i++) {
                    double temp = trialStepPoint.getEntry(i) * trialStepPoint.getEntry(j);
                    if (i == j) {
                        temp *= HALF;
                    }
                    vquad += modelSecondDerivativesValues.getEntry(ih1) * temp;
                    ih1++;
               }
            }
            vquad += HALF * modelSecondDerivativesParameters.dotProduct(work2.ebeMultiply(work2));
            final double diff = f - fopt - vquad;
            diffc = diffb;
            diffb = diffa;
            diffa = JdkMath.abs(diff);
            if (dnorm > rho) {
                nfsav = getEvaluations();
            }

            // Pick the next value of DELTA after a trust region step.
            if (trustRegionIterations > 0) {
                if (vquad >= ZERO) {
                    throw new MathIllegalStateException(LocalizedFormats.TRUST_REGION_STEP_FAILED, vquad);
                }
                ratio = (f - fopt) / vquad;
                final double hDelta = HALF * delta;
                if (ratio <= ONE_OVER_TEN) {
                    delta = JdkMath.min(hDelta, dnorm);
                } else if (ratio <= .7) {
                    delta = JdkMath.max(hDelta, dnorm);
                } else {
                    delta = JdkMath.max(hDelta, 2 * dnorm);
                }
                if (delta <= rho * 1.5) {
                    delta = rho;
                }

                // Recalculate KNEW and DENOM if the new F is less than FOPT.

                if (f < fopt) {
                    final int ksav = kNew;
                    final double densav = denominator;
                    final double delsq = delta * delta;
                    double maximumWeightedDenominator = ZERO;
                    double biglsq = ZERO;
                    kNew = 0;
                    for (int k = 0; k < numberOfInterpolationPoints; k++) {
                        final double hdiag = getSquaredNorm(zMatrix.getRowVectorRef(k));;
                        final double den = beta * hdiag + power2(startOfHw.getEntry(k));
                        final double distanceSquared = squaredL2Distance(interpolationPoints.getRowVectorRef(k), newPoint);
                        // NOTE: it is not clear in paper that a power2 is necessary here - but there is one in the implem, and it works better
                        final double temp = JdkMath.max(ONE, power2(distanceSquared / delsq));
                        if (temp * den > maximumWeightedDenominator) {
                            maximumWeightedDenominator = temp * den;
                            kNew = k;
                            denominator = den;
                        }
                        // Computing MAX
                        final double d5 = temp * power2(startOfHw.getEntry(k));
                        biglsq = JdkMath.max(biglsq, d5);
                    }
                    if (maximumWeightedDenominator <= HALF * biglsq) {
                        kNew = ksav;
                        denominator = densav;
                    }
                }
            }

            // Update BMAT and ZMAT, so that the KNEW-th interpolation point can be
            // moved. Also update the second derivative terms of the model.
            update(beta, denominator, kNew, startOfHw, tailOfHw);

            int ih2 = 0;
            final double pqold = modelSecondDerivativesParameters.getEntry(kNew);
            modelSecondDerivativesParameters.setEntry(kNew, ZERO);
            for (int i = 0; i < dimension; i++) {
                final double temp = pqold * interpolationPoints.getEntry(kNew, i);
                for (int j = 0; j <= i; j++) {
                    modelSecondDerivativesValues.setEntry(ih2, modelSecondDerivativesValues.getEntry(ih2) + temp * interpolationPoints.getEntry(kNew, j));
                    ih2++;
                }
            }
            addInPlace(modelSecondDerivativesParameters, zMatrix.operate(zMatrix.getRowVectorRef(kNew).mapMultiply(diff)));

            // Include the new interpolation point, and make the changes to gradientAtTrustRegionCenter at
            // the old XOPT that are caused by the updating of the quadratic model.
            fAtInterpolationPoints.setEntry(kNew,  f);
            setRow(interpolationPoints, kNew, newPoint);
            setInPlace(work1, bMatrix.getRowVectorRef(kNew));
            final RealVector tmp1 = zMatrix.operate(zMatrix.getRowVectorRef(kNew));
            final RealVector tmp2 = interpolationPoints.operate(trustRegionCenterOffset);
            addInPlace(work1, interpolationPoints.preMultiply(tmp1.ebeMultiply(tmp2)));
            addInPlace(gradientAtTrustRegionCenter, work1.mapMultiply(diff));

            // Update XOPT, gradientAtTrustRegionCenter and KOPT if the new calculated F is less than FOPT.
            if (f < fopt) {
                trustRegionCenterInterpolationPointIndex = kNew;
                setInPlace(trustRegionCenterOffset, newPoint);
                xoptsq = getSquaredNorm(trustRegionCenterOffset);
                int ih = 0;
                for (int j = 0; j < dimension; j++) {
                    for (int i = 0; i <= j; i++) {
                        if (i < j) {
                            gradientAtTrustRegionCenter.setEntry(j, gradientAtTrustRegionCenter.getEntry(j) + modelSecondDerivativesValues.getEntry(ih) * trialStepPoint.getEntry(i));
                        }
                        gradientAtTrustRegionCenter.setEntry(i, gradientAtTrustRegionCenter.getEntry(i) + modelSecondDerivativesValues.getEntry(ih) * trialStepPoint.getEntry(j));
                        ih++;
                    }
                }
                final RealVector gradientUpdate = interpolationPoints.preMultiply(interpolationPoints.operate(trialStepPoint)
                        .ebeMultiply(modelSecondDerivativesParameters));
                addInPlace(gradientAtTrustRegionCenter, gradientUpdate);
            }

            // Calculate the parameters of the least Frobenius norm interpolant to
            // the current data, the gradient of this interpolant at XOPT being put
            // into VLAG(numberOfInterpolationPoints+I), I=1,2,...,N.

            if (trustRegionIterations > 0) {
                final RealVector work0 = fAtInterpolationPoints.mapSubtract(fAtInterpolationPoints.getEntry(trustRegionCenterInterpolationPointIndex));
                final RealVector work11 = zMatrix.operate(zMatrix.transpose().operate(work0));
                final RealVector work22 = work11.ebeMultiply(interpolationPoints.operate(trustRegionCenterOffset));
                final RealVector leastFrobeniusNormInterpolantGradient = bMatrix
                    .getSubMatrix(0, numberOfInterpolationPoints - 1, 0, dimension -1)
                    .preMultiply(work0)
                    .add(interpolationPoints.preMultiply(work22));
                final double gqSquared = getSquaredNorm(boundSafeGradient(gradientAtTrustRegionCenter));
                final double giSquared = getSquaredNorm(boundSafeGradient(leastFrobeniusNormInterpolantGradient));

                // Test whether to replace the new quadratic model by the least Frobenius
                // norm interpolant, making the replacement if the test is satisfied.
                ++itest;
                if (gqSquared < TEN * giSquared) {
                    itest = 0;
                }
                if (itest >= 3) {
                    // perform replacement
                    setInPlace(gradientAtTrustRegionCenter, leastFrobeniusNormInterpolantGradient);
                    setInPlace(modelSecondDerivativesParameters, work11);
                    setZeroInPlace(modelSecondDerivativesValues);
                    itest = 0;
                }
            }

            // If a trust region step has provided a sufficient decrease in F, then
            // branch for another trust region calculation. The case trustRegionIterations=0 occurs
            // when the new interpolation point was reached by an alternative step.
            if (f <= fopt + ONE_OVER_TEN * vquad || trustRegionIterations == 0) {
                state = 60; break goto_switch;
            }

            // Alternatively, find out if the interpolation points are close enough
            //   to the best point so far.

            distsq = JdkMath.max(power2(TWO * delta), power2(TEN * rho));
        }
        case 650: {
            kNew = -1;
            for (int k = 0; k < numberOfInterpolationPoints; k++) {
                final double squaredL2Atk =  squaredL2Distance(interpolationPoints.getRowVectorRef(k), trustRegionCenterOffset);
                if (squaredL2Atk > distsq) {
                    kNew = k;
                    distsq = squaredL2Atk;
                }
            }

            // If KNEW is positive, then ALTMOV finds alternative new positions for
            // the KNEW-th interpolation point within distance ADELT of XOPT. It is
            // reached via label 90. Otherwise, there is a branch to label 60 for
            // another trust region iteration, unless the calculations with the
            // current RHO are complete.

            if (kNew >= 0) {
                final double dist = JdkMath.sqrt(distsq);
                if (trustRegionIterations == -1) {
                    delta = JdkMath.min(ONE_OVER_TEN * delta, HALF * dist);
                    if (delta <= rho * 1.5) {
                        delta = rho;
                    }
                }
                trustRegionIterations = 0;
                adelt = clip(delta,rho, ONE_OVER_TEN * dist);
                dsq = adelt * adelt;
                state = 90; break goto_switch;
            }
            if (trustRegionIterations == -1) {
                state = 680; break goto_switch;
            }
            if (ratio > ZERO || JdkMath.max(delta, dnorm) > rho) {
                state = 60; break goto_switch;
            }

            // The calculations with the current value of RHO are complete. Pick the
            //   next values of RHO and DELTA.
        }
        case 680: {
            if (rho > stoppingTrustRegionRadius) {
                // update rho and loop formula (6.6)
                delta = HALF * rho;
                if (rho <= SIXTEEN * stoppingTrustRegionRadius) {
                    rho = stoppingTrustRegionRadius;
                } else if (rho <= TWO_HUNDRED_FIFTY * stoppingTrustRegionRadius) {
                    rho = JdkMath.sqrt(rho * stoppingTrustRegionRadius) ;
                } else {
                    // note: the formula in the BOBYQA paper has a typo. For correct formula see (7.6) of The NEWUOA software for unconstrained optimization without derivatives.
                    rho = ONE_OVER_TEN * rho;
                }
                delta = JdkMath.max(delta, rho);
                trustRegionIterations = 0;
                nfsav = getEvaluations();
                state = 60; break goto_switch;
            } else {
                // termination condition is met
                break goto_for;
            }
        }
        default: {
            throw new MathIllegalStateException(LocalizedFormats.SIMPLE_MESSAGE, "bobyqb");
        }}}

        // perform a Newton-Raphson step if the calculation was too short to have been tried before.
        if (trustRegionIterations == -1) {
            setInPlace(currentBest, clipSelf(originShift.add(newPoint), lowerBounds, upperBounds));
            f = computeF(currentBest);
            fsave = f;
        }

        if (fAtInterpolationPoints.getEntry(trustRegionCenterInterpolationPointIndex) <= fsave) {
            setInPlace(currentBest, clipSelf(originShift.add(trustRegionCenterOffset), lowerBounds, upperBounds));
            f = fAtInterpolationPoints.getEntry(trustRegionCenterInterpolationPointIndex);
        }

        return new PointValuePair(currentBest.getDataRef(),
            (getGoalType().equals(GoalType.MINIMIZE)) ? f : -f);
    } // bobyqb

    private RealVector boundSafeGradient(final RealVector gradient) {
        final RealVector safeGradient = new ArrayRealVector(gradient);
        for (int i = 0; i < gradient.getDimension(); i++) {
            if (trustRegionCenterOffset.getEntry(i) == lowerDifference.getEntry(i)) {
                safeGradient.setEntry(i, JdkMath.min(ZERO, gradientAtTrustRegionCenter.getEntry(i)));
            } else if (trustRegionCenterOffset.getEntry(i) == upperDifference.getEntry(i)) {
                safeGradient.setEntry(i, JdkMath.max(ZERO, gradientAtTrustRegionCenter.getEntry(i)));
            }
        }
        return safeGradient;
    }

    private void initializeGradient() {
        // to use before the optimization loop
        // update gradientAtTrustRegionCenter if necessary before the first optimization iteration
        // NOTE: can also be used after call of RESCUE that makes a call of CALFUN, but RESCUE is not implemented
        if (trustRegionCenterInterpolationPointIndex != 0) {
            int ih = 0;
            for (int j = 0; j < dimension; j++) {
                for (int i = 0; i <= j; i++) {
                    if (i < j) {
                        gradientAtTrustRegionCenter.setEntry(j, gradientAtTrustRegionCenter.getEntry(j) + modelSecondDerivativesValues.getEntry(ih) * trustRegionCenterOffset.getEntry(i));
                    }
                    gradientAtTrustRegionCenter.setEntry(i, gradientAtTrustRegionCenter.getEntry(i) + modelSecondDerivativesValues.getEntry(ih) * trustRegionCenterOffset.getEntry(j));
                    ih++;
                }
            }
            // dead code - can only happen in a RESCUE context, but RESCUE is not implemented
            if (getEvaluations() > numberOfInterpolationPoints) {
                for (int k = 0; k < numberOfInterpolationPoints; k++) {
                    double temp = ZERO;
                    for (int j = 0; j < dimension; j++) {
                        temp += interpolationPoints.getEntry(k, j) * trustRegionCenterOffset.getEntry(j);
                    }
                    temp *= modelSecondDerivativesParameters.getEntry(k);
                    for (int i = 0; i < dimension; i++) {
                        gradientAtTrustRegionCenter.setEntry(i, gradientAtTrustRegionCenter.getEntry(i) + temp * interpolationPoints.getEntry(k, i));
                    }
                }
            }
        }
    }

    // ----------------------------------------------------------------------------------------

    /**
     *     The arguments N, numberOfInterpolationPoints, interpolationPoints, XOPT, BMAT, ZMAT, NDIM, SL and
     *     SU all have the same meanings as the corresponding arguments of BOBYQB.
     *     KOPT is the index of the optimal interpolation point.
     *     KNEW is the index of the interpolation point that is going to be moved.
     *     ADELT is the current trust region bound.
     *     XNEW will be set to a suitable new position for the interpolation point
     *       interpolationPoints(KNEW,.). Specifically, it satisfies the SL, SU and trust region
     *       bounds and it should provide a large denominator in the next call of
     *       UPDATE. The step XNEW-XOPT from XOPT is restricted to moves along the
     *       straight lines through XOPT and another interpolation point.
     *     XALT also provides a large value of the modulus of the KNEW-th Lagrange
     *       function subject to the constraints that have been mentioned, its main
     *       difference from XNEW being that XALT-XOPT is a constrained version of
     *       the Cauchy step within the trust region. An exception is that XALT is
     *       not calculated if all components of GLAG (see below) are zero.
     *     ALPHA will be set to the KNEW-th diagonal element of the H matrix.
     *     CAUCHY will be set to the square of the KNEW-th Lagrange function at
     *       the step XALT-XOPT from XOPT for the vector XALT that is returned,
     *       except that CAUCHY is set to zero if XALT is not calculated.
     *     GLAG is a working space vector of length N for the gradient of the
     *       KNEW-th Lagrange function at XOPT.
     *     HCOL is a working space vector of length numberOfInterpolationPoints for the second
     *     derivative coefficients of the KNEW-th Lagrange function.
     *     W is a working space vector of length 2N that is going to hold the
     *       constrained Cauchy step from XOPT of the Lagrange function, followed
     *       by the downhill version of XALT when the uphill step is calculated.
     *
     *     Set the first numberOfInterpolationPoints components of W to the leading elements of the
     *     KNEW-th column of the H matrix.
     * @param kNew
     * @param adelt
     * @return { alpha, cauchy }
     */
    private double[] altmov(final int kNew, final double adelt) {
        final RealVector hcol = zMatrix.operate(zMatrix.getRowVectorRef(kNew));

        // Calculate the gradient of the KNEW-th Lagrange function at XOPT.
        final RealVector temp1 = interpolationPoints.operate(trustRegionCenterOffset).ebeMultiply(hcol);
        final RealVector temp2 = interpolationPoints.preMultiply(temp1);
        final RealVector glag = bMatrix.getRowVectorRef(kNew).add(temp2);

        // Search for a large denominator along the straight lines through XOPT
        // and another interpolation point. SLBD and SUBD will be lower and upper
        // bounds on the step along each of these lines in turn. PREDSQ will be
        // set to the square of the predicted denominator for each line. PRESAV
        // will be set to the largest admissible value of PREDSQ that occurs.

        double presav = ZERO;
        double step = Double.NaN;
        int ksav = 0;
        int ibdsav = 0;
        double stpsav = 0;
        final double alpha = hcol.getEntry(kNew);
        final double halfAlpha = HALF * alpha;
        for (int k = 0; k < numberOfInterpolationPoints; k++) {
            if (k == trustRegionCenterInterpolationPointIndex) {
                continue;
            }
            double dderiv = ZERO;
            double distsq = ZERO;
            for (int i = 0; i < dimension; i++) {
                final double tmp = interpolationPoints.getEntry(k, i) - trustRegionCenterOffset.getEntry(i);
                dderiv += glag.getEntry(i) * tmp;
                distsq += tmp * tmp;
            }
            double subd = adelt / JdkMath.sqrt(distsq);
            double slbd = -subd;
            int ilbd = 0;
            int iubd = 0;
            final double sumin = JdkMath.min(ONE, subd);

            // Revise SLBD and SUBD if necessary because of the bounds in SL and SU.

            for (int i = 0; i < dimension; i++) {
                final double tmp = interpolationPoints.getEntry(k, i) - trustRegionCenterOffset.getEntry(i);
                if (tmp > ZERO) {
                    if (slbd * tmp < lowerDifference.getEntry(i) - trustRegionCenterOffset.getEntry(i)) {
                        slbd = (lowerDifference.getEntry(i) - trustRegionCenterOffset.getEntry(i)) / tmp;
                        ilbd = -i - 1;
                    }
                    if (subd * tmp > upperDifference.getEntry(i) - trustRegionCenterOffset.getEntry(i)) {
                        // Computing MAX
                        subd = JdkMath.max(sumin,
                                            (upperDifference.getEntry(i) - trustRegionCenterOffset.getEntry(i)) / tmp);
                        iubd = i + 1;
                    }
                } else if (tmp < ZERO) {
                    if (slbd * tmp > upperDifference.getEntry(i) - trustRegionCenterOffset.getEntry(i)) {
                        slbd = (upperDifference.getEntry(i) - trustRegionCenterOffset.getEntry(i)) / tmp;
                        ilbd = i + 1;
                    }
                    if (subd * tmp < lowerDifference.getEntry(i) - trustRegionCenterOffset.getEntry(i)) {
                        // Computing MAX
                        subd = JdkMath.max(sumin,
                                            (lowerDifference.getEntry(i) - trustRegionCenterOffset.getEntry(i)) / tmp);
                        iubd = -i - 1;
                    }
                }
            }

            // Seek a large modulus of the KNEW-th Lagrange function when the index
            // of the other interpolation point on the line through XOPT is KNEW.

            step = slbd;
            int isbd = ilbd;
            double vlag = Double.NaN;
            if (k == kNew) {
                final double diff = dderiv - ONE;
                vlag = slbd * (dderiv - slbd * diff);
                final double d1 = subd * (dderiv - subd * diff);
                if (JdkMath.abs(d1) > JdkMath.abs(vlag)) {
                    step = subd;
                    vlag = d1;
                    isbd = iubd;
                }
                final double d2 = HALF * dderiv;
                final double d3 = d2 - diff * slbd;
                final double d4 = d2 - diff * subd;
                if (d3 * d4 < ZERO) {
                    final double d5 = d2 * d2 / diff;
                    if (JdkMath.abs(d5) > JdkMath.abs(vlag)) {
                        step = d2 / diff;
                        vlag = d5;
                        isbd = 0;
                    }
                }

                // Search along each of the other lines through XOPT and another point.
            } else {
                vlag = slbd * (ONE - slbd);
                final double tmp = subd * (ONE - subd);
                if (JdkMath.abs(tmp) > JdkMath.abs(vlag)) {
                    step = subd;
                    vlag = tmp;
                    isbd = iubd;
                }
                if (subd > HALF && JdkMath.abs(vlag) < ONE_OVER_FOUR) {
                    step = HALF;
                    vlag = ONE_OVER_FOUR;
                    isbd = 0;
                }
                vlag *= dderiv;
            }

            // Calculate PREDSQ for the current line search and maintain PRESAV.

            final double tmp = step * (ONE - step) * distsq;
            final double predsq = vlag * vlag * (vlag * vlag + halfAlpha * tmp * tmp);
            if (predsq > presav) {
                presav = predsq;
                ksav = k;
                stpsav = step;
                ibdsav = isbd;
            }
        }

        // Construct XNEW in a way that satisfies the bound constraints exactly.
        setInPlace(newPoint, clipSelf(interpolationPoints.getRowVectorRef(ksav).subtract(trustRegionCenterOffset).mapMultiplyToSelf(stpsav).add(trustRegionCenterOffset),
            lowerDifference, upperDifference));
        if (ibdsav < 0) {
            newPoint.setEntry(-ibdsav - 1, lowerDifference.getEntry(-ibdsav - 1));
        }
        if (ibdsav > 0) {
            newPoint.setEntry(ibdsav - 1, upperDifference.getEntry(ibdsav - 1));
        }

        // Prepare for the iterative method that assembles the constrained Cauchy
        // step in W. The sum of squares of the fixed components of W is formed in
        // WFIXSQ, and the free components of W are set to BIGSTP.

        final double bigstp = adelt + adelt;
        int iflag = 0;
        double cauchy = Double.NaN;
        double csave = ZERO;
        final ArrayRealVector work1 = new ArrayRealVector(dimension);
        final ArrayRealVector work2 = new ArrayRealVector(dimension);
        while (true) {
            double wfixsq = ZERO;
            double ggfree = ZERO;
            for (int i = 0; i < dimension; i++) {
                final double glagValue = glag.getEntry(i);
                work1.setEntry(i, ZERO);
                if (JdkMath.min(trustRegionCenterOffset.getEntry(i) - lowerDifference.getEntry(i), glagValue) > ZERO ||
                    JdkMath.max(trustRegionCenterOffset.getEntry(i) - upperDifference.getEntry(i), glagValue) < ZERO) {
                    work1.setEntry(i, bigstp);
                    ggfree += power2(glagValue);
                }
            }
            if (ggfree == ZERO) {
                return new double[] { alpha, ZERO };
            }

            // Investigate whether more components of W can be fixed.
            final double tmp1 = adelt * adelt - wfixsq;
            if (tmp1 > ZERO) {
                step = JdkMath.sqrt(tmp1 / ggfree);
                ggfree = ZERO;
                for (int i = 0; i < dimension; i++) {
                    if (work1.getEntry(i) == bigstp) {
                        final double tmp2 = trustRegionCenterOffset.getEntry(i) - step * glag.getEntry(i);
                        if (tmp2 <= lowerDifference.getEntry(i)) {
                            work1.setEntry(i, lowerDifference.getEntry(i) - trustRegionCenterOffset.getEntry(i));
                            wfixsq += power2(work1.getEntry(i));
                        } else if (tmp2 >= upperDifference.getEntry(i)) {
                            work1.setEntry(i, upperDifference.getEntry(i) - trustRegionCenterOffset.getEntry(i));
                            wfixsq += power2(work1.getEntry(i));
                        } else {
                            ggfree += power2(glag.getEntry(i));
                        }
                    }
                }
            }

            // Set the remaining free components of W and all components of XALT,
            // except that W may be scaled later.

            double gw = ZERO;
            for (int i = 0; i < dimension; i++) {
                final double glagValue = glag.getEntry(i);
                if (work1.getEntry(i) == bigstp) {
                    work1.setEntry(i, -step * glagValue);
                    final double min = JdkMath.min(upperDifference.getEntry(i),
                                                    trustRegionCenterOffset.getEntry(i) + work1.getEntry(i));
                    alternativeNewPoint.setEntry(i, JdkMath.max(lowerDifference.getEntry(i), min));
                } else if (work1.getEntry(i) == ZERO) {
                    alternativeNewPoint.setEntry(i, trustRegionCenterOffset.getEntry(i));
                } else if (glagValue > ZERO) {
                    alternativeNewPoint.setEntry(i, lowerDifference.getEntry(i));
                } else {
                    alternativeNewPoint.setEntry(i, upperDifference.getEntry(i));
                }
                gw += glagValue * work1.getEntry(i);
            }

            // Set CURV to the curvature of the KNEW-th Lagrange function along W.
            // Scale W by a factor less than one if that can reduce the modulus of
            // the Lagrange function at XOPT+W. Set CAUCHY to the final value of
            // the square of this function.

            double curv = ZERO;
            for (int k = 0; k < numberOfInterpolationPoints; k++) {
                double tmp = ZERO;
                for (int j = 0; j < dimension; j++) {
                    tmp += interpolationPoints.getEntry(k, j) * work1.getEntry(j);
                }
                curv += hcol.getEntry(k) * tmp * tmp;
            }
            if (iflag == 1) {
                curv = -curv;
            }
            if (curv > -gw &&
                curv < -gw * (ONE + JdkMath.sqrt(TWO))) {
                final double scale = -gw / curv;
                setInPlace(alternativeNewPoint, clipSelf(work1.mapMultiply(scale).add(trustRegionCenterOffset),lowerDifference, upperDifference));
                cauchy = power2(HALF * gw * scale);
            } else {
                cauchy = power2(gw + HALF * curv);
            }

            // If IFLAG is zero, then XALT is calculated as before after reversing
            // the sign of GLAG. Thus two XALT vectors become available. The one that
            // is chosen is the one that gives the larger value of CAUCHY.

            if (iflag == 0) {
                for (int i = 0; i < dimension; i++) {
                    glag.setEntry(i, -glag.getEntry(i));
                    work2.setEntry(i, alternativeNewPoint.getEntry(i));
                }
                csave = cauchy;
                iflag = 1;
            } else {
                break;
            }
        }
        if (csave > cauchy) {
            for (int i = 0; i < dimension; i++) {
                alternativeNewPoint.setEntry(i, work2.getEntry(i));
            }
            cauchy = csave;
        }

        return new double[] { alpha, cauchy };
    } // altmov

    // used to get a matrix row as a vector without copying data
    private ArrayRealVector rowRef(final Array2DRowRealMatrix matrix, final int k) {
        return new ArrayRealVector(matrix.getDataRef()[k], false);
    }

    // ----------------------------------------------------------------------------------------

    /**
     *     Compute the trust region step.
     *     A version of the truncated conjugate gradient is applied. If a line
     *     search is restricted by a constraint, then the procedure is restarted,
     *     the values of the variables that are at their bounds being fixed. If
     *     the trust region boundary is reached, then further changes may be made
     *     to D, each one being in the two dimensional space that is spanned
     *     by the current D and the gradient of Q at XOPT+D, staying on the trust
     *     region boundary. Termination occurs when the reduction in Q seems to
     *     be close to the greatest reduction that can be achieved.
     *     The arguments N, numberOfInterpolationPoints, interpolationPoints, XOPT,
     *     gradientAtTrustRegionCenter, HQ, PQ, SL and SU
     *     have the same meanings as the corresponding arguments of BOBYQB.
     *     DELTA is the trust region radius for the present calculation, which
     *       seeks a small value of the quadratic model within distance DELTA of
     *       XOPT subject to the bounds on the variables.
     *     XNEW will be set to a new vector of variables that is approximately
     *       the one that minimizes the quadratic model within the trust region
     *       subject to the SL and SU constraints on the variables. It satisfies
     *       as equations the bounds that become active during the calculation.
     *     D is the calculated trial step from XOPT, generated iteratively from an
     *       initial value of zero. Thus XNEW is XOPT+D after the final iteration.
     *     GNEW holds the gradient of the quadratic model at XOPT+D. It is updated
     *       when D is updated.
     *     xbdi.get( is a working space vector. For I=1,2,...,N, the element xbdi.get((I) is
     *       set to -1.0, 0.0, or 1.0, the value being nonzero if and only if the
     *       I-th variable has become fixed at a bound, the bound being SL(I) or
     *       SU(I) in the case xbdi.get((I)=-1.0 or xbdi.get((I)=1.0, respectively. This
     *       information is accumulated during the construction of XNEW.
     *     The arrays S, HS and HRED are also used for working space. They hold the
     *       current search direction, and the changes in the gradient of Q along S
     *       and the reduced D, respectively, where the reduced D is the same as D,
     *       except that the components of the fixed variables are zero.
     *     DSQ will be set to the square of the length of XNEW-XOPT.
     *     CRVMIN is set to zero if D reaches the trust region boundary. Otherwise
     *       it is set to the least curvature of H that occurs in the conjugate
     *       gradient searches that are not restricted by any constraints. The
     *       value CRVMIN=-1.0D0 is set, however, if all of these searches are
     *       constrained.
     * @param delta
     * @return { dsq, crvmin }
     */
    private double[] trsbox(final double delta) {
        final ArrayRealVector gnew = new ArrayRealVector(dimension);
        final ArrayRealVector s = new ArrayRealVector(dimension);
        final ArrayRealVector hred = new ArrayRealVector(dimension);
        final ArrayRealVector hs = new ArrayRealVector(dimension);
        int iact = -1;
        double qred;
        double xsav = 0, angbd = 0, dredg = 0, sredg = 0;
        int iterc = 0;
        double dredsq = 0, gredsq = 0;
        int itcsav = 0;
        double stepsq = 0;
        int itermax = 0;

        // The sign of gradientAtTrustRegionCenter(I) gives the sign of the change to the I-th variable
        // that will reduce Q from its value at XOPT. Thus xbdi.get((I) shows whether
        // or not to fix the I-th variable at one of its bounds initially, with
        // NACT being set to the number of fixed variables. D and GNEW are also
        // set for the first iteration. DELSQ is the upper bound on the sum of
        // squares of the free variables. QRED is the reduction in Q so far.

        final ArrayRealVector xbdi = new ArrayRealVector(dimension);
        int nact = 0;
        for (int i = 0; i < dimension; i++) {
            if (trustRegionCenterOffset.getEntry(i) <= lowerDifference.getEntry(i)) {
                if (gradientAtTrustRegionCenter.getEntry(i) >= ZERO) {
                    xbdi.setEntry(i, MINUS_ONE);
                }
            } else if (trustRegionCenterOffset.getEntry(i) >= upperDifference.getEntry(i) &&
                    gradientAtTrustRegionCenter.getEntry(i) <= ZERO) {
                xbdi.setEntry(i, ONE);
            }
            if (xbdi.getEntry(i) != ZERO) {
                ++nact;
            }
        }
        for (int i = 0; i < dimension; i++) {
            trialStepPoint.setEntry(i, ZERO);
            gnew.setEntry(i, gradientAtTrustRegionCenter.getEntry(i));
        }
        double delsq = delta * delta;
        qred = ZERO;
        double crvmin = MINUS_ONE;

        // Set the next search direction of the conjugate gradient method. It is
        // the steepest descent direction initially and when the iterations are
        // restarted because a variable has just been fixed by a bound, and of
        // course the components of the fixed variables are zero. ITERMAX is an
        // upper bound on the indices of the conjugate gradient iterations.

        double beta = ZERO;
        int state = 30;
        for(;;) {
        goto_trsbox: switch (state) {
        case 30: {
            stepsq = ZERO;
            for (int i = 0; i < dimension; i++) {
                if (xbdi.getEntry(i) != ZERO) {
                    s.setEntry(i, ZERO);
                } else if (beta == ZERO) {
                    s.setEntry(i, -gnew.getEntry(i));
                } else {
                    s.setEntry(i, beta * s.getEntry(i) - gnew.getEntry(i));
                }
                stepsq += power2(s.getEntry(i));
            }
            if (stepsq == ZERO) {
                state = 190; break;
            }
            if (beta == ZERO) {
                gredsq = stepsq;
                itermax = iterc + dimension - nact;
            }
            if (gredsq * delsq <= qred * 1e-4 * qred) {
                state = 190; break;
            }

            // Multiply the search direction by the second derivative matrix of Q and
            // calculate some scalars for the choice of steplength. Then set BLEN to
            // the length of the step to the trust region boundary and STPLEN to
            // the steplength, ignoring the simple bounds.

            state = 210; break;
        }
        case 50: {
            double resid = delsq;
            double ds = ZERO;
            double shs = ZERO;
            for (int i = 0; i < dimension; i++) {
                if (xbdi.getEntry(i) == ZERO) {
                    resid -= power2(trialStepPoint.getEntry(i));
                    ds += s.getEntry(i) * trialStepPoint.getEntry(i);
                    shs += s.getEntry(i) * hs.getEntry(i);
                }
            }
            if (resid <= ZERO) {
                crvmin = ZERO;
                state = 100; break;
            }
            double temp = JdkMath.sqrt(stepsq * resid + ds * ds);
            double blen;
            if (ds < ZERO) {
                blen = (temp - ds) / stepsq;
            } else {
                blen = resid / (temp + ds);
            }
            double stplen = blen;
            if (shs > ZERO) {
                stplen = JdkMath.min(blen, gredsq / shs);
            }

            // Reduce STPLEN if necessary in order to preserve the simple bounds,
            // letting IACT be the index of the new constrained variable.

            iact = -1;
            for (int i = 0; i < dimension; i++) {
                if (s.getEntry(i) != ZERO) {
                    final double xsum = trustRegionCenterOffset.getEntry(i) + trialStepPoint.getEntry(i);
                    final double temp2;
                    if (s.getEntry(i) > ZERO) {
                        temp2 = (upperDifference.getEntry(i) - xsum) / s.getEntry(i);
                    } else {
                        temp2 = (lowerDifference.getEntry(i) - xsum) / s.getEntry(i);
                    }
                    if (temp2 < stplen) {
                        stplen = temp2;
                        iact = i;
                    }
                }
            }

            // Update CRVMIN, GNEW and D. Set SDEC to the decrease that occurs in Q.

            double sdec = ZERO;
            double ggsav = ZERO;
            if (stplen > ZERO) {
                ++iterc;
                temp = shs / stepsq;
                if (iact == -1 && temp > ZERO) {
                    crvmin = JdkMath.min(crvmin,temp);
                    if (crvmin == MINUS_ONE) {
                        crvmin = temp;
                    }
                }
                ggsav = gredsq;
                gredsq = ZERO;
                for (int i = 0; i < dimension; i++) {
                    gnew.setEntry(i, gnew.getEntry(i) + stplen * hs.getEntry(i));
                    if (xbdi.getEntry(i) == ZERO) {
                        gredsq += power2(gnew.getEntry(i));
                    }
                    trialStepPoint.setEntry(i, trialStepPoint.getEntry(i) + stplen * s.getEntry(i));
                }
                // Computing MAX
                final double d1 = stplen * (ggsav - HALF * stplen * shs);
                sdec = JdkMath.max(d1, ZERO);
                qred += sdec;
            }

            // Restart the conjugate gradient method if it has hit a new bound.

            if (iact >= 0) {
                ++nact;
                xbdi.setEntry(iact, ONE);
                if (s.getEntry(iact) < ZERO) {
                    xbdi.setEntry(iact, MINUS_ONE);
                }
                delsq -= power2(trialStepPoint.getEntry(iact));
                if (delsq <= ZERO) {
                    state = 190; break;
                }
                beta = ZERO;
                state = 30; break;
            }

            // If STPLEN is less than BLEN, then either apply another conjugate
            // gradient iteration or RETURN.

            if (stplen < blen) {
                if (iterc == itermax) {
                    state = 190; break;
                }
                if (sdec <= qred * .01) {
                    state = 190; break;
                }
                beta = gredsq / ggsav;
                state = 30; break;
            }
            crvmin = ZERO;
            state = 100; break;
        }
        // Prepare for the alternative iteration by calculating some scalars
        // and by multiplying the reduced D by the second derivative matrix of
        // Q, where S holds the reduced D in the call of GGMULT.
        case 100: {
            if (nact >= dimension - 1) {
                state = 190; break;
            }
            dredsq = ZERO;
            dredg = ZERO;
            gredsq = ZERO;
            for (int i = 0; i < dimension; i++) {
                if (xbdi.getEntry(i) == ZERO) {
                    dredsq += power2(trialStepPoint.getEntry(i));
                    dredg += trialStepPoint.getEntry(i) * gnew.getEntry(i);
                    gredsq += power2(gnew.getEntry(i));
                    s.setEntry(i, trialStepPoint.getEntry(i));
                } else {
                    s.setEntry(i, ZERO);
                }
            }
            itcsav = iterc;
            state = 210; break;
            // Let the search direction S be a linear combination of the reduced D
            // and the reduced G that is orthogonal to the reduced D.
        }
        case 120: {
            ++iterc;
            double temp = gredsq * dredsq - dredg * dredg;
            if (temp <= qred * 1e-4 * qred) {
                state = 190; break;
            }
            temp = JdkMath.sqrt(temp);
            for (int i = 0; i < dimension; i++) {
                if (xbdi.getEntry(i) == ZERO) {
                    s.setEntry(i, (dredg * trialStepPoint.getEntry(i) - dredsq * gnew.getEntry(i)) / temp);
                } else {
                    s.setEntry(i, ZERO);
                }
            }
            sredg = -temp;

            // By considering the simple bounds on the variables, calculate an upper
            // bound on the tangent of half the angle of the alternative iteration,
            // namely ANGBD, except that, if already a free variable has reached a
            // bound, there is a branch back to label 100 after fixing that variable.

            angbd = ONE;
            iact = -1;
            for (int i = 0; i < dimension; i++) {
                if (xbdi.getEntry(i) == ZERO) {
                    final double tempa = trustRegionCenterOffset.getEntry(i) + trialStepPoint.getEntry(i) - lowerDifference.getEntry(i);
                    final double tempb = upperDifference.getEntry(i) - trustRegionCenterOffset.getEntry(i) - trialStepPoint.getEntry(i);
                    if (tempa <= ZERO) {
                        ++nact;
                        xbdi.setEntry(i, MINUS_ONE);
                        state = 100; break goto_trsbox;
                    } else if (tempb <= ZERO) {
                        ++nact;
                        xbdi.setEntry(i, ONE);
                        state = 100; break goto_trsbox;
                    }
                    final double ssq = power2(trialStepPoint.getEntry(i)) + power2(s.getEntry(i));
                    double temp2 = ssq - power2(trustRegionCenterOffset.getEntry(i) - lowerDifference.getEntry(i));
                    if (temp2 > ZERO) {
                        temp2 = JdkMath.sqrt(temp2) - s.getEntry(i);
                        if (angbd * temp2 > tempa) {
                            angbd = tempa / temp2;
                            iact = i;
                            xsav = MINUS_ONE;
                        }
                    }
                    double temp3 = ssq - power2(upperDifference.getEntry(i) - trustRegionCenterOffset.getEntry(i));
                    if (temp3 > ZERO) {
                        temp3 = JdkMath.sqrt(temp3) + s.getEntry(i);
                        if (angbd * temp3 > tempb) {
                            angbd = tempb / temp3;
                            iact = i;
                            xsav = ONE;
                        }
                    }
                }
            }

            // Calculate HHD and some curvatures for the alternative iteration.

            state = 210; break;
        }
        case 150: {
            double shs = ZERO;
            double dhs = ZERO;
            double dhd = ZERO;
            for (int i = 0; i < dimension; i++) {
                if (xbdi.getEntry(i) == ZERO) {
                    shs += s.getEntry(i) * hs.getEntry(i);
                    dhs += trialStepPoint.getEntry(i) * hs.getEntry(i);
                    dhd += trialStepPoint.getEntry(i) * hred.getEntry(i);
                }
            }

            // Seek the greatest reduction in Q for a range of equally spaced values
            // of ANGT in [0,ANGBD], where ANGT is the tangent of half the angle of
            // the alternative iteration.

            double redmax = ZERO;
            int isav = -1;
            double redsav = ZERO;
            final int iu = (int) (angbd * 17. + 3.1);
            double rdprev = 0;
            double rdnext = 0;
            double angt = 0;
            for (int i = 0; i < iu; i++) {
                angt = angbd * i / iu;
                final double sth = (angt + angt) / (ONE + angt * angt);
                final double temp = shs + angt * (angt * dhd - dhs - dhs);
                final double rednew = sth * (angt * dredg - sredg - HALF * sth * temp);
                if (rednew > redmax) {
                    redmax = rednew;
                    isav = i;
                    rdprev = redsav;
                } else if (i == isav + 1) {
                    rdnext = rednew;
                }
                redsav = rednew;
            }

            // Return if the reduction is zero. Otherwise, set the sine and cosine
            // of the angle of the alternative iteration, and calculate SDEC.

            if (isav < 0) {
                state = 190; break;
            }
            if (isav < iu) {
                final double temp = (rdnext - rdprev) / (redmax + redmax - rdprev - rdnext);
                angt = angbd * (isav + HALF * temp) / iu;
            }
            final double cth = (ONE - angt * angt) / (ONE + angt * angt);
            final double sth = (angt + angt) / (ONE + angt * angt);
            final double temp = shs + angt * (angt * dhd - dhs - dhs);
            final double sdec = sth * (angt * dredg - sredg - HALF * sth * temp);
            if (sdec <= ZERO) {
                state = 190; break;
            }

            // Update GNEW, D and HRED. If the angle of the alternative iteration
            // is restricted by a bound on a free variable, that variable is fixed
            // at the bound.

            dredg = ZERO;
            gredsq = ZERO;
            for (int i = 0; i < dimension; i++) {
                gnew.setEntry(i, gnew.getEntry(i) + (cth - ONE) * hred.getEntry(i) + sth * hs.getEntry(i));
                if (xbdi.getEntry(i) == ZERO) {
                    trialStepPoint.setEntry(i, cth * trialStepPoint.getEntry(i) + sth * s.getEntry(i));
                    dredg += trialStepPoint.getEntry(i) * gnew.getEntry(i);
                    gredsq += power2(gnew.getEntry(i));
                }
                hred.setEntry(i, cth * hred.getEntry(i) + sth * hs.getEntry(i));
            }
            qred += sdec;
            if (iact >= 0 && isav == iu) {
                ++nact;
                xbdi.setEntry(iact, xsav);
                state = 100; break;
            }

            // If SDEC is sufficiently small, then RETURN after setting XNEW to
            // XOPT+D, giving careful attention to the bounds.

            if (sdec > qred * .01) {
                state = 120; break;
            }
        }
        case 190: {
            setInPlace(newPoint, clipSelf(trustRegionCenterOffset.add(trialStepPoint), lowerDifference, upperDifference));
            for (int i = 0; i < dimension; i++) {
                if (xbdi.getEntry(i) == MINUS_ONE) {
                    newPoint.setEntry(i, lowerDifference.getEntry(i));
                }
                if (xbdi.getEntry(i) == ONE) {
                    newPoint.setEntry(i, upperDifference.getEntry(i));
                }
            }
            setInPlace(trialStepPoint, newPoint.subtract(trustRegionCenterOffset));
            final double dsq = getSquaredNorm(trialStepPoint);
            return new double[] { dsq, crvmin };
            // The following instructions multiply the current S-vector by the second
            // derivative matrix of the quadratic model, putting the product in HS.
            // They are reached from three different parts of the software above and
            // they can be regarded as an external subroutine.
        }
        case 210: {
            int ih = 0;
            for (int j = 0; j < dimension; j++) {
                hs.setEntry(j, ZERO);
                for (int i = 0; i <= j; i++) {
                    if (i < j) {
                        hs.setEntry(j, hs.getEntry(j) + modelSecondDerivativesValues.getEntry(ih) * s.getEntry(i));
                    }
                    hs.setEntry(i, hs.getEntry(i) + modelSecondDerivativesValues.getEntry(ih) * s.getEntry(j));
                    ih++;
                }
            }
            final RealVector tmp = interpolationPoints.operate(s).ebeMultiply(modelSecondDerivativesParameters);
            for (int k = 0; k < numberOfInterpolationPoints; k++) {
                if (modelSecondDerivativesParameters.getEntry(k) != ZERO) {
                    for (int i = 0; i < dimension; i++) {
                        hs.setEntry(i, hs.getEntry(i) + tmp.getEntry(k) * interpolationPoints.getEntry(k, i));
                    }
                }
            }
            if (crvmin != ZERO) {
                state = 50; break;
            }
            if (iterc > itcsav) {
                state = 150; break;
            }
            for (int i = 0; i < dimension; i++) {
                hred.setEntry(i, hs.getEntry(i));
            }
            state = 120; break;
        }
        default: {
            throw new MathIllegalStateException(LocalizedFormats.SIMPLE_MESSAGE, "trsbox");
        }}
        }
    } // trsbox

    // ----------------------------------------------------------------------------------------

    /**
     *     The arrays BMAT and ZMAT are updated, as required by the new position
     *     of the interpolation point that has the index KNEW. The vector VLAG has
     *     N+numberOfInterpolationPoints components, set on entry
     *     to the first numberOfInterpolationPoints and last N components
     *     of the product Hw in equation (4.11) of the Powell (2006) paper on
     *     NEWUOA. Further, BETA is set on entry to the value of the parameter
     *     with that name, and DENOM is set to the denominator of the updating
     *     formula. The first NDIM elements of W are used for working space.
     * @param beta
     * @param denom
     * @param kNew
     * @param startOfHw has NPT components, the first NPT components of the product Hw in equation (4.11) of the Powell (2006) paper on NEWUOA.
     * @param tailOfHw has N components, the last N components of the product Hw in equation (4.11) of the Powell (2006) paper on NEWUOA.
     */
    private void update(
            final double beta,
            final double denom,
            final int kNew,
            final RealVector startOfHw,
            final RealVector tailOfHw
    ) {

        // Elements of zMatrix that are <= zeroThreshold are replaced by zero. Important for numerical stability.
        final double zeroThreshold = maxAbs(zMatrix) * 1e-20;
        // Apply the rotations that put zeros in the KNEW-th row of ZMAT.
        for (int j = 1; j < zMatrix.getColumnDimension(); j++) {
            final double d1 = zMatrix.getEntry(kNew, j);
            if (JdkMath.abs(d1) > zeroThreshold) {
                final double d2 = power2(zMatrix.getEntry(kNew, 0));
                final double d3 = power2(zMatrix.getEntry(kNew, j));
                final double d4 = JdkMath.sqrt(d2 + d3);
                final double d5 = zMatrix.getEntry(kNew, 0) / d4;
                final double d6 = zMatrix.getEntry(kNew, j) / d4;
                for (int i = 0; i < numberOfInterpolationPoints; i++) {
                    final double d7 = d5 * zMatrix.getEntry(i, 0) + d6 * zMatrix.getEntry(i, j);
                    zMatrix.setEntry(i, j, d5 * zMatrix.getEntry(i, j) - d6 * zMatrix.getEntry(i, 0));
                    zMatrix.setEntry(i, 0, d7);
                }
            }
            zMatrix.setEntry(kNew, j, ZERO);
        }

        // Put the first numberOfInterpolationPoints components of the KNEW-th column of HLAG
        // into W, and calculate the parameters of the updating formula.
        final ArrayRealVector work = new ArrayRealVector(numberOfInterpolationPoints + dimension);
        for (int i = 0; i < numberOfInterpolationPoints; i++) {
            work.setEntry(i, zMatrix.getEntry(kNew, 0) * zMatrix.getEntry(i, 0));
        }

        final double alpha = work.getEntry(kNew);
        final double tau = startOfHw.getEntry(kNew);
        startOfHw.setEntry(kNew, startOfHw.getEntry(kNew) - ONE);

        // Complete the updating of ZMAT.

        final double sqrtDenom = JdkMath.sqrt(denom);
        final double d1 = tau / sqrtDenom;
        final double d2 = zMatrix.getEntry(kNew, 0) / sqrtDenom;
        for (int i = 0; i < numberOfInterpolationPoints; i++) {
            zMatrix.setEntry(i, 0,
                          d1 * zMatrix.getEntry(i, 0) - d2 * startOfHw.getEntry(i));
        }

        // Finally, update the matrix BMAT.

        for (int j = 0; j < dimension; j++) {
            final int jp = numberOfInterpolationPoints + j;
            work.setEntry(jp, bMatrix.getEntry(kNew, j));
            final double d3 = (alpha * tailOfHw.getEntry(j) - tau * work.getEntry(jp)) / denom;
            final double d4 = (-beta * work.getEntry(jp) - tau * tailOfHw.getEntry(j)) / denom;
            for (int i = 0; i <= jp; i++) {
                if (i < numberOfInterpolationPoints) {
                    bMatrix.setEntry(i, j,
                        bMatrix.getEntry(i, j) + d3 * startOfHw.getEntry(i) + d4 * work.getEntry(i));
                } else {
                    bMatrix.setEntry(i, j,
                        bMatrix.getEntry(i, j) + d3 * tailOfHw.getEntry(i - numberOfInterpolationPoints) + d4 * work.getEntry(
                            i));
                    bMatrix.setEntry(jp, (i - numberOfInterpolationPoints), bMatrix.getEntry(i, j));
                }
            }
        }
    } // update

    /**
     * Performs validity checks and initializes fields.
     */
    private void init() {
        lowerBounds = new ArrayRealVector(getLowerBound(), false);
        upperBounds = new ArrayRealVector(getUpperBound(), false);
        currentBest = new ArrayRealVector(getStartPoint());
        dimension = currentBest.getDimension();

        // Check problem dimension.
        if (dimension < MINIMUM_PROBLEM_DIMENSION) {
            throw new NumberIsTooSmallException(dimension, MINIMUM_PROBLEM_DIMENSION, true);
        }
        // Check number of interpolation points
        final int minInterpolationPoints = dimension + 2;
        final int maxInterpolationPoints = (dimension + 2) * (dimension + 1) / 2;
        if (numberOfInterpolationPoints < minInterpolationPoints ||
            numberOfInterpolationPoints > maxInterpolationPoints) {
            throw new OutOfRangeException(LocalizedFormats.NUMBER_OF_INTERPOLATION_POINTS,
                                          numberOfInterpolationPoints,
                                          minInterpolationPoints,
                                          maxInterpolationPoints);
        }

        // Initialize data structures used outside of this method
        trustRegionCenterOffset = new ArrayRealVector(dimension);
        newPoint = new ArrayRealVector(dimension);
        alternativeNewPoint = new ArrayRealVector(dimension);
        trialStepPoint = new ArrayRealVector(dimension);
        modelSecondDerivativesParameters = new ArrayRealVector(numberOfInterpolationPoints);

        // Initialize bound differences: differences between the upper and lower bounds.
        final RealVector boundDifference = upperBounds.subtract(lowerBounds);
        final double requiredMinDiff = 2 * initialTrustRegionRadius;
        final double minDiff = boundDifference.getMinValue();
        if (minDiff < requiredMinDiff) {
            initialTrustRegionRadius = minDiff / 3.0;
        }
        // Modify the initial starting point if necessary in order to avoid conflicts between the
        // bounds and the construction of the first quadratic model.
        for (int i = 0; i < dimension; i++) {
            final double xi = currentBest.getEntry(i);
            if (xi <= lowerBounds.getEntry(i)) {
                currentBest.setEntry(i, lowerBounds.getEntry(i));
            } else if (xi < lowerBounds.getEntry(i) + initialTrustRegionRadius) {
                currentBest.setEntry(i, lowerBounds.getEntry(i) + initialTrustRegionRadius);
            } else if (xi >= upperBounds.getEntry(i)) {
                currentBest.setEntry(i, upperBounds.getEntry(i));
            } else if (xi > upperBounds.getEntry(i) - initialTrustRegionRadius) {
                currentBest.setEntry(i, upperBounds.getEntry(i) - initialTrustRegionRadius);
            }
        }
        // The lower and upper bounds on moves from the updated X are set now, in the ISL and ISU
        // partitions of W, in order to provide useful and exact information about
        // components of X that become within distance RHOBEG from their bounds.
        lowerDifference = lowerBounds.subtract(currentBest);
        upperDifference = upperBounds.subtract(currentBest);

        initInterpolationPoints(dimension);
        initInterpolationMatrices(dimension);
        // init for numberOfInterpolationPoints > 2n+1 - not recommended - see abstract
        initAdditionalPoints(dimension);
    }

    private void initInterpolationPoints(final int dimension) {
        // Begin the initialization procedure. NF becomes one more than the number
        // of function values so far. The coordinates of the displacement of the
        // next initial interpolation point from XBASE are set
        // in interpolationPoints(NF+1,.).

        // Set originShift to the initial vector of variables
        // Prepare the interpolationPoints matrix as defined in (2.2)
        originShift = new ArrayRealVector(currentBest);
        interpolationPoints = new Array2DRowRealMatrix(numberOfInterpolationPoints, dimension);
        fAtInterpolationPoints = new ArrayRealVector(numberOfInterpolationPoints);
        // first row
        final double firstF = computeF(currentBest);
        trustRegionCenterInterpolationPointIndex = 0;
        fAtInterpolationPoints.setEntry(0, firstF);
        // prepare points as defined in (2.2), left column
        final int ruleAMax = Math.min(dimension, numberOfInterpolationPoints);
        for (int i = 0; i < ruleAMax; i++) {
            if (upperDifference.getEntry(i) == ZERO) {
                interpolationPoints.setEntry(i+1, i, -initialTrustRegionRadius);
            } else {
                interpolationPoints.setEntry(i+1, i, initialTrustRegionRadius);
            }
        }

        // prepare points as defined in (2.2), right column
        final int ruleBMax = Math.min(2 * dimension, numberOfInterpolationPoints);
        for (int i = dimension; i < ruleBMax; i++) {
            if (lowerDifference.getEntry(i - dimension) == ZERO) {
                final double stepB = JdkMath.min(TWO * initialTrustRegionRadius, upperDifference.getEntry(
                    i - dimension));
                interpolationPoints.setEntry(i + 1, i - dimension, stepB);
            } else if (upperDifference.getEntry(i - dimension) == ZERO) {
                final double stepB = JdkMath.max(-TWO * initialTrustRegionRadius, lowerDifference.getEntry(
                    i - dimension));
                interpolationPoints.setEntry(i + 1, i - dimension, stepB);
            } else {
                interpolationPoints.setEntry(i + 1, i - dimension, -initialTrustRegionRadius);
            }
        }

        // evaluate F for each point
        for (int j = 1; j <= ruleBMax; j++) {
            setInPlace(currentBest, clipSelf(originShift.add(interpolationPoints.getRowVectorRef(j)), lowerBounds, upperBounds));
            final double f = computeF(currentBest);

            fAtInterpolationPoints.setEntry(j, f);
            if (f < fAtInterpolationPoints.getEntry(trustRegionCenterInterpolationPointIndex)) {
                trustRegionCenterInterpolationPointIndex = j;
            }
        }

        // re-order points, with smaller values of F first, in order to bias such
        // that the smallest function value contributes to the
        // off-diagonal second derivative terms of the initial quadratic model.
        for (int i = dimension; i < ruleBMax; i++) {
            final int ruleBRowIdx = i + 1;
            final int ruleARowIdx = ruleBRowIdx - dimension;
            final double stepA = interpolationPoints.getEntry(ruleARowIdx, i - dimension);
            final double fAtA = fAtInterpolationPoints.getEntry(ruleARowIdx);
            final double stepB = interpolationPoints.getEntry(ruleBRowIdx, i - dimension);
            final double f = fAtInterpolationPoints.getEntry(ruleBRowIdx);
            if (stepA * stepB < ZERO && f < fAtA) {
                // swap the interpolation points
                interpolationPoints.setEntry(ruleARowIdx, i - dimension, stepB);
                interpolationPoints.setEntry(ruleBRowIdx, i - dimension, stepA);
                fAtInterpolationPoints.setEntry(ruleBRowIdx, fAtA);
                fAtInterpolationPoints.setEntry(ruleARowIdx, f);
                if (trustRegionCenterInterpolationPointIndex == ruleBRowIdx) {
                    trustRegionCenterInterpolationPointIndex = ruleARowIdx;
                }
            }
        }
    }

    private void initInterpolationMatrices(final int dimension) {
        // Set the nonzero initial elements of BMAT and the quadratic model in the
        // cases when NF is at most 2*N+1.
        modelSecondDerivativesValues = new ArrayRealVector(dimension * (dimension + 1) / 2);
        bMatrix = new Array2DRowRealMatrix(dimension + numberOfInterpolationPoints,
            dimension);
        zMatrix = new Array2DRowRealMatrix(numberOfInterpolationPoints,
            numberOfInterpolationPoints - dimension - 1);
        gradientAtTrustRegionCenter = new ArrayRealVector(dimension);
        final double rhosq = initialTrustRegionRadius * initialTrustRegionRadius;
        final double firstF = fAtInterpolationPoints.getEntry(0);
        for (int j = 1; j <= 2 * dimension; j++) {
            final double f = fAtInterpolationPoints.getEntry(j);
            if (j <= dimension) {
                double stepA = interpolationPoints.getEntry(j, j -1);
                gradientAtTrustRegionCenter.setEntry(j - 1, (f - firstF) / stepA);
                if (numberOfInterpolationPoints < j + dimension + 1) {
                    final double oneOverStepA = ONE / stepA;
                    bMatrix.setEntry(0, j - 1, -oneOverStepA);
                    bMatrix.setEntry(j, j - 1, oneOverStepA);
                    bMatrix.setEntry(numberOfInterpolationPoints + j - 1, j - 1, -HALF * rhosq);
                }
            } else if (j >= dimension + 1) {
                final double stepA = interpolationPoints.getEntry(j - dimension,
                    j - dimension - 1);
                final double stepB = interpolationPoints.getEntry(j, j - dimension - 1);
                final int ih = (j - dimension) * (j - dimension + 1) / 2 - 1;
                final double tmp = (f - firstF) / stepB;
                final double diff = stepB - stepA;
                modelSecondDerivativesValues.setEntry(ih, TWO * (tmp - gradientAtTrustRegionCenter.getEntry(
                    j - dimension - 1)) / diff);
                gradientAtTrustRegionCenter.setEntry(
                    j - dimension - 1, (gradientAtTrustRegionCenter.getEntry(j - dimension - 1) * stepB - tmp * stepA) / diff);
                bMatrix.setEntry(0, j - dimension - 1, -(stepA + stepB) / (stepA * stepB));
                bMatrix.setEntry(
                    j, j - dimension - 1, -HALF / interpolationPoints.getEntry(j - dimension,
                        j - dimension - 1));
                bMatrix.setEntry(j - dimension, j - dimension - 1,
                    -bMatrix.getEntry(0, j - dimension - 1) - bMatrix.getEntry(j,
                        j - dimension - 1));
                zMatrix.setEntry(0, j - dimension - 1, JdkMath.sqrt(TWO) / (stepA * stepB));
                zMatrix.setEntry(j, j - dimension - 1, JdkMath.sqrt(HALF) / rhosq);
                // zMatrix.setEntry(nfm, nfxm, JdkMath.sqrt(HALF) * recip); // XXX "testAckley" and "testDiffPow" fail.
                zMatrix.setEntry(j - dimension, j - dimension - 1,
                    -zMatrix.getEntry(0, j - dimension - 1) - zMatrix.getEntry(j,
                        j - dimension - 1));
            }
        }
    }

    private void initAdditionalPoints(final int dimension) {
        // add points following (2.3) and (2.4)
        final double rhosq = initialTrustRegionRadius * initialTrustRegionRadius;
        final double recip = 1d / rhosq;
        final double firstF = fAtInterpolationPoints.getEntry(0);
        for (int j = 2 * dimension + 1; j < numberOfInterpolationPoints; j++) {
            // prepare interpolation point following (2.3)
            final int tmp1 = (j - (dimension + 1)) / dimension;
            int jpt = j - tmp1 * dimension - dimension;
            int ipt = jpt + tmp1;
            if (ipt > dimension) {
                final int tmp2 = jpt;
                jpt = ipt - dimension;
                ipt = tmp2;
            }
            interpolationPoints.setEntry(j, ipt - 1, interpolationPoints.getEntry(ipt, ipt - 1));
            interpolationPoints.setEntry(j, jpt - 1, interpolationPoints.getEntry(jpt, jpt - 1));

            // Calculate the next value of F.
            setInPlace(currentBest, clipSelf(originShift.add(interpolationPoints.getRowVectorRef(j)), lowerBounds, upperBounds));
            final double f = computeF(currentBest);
            fAtInterpolationPoints.setEntry(j, f);
            if (f < fAtInterpolationPoints.getEntry(trustRegionCenterInterpolationPointIndex)) {
                trustRegionCenterInterpolationPointIndex = j;
            }

            // Set the off-diagonal second derivatives of the Lagrange functions and
            // the initial quadratic model.
            zMatrix.setEntry(0, j - dimension - 1, recip);
            zMatrix.setEntry(j, j - dimension - 1, recip);
            zMatrix.setEntry(ipt, j - dimension - 1, -recip);
            zMatrix.setEntry(jpt, j - dimension - 1, -recip);

            final int ih = ipt * (ipt - 1) / 2 + jpt - 1;
            final double tmp = interpolationPoints.getEntry(j, ipt - 1) * interpolationPoints.getEntry(
                j, jpt - 1);
            modelSecondDerivativesValues.setEntry(ih, (firstF - fAtInterpolationPoints.getEntry(ipt) - fAtInterpolationPoints.getEntry(jpt) + f) / tmp);
        }
    }

    private double computeF(final RealVector point) {
        final double objectiveValue = computeObjectiveValue(point.toArray());
        return getGoalType().equals(GoalType.MINIMIZE) ? objectiveValue : -objectiveValue;
    }

    private double power2(final double val) {
        // will replace by JdkMath only at the end of the refactoring because this may impact floating point errors
        return val * val;
    }

    // Vector and Matrix utils - should be in interface IMO
    private static double squaredL2Distance(final RealVector thiz, final RealVector b) {
        // bad performance - to optimize later
        final RealVector tmp = thiz.subtract(b);
        return tmp.dotProduct(tmp);
    }

    private static RealVector setInPlace(final RealVector thiz, final RealVector newValues) {
        for (int i = 0; i< newValues.getDimension(); i++) {
            thiz.setEntry(i, newValues.getEntry(i));
        }
        return thiz;
    }

    private static void setZeroInPlace(final RealVector thiz) {
        for (int i = 0; i< thiz.getDimension(); i++) {
            thiz.setEntry(i, 0);
        }
    }

    private static void addInPlace(final RealVector thiz, final RealVector addValues) {
        for (int i = 0; i< addValues.getDimension(); i++) {
            thiz.setEntry(i, thiz.getEntry(i) + addValues.getEntry(i));
        }
    }

    private static void subtractInPlace(final RealVector thiz, final RealVector subtractValues) {
        for (int i = 0; i < subtractValues.getDimension(); i++) {
            thiz.setEntry(i, thiz.getEntry(i) - subtractValues.getEntry(i));
        }
    }


    // subtract the vector to each row
    private static RealMatrix subtractInPlace(final RealMatrix thiz, final RealVector subtractValues) {
        for (int i = 0; i < thiz.getColumnDimension(); i++) {
            for (int k=0; k<thiz.getRowDimension(); k++) {
                thiz.setEntry(k, i,thiz.getEntry(k, i) - subtractValues.getEntry(i));
            }
        }
        return thiz;
    }

    private static RealMatrix addToColInPlace(final RealMatrix thiz, final int colIndex, final RealVector subtractValues) {
        for (int k=0; k<thiz.getRowDimension(); k++) {
            thiz.setEntry(k, colIndex,thiz.getEntry(k, colIndex) + subtractValues.getEntry(k));
        }
        return thiz;
    }

    private static void setRow(final RealMatrix thiz, final int rowIndex, final RealVector newValues) {
        for (int i = 0; i< newValues.getDimension(); i++) {
            thiz.setEntry(rowIndex, i, newValues.getEntry(i));
        }
    }

    private static RealVector clipSelf(final RealVector thiz, final RealVector lowerBounds, final RealVector upperBounds) {
        for (int i = 0; i<thiz.getDimension(); i++) {
            thiz.setEntry(i, clip(thiz.getEntry(i), lowerBounds.getEntry(i), upperBounds.getEntry(i)));
        }
        return thiz;
    }

    private static double clip(final double value, final double lowerBound, final double upperBound) {
        return JdkMath.min(upperBound, JdkMath.max(lowerBound, value));
    }

    public static double getSquaredNorm(final RealVector thiz) {
        return thiz.dotProduct(thiz);
    }


    public static double getSum(final RealVector thiz) {
        double sum = 0;
        for (int i = 0; i < thiz.getDimension(); i++) {
            sum += thiz.getEntry(i);
        }
        return sum;
    }

    // return the maximum absolute value in a matrix
    private static double maxAbs(final RealMatrix mat) {
        double max = 0;
        for (int k = 0; k < mat.getRowDimension(); k++) {
            for (int j = 0; j < mat.getColumnDimension(); j++) {
                max = JdkMath.max(max, JdkMath.abs(mat.getEntry(k, j)));
            }
        }
        return max;
    }
}
//CHECKSTYLE: resume all
