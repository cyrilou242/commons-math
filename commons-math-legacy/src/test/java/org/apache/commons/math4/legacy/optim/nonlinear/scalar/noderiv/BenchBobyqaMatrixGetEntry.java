package org.apache.commons.math4.legacy.optim.nonlinear.scalar.noderiv;

import org.apache.commons.math4.legacy.analysis.MultivariateFunction;
import org.apache.commons.math4.legacy.optim.InitialGuess;
import org.apache.commons.math4.legacy.optim.MaxEval;
import org.apache.commons.math4.legacy.optim.SimpleBounds;
import org.apache.commons.math4.legacy.optim.nonlinear.scalar.GoalType;
import org.apache.commons.math4.legacy.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math4.legacy.optim.nonlinear.scalar.TestFunction;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.infra.Blackhole;

public class BenchBobyqaMatrixGetEntry {

    @State(Scope.Benchmark)
    public static class ExecPlan {
        private final int dim = 12;
        private final double[] startPoint = OptimTestUtils.point(dim, 0.1);
        private final MultivariateFunction func = TestFunction.ROSENBROCK.withDimension(dim);
        private final int dim1 = startPoint.length;
        private final int numIterpolationPoints = 2 * dim1 + 1;
        private final BOBYQAOptimizer optimizer = new BOBYQAOptimizer(numIterpolationPoints);
    }

    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    public void bench(ExecPlan plan, Blackhole blackhole) {
        blackhole.consume(plan.optimizer.optimize(new MaxEval(3000),
          new ObjectiveFunction(plan.func),
          GoalType.MINIMIZE,
          SimpleBounds.unbounded(plan.dim1),
          new InitialGuess(plan.startPoint)));
    }
}
