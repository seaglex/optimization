package org.dataworm.optimization;

/**
 * Created with IntelliJ IDEA.
 * User: weixuan
 * Date: 10/26/13
 * Time: 5:44 PM
 * To change this template use File | Settings | File Templates.
 */

import org.junit.*;

public class TestMinimizer {
    static class MSECostFunction implements DiffFunction {
        double[] center;
        double[] weights;
        double[] grad;
        double[] init;
        public MSECostFunction(double[] center, double[] weights, double[] init) {
            this.center = center;
            this.grad = new double[center.length];
            this.weights = weights;
            this.init = init;
        }

        public double[] getInit() {
            return init;
        }
        public void test(double[] result, double tolerance) {
            for(int n=0; n<center.length; ++n) {
                assert (Math.abs(result[n] - center[n]) <= 1e-6);
            }
            System.out.println("Index\tTrue\tResult");
            for(int n=0; n<center.length; ++n) {
                System.out.printf("%d\t%e\t%e\n", n, center[n], result[n]);
            }
        }
        @Override
        public double getValue(double[] x) {
            double val = 0;
            for (int n=0; n<x.length; ++n) {
                val += weights[n] * (x[n] - center[n]) * (x[n] - center[n]);
            }
            return 0.5 * val;
        }

        @Override
        public double[] getGradient(double[] x) {
            for(int n=0; n<x.length; ++n) {
                grad[n] = weights[n] * (x[n] - center[n]);
            }
            return grad;
        }
    }

    @Test
    public void TestLBGS() {
        GradientMinimizer minimizer = new LBFGSMinimizer();
        TerminationCriterion criterion = new RelativeMeanImprovementCriterion(1e-4);
        MSECostFunction func = getMSECostFunc();

        double[] result = minimizer.minimize(func, func.getInit(), criterion);
        func.test(result, 1e-6);
    }

    @Test
    public void TestOWLQN() {
    GradientMinimizer minimizer = new OWLQNMinimizer(1e-6);
        TerminationCriterion criterion = new RelativeMeanImprovementCriterion(1e-4);
        MSECostFunction func = getMSECostFunc();

        double[] result = minimizer.minimize(func, func.getInit(), criterion);
        func.test(result, 0);
    }

    public MSECostFunction getMSECostFunc() {
        double[] center = new double[2];
        double[] init = new double[center.length];
        double[] weights = new double[center.length];
        for(int n=0; n<center.length; ++n) {
            center[n] = 0;
            init[n] = 1;
            weights[n] = Math.pow(0.01, n);
        }
        return new MSECostFunction(center, weights, init);
    }
}
