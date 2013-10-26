package org.dataworm.optimization;

/**
 * It's based the paper "Scalable Training of L1-Regularized Log-Linear Models"
 * by Galen Andrew and Jianfeng Gao
 * Created with IntelliJ IDEA.
 * User: weixuan
 * Date: 10/26/13
 * Time: 5:29 PM
 * To change this template use File | Settings | File Templates.
 */


public class LBFGSMinimizer extends GradientMinimizer implements QuasiNewtonHelper.Minimizer {
    // config
    final int numHist = 10;
    final int maxIteration = 200;

    DiffFunction func;

    // current and state
    double[] curPos;
    double[] curGrad;
    double curCost;

    @Override
    public double[] minimize(DiffFunction func, double[] init, TerminationCriterion criterion) {
        this.func = func;

        this.curPos = init;
        this.curGrad = func.getGradient(curPos).clone();
        this.curCost = evaluate(curPos);
        criterion.addCost(curCost);

        QuasiNewtonHelper helper = new QuasiNewtonHelper(this, numHist, curPos, curGrad);
        System.err.println("Iter\tcost\timprovement");
        System.err.printf("%d\t%e\tUndefined\n", 0, curCost);
        for (int iter=1; iter <= maxIteration; ++iter) {
            double[] dir = getUpdateDir(curGrad, helper);
            double[] newPos = new double[curPos.length];
            double cost = helper.backTrackingLineSearch(curPos, curGrad, curCost, dir, iter==1, newPos);
            if(cost == curCost)
                break;
            curCost = cost;
            curPos = newPos;
            criterion.addCost(cost);
            System.err.printf("%d\t%e\t%e\n", iter, cost, criterion.getImprovement());
            if(criterion.isTerminable())
                break;
            curGrad = func.getGradient(newPos).clone();
            if(helper.updateState(curPos, curGrad))
                break;
        }
        return curPos;
    }

    @Override
    public double evaluate(double[] x) {
        return func.getValue(x);
    }
    @Override
    public double getGradDirProduct(double[] x, double[] grad, double[] dir) {
        return getProduct(dir, grad);
    }
    @Override
    public double[] getNextPoint(double[] x, double[] grad, double[] dir, double alpha) {
        double[] y = x.clone();
        addMult(y, dir, alpha);
        return y;
    }

    double[] getSteepestDescDir(double[] grad) {
        double[] dir = grad.clone();
        mult(dir, -1);
        return dir;
    }

    double[] getUpdateDir(double[] grad, QuasiNewtonHelper helper) {
        double[] dir = getSteepestDescDir(grad);
        helper.applyQuasiInverseHession(dir);
        return dir;
    }

    void mult(double[] y, double scalar) {
        for(int n=0; n<y.length; ++n) {
            y[n] *= scalar;
        }
    }
    void addMult(double[] y, double[] x, double scalar) {
        for(int n=0; n<x.length; ++n) {
            y[n] += x[n] * scalar;
        }
    }
    void copy(double[] y, double[] x) {
        for(int n=0; n<x.length; ++n) {
            y[n] = x[n];
        }
    }
    double getProduct(double[] x, double[] y) {
        double prod = 0;
        for(int n=0; n<x.length; ++n)
            prod += x[n] * y[n];
        return prod;
    }
}