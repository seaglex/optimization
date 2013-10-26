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

public class OWLQNMinimizer extends GradientMinimizer implements QuasiNewtonHelper.Minimizer{
    // config
    final int numHist = 10;
    final int maxIteration = 200;
    double l1reg;

    // func
    DiffFunction func;

    // current and state
    double[] curPos;
    double[] curGrad;
    double curCost;

    public OWLQNMinimizer(double l1reg) {
        this.l1reg = l1reg;
    }

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
            double[] dir = getUpdateDir(curPos, curGrad, helper);
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
        double val = this.func.getValue(x);
        for(int n=0; n<x.length; ++n)
            val += Math.abs(x[n]) * this.l1reg;
        return val;
    }
    @Override
    public double getGradDirProduct(double[] x, double[] grad, double[] dir) {
        if(l1reg == 0)
            return getProduct(grad, dir);
        double val = 0;
        for(int n=0; n<grad.length; ++n) {
            if(x[n] < 0)
                val += dir[n] * (grad[n] - l1reg);
            else if (x[n] < 0)
                val += dir[n] * (grad[n] + l1reg);
            else if(dir[n] < 0)
                val += dir[n] * (grad[n] - l1reg);
            else if(dir[n] > 0)
                val += dir[n] * (grad[n] + l1reg);
        }
        return val;
    }
    @Override
    public double[] getNextPoint(double[] x, double[] grad, double[] dir, double alpha) {
        double[] nextX = new double[x.length];
        copy(nextX, x);
        addMult(nextX, dir, alpha);
        if (l1reg > 0) {
            for(int n=0; n<x.length; ++n) {
                if (x[n] * nextX[n] < 0)
                    nextX[n] = 0;
            }
        }
        return nextX;
    }

    double[] getUpdateDir(double[]x, double[] grad, QuasiNewtonHelper helper) {
        double[] steepestDescDir = getSteepestDescDir(x, grad);
        double[] dir = steepestDescDir.clone();
        helper.applyQuasiInverseHession(dir);
        fixDirSigns(dir, steepestDescDir);
        return dir;
    }

    double[] getSteepestDescDir(double[] x, double[] grad) {
        double[] dir = null;
        if(this.l1reg == 0) {
            dir = grad.clone();
            mult(dir, -1);
            return dir;
        }
        dir = new double[x.length];
        for(int n=0; n<grad.length; ++n) {
            if(x[n] < 0)
                dir[n] = -(grad[n] - this.l1reg);
            else if(x[n] > 0)
                dir[n] = -(grad[n] + this.l1reg);
            else {
                if (grad[n] < -l1reg)
                    dir[n] = -(grad[n] + l1reg);
                else if(grad[n] > l1reg)
                    dir[n] = -(grad[n] - l1reg);
                else
                    dir[n] = 0;
            }
        }
        return dir;
    }
    // the sign should be fixed, so that there are more zeros
    void fixDirSigns(double[] dir, double[] steepestDescDir) {
        if(l1reg == 0)
            return;
        for(int n=0; n<dir.length; ++n) {
            if(dir[n] * steepestDescDir[n] <= 0)
                dir[n] = 0;
        }
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