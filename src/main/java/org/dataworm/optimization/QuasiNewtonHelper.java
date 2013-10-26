package org.dataworm.optimization;

import java.util.LinkedList;

/**
 * It's based the paper "Scalable Training of L1-Regularized Log-Linear Models"
 * by Galen Andrew and Jianfeng Gao
 * Created with IntelliJ IDEA.
 * User: weixuan
 * Date: 10/26/13
 * Time: 5:29 PM
 * To change this template use File | Settings | File Templates.
 */

public class QuasiNewtonHelper {
    public static interface Minimizer {
        double evaluate(double[] x);
        double[] getNextPoint(double[] x, double[] grad, double[] dir, double alpha);
        double getGradDirProduct(double[] x, double[] grad, double[] dir);
    }
    // config
    final int MAX_BACKTRACKING_ITER = 50;
    int numHist;

    // historical data
    LinkedList<double[]> sList = new LinkedList<double[]>();
    LinkedList<double[]> yList = new LinkedList<double[]>();
    LinkedList<Double> roList = new LinkedList<Double>();
    double[] curPos;
    double[] curGrad;

    // the cost function
    Minimizer minimizer;

    public QuasiNewtonHelper(Minimizer minimizer, int numHist, double[] curPos, double[] curGrad) {
        this.minimizer = minimizer;
        this.numHist = numHist;
        this.curPos = curPos.clone();
        this.curGrad = curGrad.clone();
    }

    public void applyQuasiInverseHession(double[] dir) {
        int count = sList.size();
        if(count == 0)
            return;
        double[] alphas = new double[count];
        for(int n=count-1; n>=0; --n) {
            alphas[n] = -getProduct(sList.get(n), dir) / roList.get(n);
            addMult(dir, yList.get(n), alphas[n]);
        }
        double[] lastY = yList.get(count-1);
        double yDotY = getProduct(lastY, lastY);
        double scalar = roList.get(count-1) / yDotY;
        mult(dir, scalar);

        for(int n=0; n<count; ++n) {
            double beta = getProduct(yList.get(n), dir) / roList.get(n);
            addMult(dir, sList.get(n), -alphas[n] - beta);
        }
    }

    // return old cost if it's already optimal
    public double backTrackingLineSearch(double[] curPos, double[] curGrad, double curCost,
                                         double[] dir, boolean isFirst,
                                         double[] nextPos) {
        double orgDirDerivative = this.minimizer.getGradDirProduct(curPos, curGrad, dir);
        if(orgDirDerivative == 0) {
            copy(nextPos, curPos);
            return curCost;
        }
        if(orgDirDerivative > 0)
            throw new RuntimeException("BackTracking: to the opposite direction of grad");

        double alpha = 1.0;
        double backoff = 0.5;
        if(isFirst) {
            double normDir = Math.sqrt(getProduct(dir, dir));
            alpha = (1/normDir);
            backoff = 0.1;
        }

        double nextCost = 0;
        double c1 = 1e-4;
        double[] tmpPos = null;
        for (int cntItr=0; cntItr <= MAX_BACKTRACKING_ITER ; ++cntItr) {
            tmpPos = this.minimizer.getNextPoint(curPos, curGrad, dir, alpha);
            nextCost = this.minimizer.evaluate(tmpPos);
            if (nextCost <= curCost + c1 * orgDirDerivative * alpha)
                break;
            alpha *= backoff;
        }
        copy(nextPos, tmpPos);
        return nextCost;
    }

    // return true if it's already optimal
    public boolean updateState(double[] nextPos, double[] nextGrad) {
        double[] newS = null;
        double[] newY = null;
        if(sList.size() < numHist) {
            newS = new double[nextPos.length];
            newY = new double[nextPos.length];
        }
        else {
            newS = sList.poll();  // reuse the same memory
            newY = yList.poll();  // to reduce memory release & allocation
            roList.poll();
        }
        copy(newS, nextPos);
        addMult(newS, curPos, -1);
        copy(newY, nextGrad);
        addMult(newY, curGrad, -1);

        double ro = getProduct(newS, newY);

        sList.offer(newS);
        yList.offer(newY);
        roList.offer(ro);
        // swap
        curPos = nextPos;
        curGrad = nextGrad;
        if (ro == 0)
            return true;
        return false;
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
