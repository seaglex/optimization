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

public abstract class TerminationCriterion {
    public abstract void addCost(double latestCost);
    public abstract double getImprovement();
    public abstract boolean isTerminable();
}

class RelativeMeanImprovementCriterion extends TerminationCriterion {
    int minHist;
    int maxHist;
    LinkedList<Double> costList;
    double improvement;
    double tolerance;
    public RelativeMeanImprovementCriterion(double tolerance) {
        this.minHist = 5;
        this.maxHist = 10;
        this.costList = new LinkedList<Double>();
        this.tolerance = tolerance;
    }

    @Override
    public void addCost(double latestCost) {
        this.costList.offer(latestCost);
        improvement = calImprovement();
    }

    @Override
    public double getImprovement() {
        return improvement;
    }

    @Override
    public boolean isTerminable() {
        return improvement <= tolerance;
    }

    double calImprovement() {
        if (costList.size() <= minHist)
            return Double.POSITIVE_INFINITY;
        double improvement = (costList.peekFirst() - costList.peekLast()) / (costList.size()-1);
        if (costList.peekLast() != 0)
            improvement = Math.abs(improvement/costList.peekLast());
        else if (costList.peekFirst() != 0)
            improvement = Math.abs(improvement/costList.peekFirst());
        else
            improvement = 0;
        if (costList.size() > maxHist)
            costList.poll();
        return improvement;
    }
}