package org.dataworm.optimization;

/**
 * It's based the paper "Scalable Training of L1-Regularized Log-Linear Models"
 * by Galen Andrew and Jianfeng Gao
 * Created with IntelliJ IDEA.
 * User: weixuan
 * Date: 10/26/13
 * Time: 5:53 PM
 * To change this template use File | Settings | File Templates.
 */
public abstract class GradientMinimizer {
    public abstract double[] minimize(DiffFunction func, double[] init, TerminationCriterion criterion);
}
