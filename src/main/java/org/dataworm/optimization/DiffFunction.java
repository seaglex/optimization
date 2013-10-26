/**
 * It's based the paper "Scalable Training of L1-Regularized Log-Linear Models"
 * by Galen Andrew and Jianfeng Gao
 * Created with IntelliJ IDEA.
 * User: weixuan
 * Date: 10/26/13
 * Time: 5:29 PM
 * To change this template use File | Settings | File Templates.
 */
package org.dataworm.optimization;

public interface DiffFunction {
    double getValue(double[] x);
    double[] getGradient(double[] x);
}
