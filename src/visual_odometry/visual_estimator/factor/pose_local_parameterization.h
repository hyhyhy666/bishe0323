#pragma once

//拓展了ceres solver中关于参数的处理
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include "../utility/utility.h"

class PoseLocalParameterization : public ceres::LocalParameterization
{
    //平移和旋转的假发实现，旋转的加法需要处理四元数的运算。这里的加法是广义加法，目的是在于实现优化变量的更新
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
    //计算雅克比矩阵
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;
    //表示参数的自由度
    virtual int GlobalSize() const { return 7; };
    //表示delta_x所在的正切空间的自由度
    virtual int LocalSize() const { return 6; };
};
