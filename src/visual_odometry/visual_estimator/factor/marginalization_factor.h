#pragma once

#include <ros/ros.h>
#include <ros/console.h>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>

#include "../utility/utility.h"
#include "../utility/tic_toc.h"

const int NUM_THREADS = 4;

struct ResidualBlockInfo
{
    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
        : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    //用于计算当前时刻的残差和雅克比
    void Evaluate();

    //各类残差的factor
    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;
    //残差块
    std::vector<double *> parameter_blocks;
    //丢弃的参数块的索引
    std::vector<int> drop_set;

    double **raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    Eigen::VectorXd residuals;

    int localSize(int size)
    {
        return size == 7 ? 6 : size;
    }
};

struct ThreadsStruct
{
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size; //global size
    std::unordered_map<long, int> parameter_block_idx; //local size
};

class MarginalizationInfo
{
  public:
    ~MarginalizationInfo(); // 析构函数
    int localSize(int size) const;
    int globalSize(int size) const;
    //添加残差块的相关信息
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);
    //计算每个残差所对应的雅克比，并更新block块内信息
    void preMarginalize();
    //求解
    void marginalize();
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    //所有的观测项
    std::vector<ResidualBlockInfo *> factors;
    //m是要边缘化的变量个数，n为要保留下来的变量个数
    //m表示marg变量/parameter_block_idx的总localSize
    //n表示保留变量的总localSize
    int m, n;
    //每个优化变量块的大小
    std::unordered_map<long, int> parameter_block_size; //global size
    int sum_block_size;
    //被merge掉的变量
    std::unordered_map<long, int> parameter_block_idx; //local size
    //每个优化块的数据
    std::unordered_map<long, double *> parameter_block_data;

    //他们的key都是long类型
    //而value分别是 各个优化变量的长度，各个优化变量在id以各个优化变量对应的double指针类型的数据。
    std::vector<int> keep_block_size; //global size
    std::vector<int> keep_block_idx;  //local size
    std::vector<double *> keep_block_data;

    //分别指的是边缘化之后从矩阵当中恢复出来的雅可比矩阵和残差矩阵
    Eigen::MatrixXd linearized_jacobians;
    Eigen::VectorXd linearized_residuals;
    const double eps = 1e-8;

};

class MarginalizationFactor : public ceres::CostFunction
{
  public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo* marginalization_info;
};
