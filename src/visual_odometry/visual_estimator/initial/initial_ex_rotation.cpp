#include "initial_ex_rotation.h"

InitialEXRotation::InitialEXRotation(){
    frame_count = 0;
    Rc.push_back(Matrix3d::Identity());
    Rc_g.push_back(Matrix3d::Identity());
    Rimu.push_back(Matrix3d::Identity());
    ric = Matrix3d::Identity();
}

//函数参数说明
//corres:两帧之间匹配的特征点，归一化坐标，构建本质矩阵E，分解得到两帧之间的旋转矩阵
//solveRelativeR(corres)函数实现，也就是开头提到的通过相机获得的旋转量

//delta_q_imu：为相邻两时刻 IMU预积分的旋转量，四元数表示传入进来的 匹配的归一化坐标，
//imu预积分旋转量，会先保存在vector中，因为需要多对信息，构建超定方程矩阵A

//calib_ric_result：为待求的外参数
bool InitialEXRotation::CalibrationExRotation(vector<pair<Vector3d, Vector3d>> corres, Quaterniond delta_q_imu, Matrix3d &calib_ric_result)
{
    frame_count++;
    //利用下面的函数计算得到旋转量1
    Rc.push_back(solveRelativeR(corres));
    //将原来的四元数转化成了旋转矩阵，方便后面的计算
    Rimu.push_back(delta_q_imu.toRotationMatrix());
    //每帧IMU相对于起始帧IMU的R，初始化为IMU预积分
    Rc_g.push_back(ric.inverse() * delta_q_imu * ric);

    //构建公式当中出现的矩阵
    Eigen::MatrixXd A(frame_count * 4, 4);
    A.setZero();
    int sum_ok = 0;
    for (int i = 1; i <= frame_count; i++)
    {
        Quaterniond r1(Rc[i]);
        Quaterniond r2(Rc_g[i]);

        // angular_distance就是计算两个坐标系之间相对旋转矩阵在做轴角变换后(u*theta)
        // 的角度theta，theta越小说明两个坐标系的姿态越接近，这个角度距离用于后面计算
        // 权重，这里计算权重就是为了降低外点的干扰，意思就是为了防止出现误差非常大的
        // R_bk+1^bk和R_ck+1^ck约束导致估计的结果偏差太大。
        double angular_distance = 180 / M_PI * r1.angularDistance(r2);
        ROS_DEBUG(
            "%d %f", i, angular_distance);

        //核函数 加权
        double huber = angular_distance > 5.0 ? 5.0 / angular_distance : 1.0;
        ++sum_ok;
        Matrix4d L, R;

        //公式做计算
        double w = Quaterniond(Rc[i]).w();
        Vector3d q = Quaterniond(Rc[i]).vec();
        L.block<3, 3>(0, 0) = w * Matrix3d::Identity() + Utility::skewSymmetric(q);
        L.block<3, 1>(0, 3) = q;
        L.block<1, 3>(3, 0) = -q.transpose();
        L(3, 3) = w;

        Quaterniond R_ij(Rimu[i]);
        w = R_ij.w();
        q = R_ij.vec();
        R.block<3, 3>(0, 0) = w * Matrix3d::Identity() - Utility::skewSymmetric(q);
        R.block<3, 1>(0, 3) = q;
        R.block<1, 3>(3, 0) = -q.transpose();
        R(3, 3) = w;

        A.block<4, 4>((i - 1) * 4, 0) = huber * (L - R);
    }

    //进行SVD分解，求解
    JacobiSVD<MatrixXd> svd(A, ComputeFullU | ComputeFullV);
    Matrix<double, 4, 1> x = svd.matrixV().col(3);
    Quaterniond estimated_R(x);
    ric = estimated_R.toRotationMatrix().inverse();
    //cout << svd.singularValues().transpose() << endl;
    //cout << ric << endl;
    Vector3d ric_cov;
    ric_cov = svd.singularValues().tail<3>();
    //迭代次数大于WINDOW_SIZE，且第二次的奇异值要大于0.25
    if (frame_count >= WINDOW_SIZE && ric_cov(1) > 0.25)
    {
        calib_ric_result = ric;
        return true;
    }
    else
        return false;
}

//这个函数是用来处理对极约束的，传入的是图像
Matrix3d InitialEXRotation::solveRelativeR(const vector<pair<Vector3d, Vector3d>> &corres)
{
    //如果像素点，或者说特征点过少的话 是没办法计算的，直接返回单位矩阵
    if (corres.size() >= 9)
    {
        //给出两个容器来接受帧
        vector<cv::Point2f> ll, rr;
        for (int i = 0; i < int(corres.size()); i++)
        {
            ll.push_back(cv::Point2f(corres[i].first(0), corres[i].first(1)));
            rr.push_back(cv::Point2f(corres[i].second(0), corres[i].second(1)));
        }
        //利用OpenCV来处理两个帧，计算得出本质矩阵
        cv::Mat E = cv::findFundamentalMat(ll, rr);
        cv::Mat_<double> R1, R2, t1, t2;
        decomposeE(E, R1, R2, t1, t2);

        //行列式如果是负数的话要变成正的算
        if (determinant(R1) + 1.0 < 1e-09)
        {
            E = -E;
            decomposeE(E, R1, R2, t1, t2);
        }
        //三角化
        double ratio1 = max(testTriangulation(ll, rr, R1, t1), testTriangulation(ll, rr, R1, t2));
        double ratio2 = max(testTriangulation(ll, rr, R2, t1), testTriangulation(ll, rr, R2, t2));
        cv::Mat_<double> ans_R_cv = ratio1 > ratio2 ? R1 : R2;
        //转置矩阵
        Matrix3d ans_R_eigen;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                ans_R_eigen(j, i) = ans_R_cv(i, j);
        return ans_R_eigen;
    }
    return Matrix3d::Identity();
}

double InitialEXRotation::testTriangulation(const vector<cv::Point2f> &l,
                                          const vector<cv::Point2f> &r,
                                          cv::Mat_<double> R, cv::Mat_<double> t)
{
    cv::Mat pointcloud;
    cv::Matx34f P = cv::Matx34f(1, 0, 0, 0,
                                0, 1, 0, 0,
                                0, 0, 1, 0);
    cv::Matx34f P1 = cv::Matx34f(R(0, 0), R(0, 1), R(0, 2), t(0),
                                 R(1, 0), R(1, 1), R(1, 2), t(1),
                                 R(2, 0), R(2, 1), R(2, 2), t(2));
    cv::triangulatePoints(P, P1, l, r, pointcloud);
    int front_count = 0;
    for (int i = 0; i < pointcloud.cols; i++)
    {
        double normal_factor = pointcloud.col(i).at<float>(3);

        cv::Mat_<double> p_3d_l = cv::Mat(P) * (pointcloud.col(i) / normal_factor);
        cv::Mat_<double> p_3d_r = cv::Mat(P1) * (pointcloud.col(i) / normal_factor);
        if (p_3d_l(2) > 0 && p_3d_r(2) > 0)
            front_count++;
    }
    ROS_DEBUG("MotionEstimator: %f", 1.0 * front_count / pointcloud.cols);
    return 1.0 * front_count / pointcloud.cols;
}

//这个函数用来做SVD分解的
void InitialEXRotation::decomposeE(cv::Mat E,
                                 cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                                 cv::Mat_<double> &t1, cv::Mat_<double> &t2)
{
    cv::SVD svd(E, cv::SVD::MODIFY_A);
    cv::Matx33d W(0, -1, 0,
                  1, 0, 0,
                  0, 0, 1);
    cv::Matx33d Wt(0, 1, 0,
                   -1, 0, 0,
                   0, 0, 1);
    R1 = svd.u * cv::Mat(W) * svd.vt;
    R2 = svd.u * cv::Mat(Wt) * svd.vt;
    t1 = svd.u.col(2);
    t2 = -svd.u.col(2);
}

//pnp是对于3D-2D图像的算法，而对极约束是2d-2d的算法，用于在已知相机的坐标的前提下，
//用两组图像的多组2d像素点来估计相机的运动。

// 两幅视图存在两个关系：第一种，通过对极几何，一幅图像上的点可以确定另外一幅图像上的一条直线；
// 另外一种，通过上一种映射，一幅图像上的点可以确定另外一幅图像上的点，
// 这个点是第一幅图像通过光心和图像点的射线与一个平面的交点在第二幅图像上的影像。
// 第一种情况可以用基本矩阵来表示，第二种情况则用单应矩阵来表示。
// 而本质矩阵则是基本矩阵的一种特殊情况，是在归一化图像坐标系下的基本矩阵。