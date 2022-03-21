#include "utility.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

//GTSAM是用于计算机视觉和多传感器融合方面用于平滑和建图的C++库，采用因子图和贝叶斯网络的方式最大化后验概率

using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y) 姿态
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot) 速度 这个dot表示导数的意思
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz) 偏差

class IMUPreintegration : public ParamServer
{
public:

    //输入
    ros::Subscriber subImu; // 订阅器
    ros::Subscriber subOdometry;
    //输出
    ros::Publisher pubImuOdometry; // 发布器
    ros::Publisher pubImuPath;

    // map -> odom
    tf::Transform map_to_odom;
    tf::TransformBroadcaster tfMap2Odom;
    // odom -> base_link
    tf::TransformBroadcaster tfOdom2BaseLink;

    bool systemInitialized = false;

    //噪声协方差
    gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
    gtsam::Vector noiseModelBetweenBias;


    //预积分
    //负责预积分两个激光里程计之间的imu数据，作为约束加入因子图，并且优化出bias
    gtsam::PreintegratedImuMeasurements *imuIntegratorOpt_;
    //用来根据新的激光里程计到达后已经优化好的bias，预测从当前帧开始，下一帧激光里程计到达之前的imu里程计增量
    gtsam::PreintegratedImuMeasurements *imuIntegratorImu_;

    //给imuIntegratorPot_提供数据来源，不要的就弹出（从队头开始出发，比当前激光里程计数据早的imu通通积分，用一个仍一个）
    std::deque<sensor_msgs::Imu> imuQueOpt;
    //给imuIntegratorImu_提供数据来源，不要的就弹出（弹出当前激光里程计之前的imu数据，预计分用完一个弹一个）
    std::deque<sensor_msgs::Imu> imuQueImu;

    //imu因子图优化过程中的状态变量
    gtsam::Pose3 prevPose_;
    gtsam::Vector3 prevVel_;
    gtsam::NavState prevState_;
    gtsam::imuBias::ConstantBias prevBias_;

    //imu状态
    gtsam::NavState prevStateOdom;
    gtsam::imuBias::ConstantBias prevBiasOdom;

    bool doneFirstOpt = false;
    double lastImuT_imu = -1;
    double lastImuT_opt = -1;

    //iSAM2优化器
    gtsam::ISAM2 optimizer;
    gtsam::NonlinearFactorGraph graphFactors; //总的因子图模型
    gtsam::Values graphValues; //因子图模型中的值

    const double delta_t = 0;

    int key = 1;
    int imuPreintegrationResetId = 0;

    //这里是imu-lidar位姿变换
    //这里不可以理解为把imu数据转到lidar下的变换矩阵
    //作者把imu数据先用imuConverter旋转到雷达系下（但其实还差个平移）
    //通过lidar2Imu将雷达数据反响平移了以下，和转换以后差了个平移的imu数据在“中间系”对齐
    //之后算完又从中间系通过imu2lidar挪回了雷达系进行publish
    gtsam::Pose3 imu2Lidar = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(-extTrans.x(), -extTrans.y(), -extTrans.z()));
    gtsam::Pose3 lidar2Imu = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z()));;


    //构造函数
    IMUPreintegration()
    {
        //nh == nodehandle 是个函数句柄
        //订阅imu原始数据，用下面因子图优化的结果，施加两帧之间的imu预积分量，预测每一时刻（imu频率）的imu里程计
        subImu      = nh.subscribe<sensor_msgs::Imu>  (imuTopic, 2000, &IMUPreintegration::imuHandler, this, ros::TransportHints().tcpNoDelay());
        //订阅激光里程计，来自mapOptimization，用两帧之间的imu预积分量构建因子图，优化当前帧位姿（这个位姿仅用于更新每时刻的imu里程计，以及下一次因子图优化）
        subOdometry = nh.subscribe<nav_msgs::Odometry>(PROJECT_NAME + "/lidar/mapping/odometry", 5, &IMUPreintegration::odometryHandler, this, ros::TransportHints().tcpNoDelay());

        //发布imu里程计
        pubImuOdometry = nh.advertise<nav_msgs::Odometry> ("odometry/imu", 2000);
        pubImuPath     = nh.advertise<nav_msgs::Path>     (PROJECT_NAME + "/lidar/imu/path", 1);

        map_to_odom = tf::Transform(tf::createQuaternionFromRPY(0, 0, 0), tf::Vector3(0, 0, 0));

        //imu预积分的噪声协方差
        boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(imuGravity);
        p->accelerometerCovariance  = gtsam::Matrix33::Identity(3,3) * pow(imuAccNoise, 2); // acc white noise in continuous
        p->gyroscopeCovariance      = gtsam::Matrix33::Identity(3,3) * pow(imuGyrNoise, 2); // gyro white noise in continuous
        p->integrationCovariance    = gtsam::Matrix33::Identity(3,3) * pow(1e-4, 2); // error committed in integrating position from velocities
        gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());; // assume zero initial bias

        //噪声先验
        priorPoseNoise  = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()); // rad,rad,rad,m, m, m
        priorVelNoise   = gtsam::noiseModel::Isotropic::Sigma(3, 1e2); // m/s
        priorBiasNoise  = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3); // 1e-2 ~ 1e-3 seems to be good
        //激光里程计scan-to-map优化过程中发生退化，则选择一个较大的协方差
        correctionNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-2); // meter
        noiseModelBetweenBias = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished();
        //imu预积分器，用于预测每一时刻（imu频率）的imu里程计（转到lidar系，与激光里程计同一个系）
        imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for IMU message thread
        //imu预积分器，用于因子图优化
        imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for optimization        
    }

    void resetOptimization()
    {
        //ISAM2的参数类
        gtsam::ISAM2Params optParameters;
        optParameters.relinearizeThreshold = 0.1;
        optParameters.relinearizeSkip = 1;
        //优化器
        optimizer = gtsam::ISAM2(optParameters);

        gtsam::NonlinearFactorGraph newGraphFactors;
        graphFactors = newGraphFactors;

        gtsam::Values NewGraphValues;
        graphValues = NewGraphValues;
    }

    void resetParams()
    {
        //将上一帧imu数据的时间戳更新成-1（代表还没有imu数据进来）
        lastImuT_imu = -1;
        //false代表需要重新进行一次odom优化（跟imuHandler联系起来）
        doneFirstOpt = false;
        //系统关闭
        systemInitialized = false;
    }

    /*
    里程计回调函数
    每隔100帧激光里程计，重置iSAM2优化器，添加里程计、速度、偏置先验因子，执行优化
    计算前一帧激光里程计与当前帧激光里程计之间的imu预积分量，用前一帧状态施加预积分量得到当前帧初始状态估计，添加来自mapOptimization的
    当前位姿，进行因子图优化，更新当前帧状态。
    优化之后，执行重传般，获得imu真实的bias，用来计算当前时刻之后的imu预积分。
    */
    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        //当前帧激光里程计时间戳
        double currentCorrectionTime = ROS_TIME(odomMsg);

        // make sure we have imu data to integrate
        //确保imu优化队列中有imu数据进行预积分
        //在imuHander函数中，会将imu数据push到队列中，数据是imu原始数据经过旋转但没平移到雷达坐标系下的数据
        if (imuQueOpt.empty())
            return;

        //当前帧激光位姿，来自scan-to-map匹配，因子图优化后的位姿
        float p_x = odomMsg->pose.pose.position.x;
        float p_y = odomMsg->pose.pose.position.y;
        float p_z = odomMsg->pose.pose.position.z;
        float r_x = odomMsg->pose.pose.orientation.x;
        float r_y = odomMsg->pose.pose.orientation.y;
        float r_z = odomMsg->pose.pose.orientation.z;
        float r_w = odomMsg->pose.pose.orientation.w;
        //四舍五入
        int currentResetId = round(odomMsg->pose.covariance[0]);
        //用订阅的激光雷达里程计消息，初始化一个lidarpose 包括一个四元数和points
        gtsam::Pose3 lidarPose = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));

        // correction pose jumped, reset imu pre-integration
        //比较当前里程计id 和 imu预积分的id 来判断是否属于同一个时间段内，雷达的id 可以连续几帧都是一个id
        if (currentResetId != imuPreintegrationResetId)
        {
            resetParams();
            imuPreintegrationResetId = currentResetId;
            return;
        }


        // 0. initialize system
        //系统初始化，第一帧
        if (systemInitialized == false)
        {
            resetOptimization(); // 具体可以转到这个函数的声明，没啥难度

            // pop old IMU message
            //从imu优化队列中删除当前帧激光里程计时刻之前的imu数据
            while (!imuQueOpt.empty())
            {
                if (ROS_TIME(&imuQueOpt.front()) < currentCorrectionTime - delta_t)
                {
                    //将队首元素的时间戳记录下来，更新lastImuT_opt，再将元素弹出
                    lastImuT_opt = ROS_TIME(&imuQueOpt.front());
                    imuQueOpt.pop_front();
                }
                else
                    break;
            }
            // initial pose
            //添加里程计位姿先验因子
            //将雷达位置转换到Imu坐标系下
            prevPose_ = lidarPose.compose(lidar2Imu);
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise);
            //先验因子中的参数：位姿信息和噪声，在构造函数中初始化过，是一个常量
            graphFactors.add(priorPose);
            // initial velocity
            //添加里程计速度先验因子
            prevVel_ = gtsam::Vector3(0, 0, 0);
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);
            //类似于位姿先验因子的添加
            graphFactors.add(priorVel);
            // initial bias
            //添加imu偏差先验因子
            prevBias_ = gtsam::imuBias::ConstantBias();
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise);
            graphFactors.add(priorBias);
            // add values
            //变量节点赋初值
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
            // optimize once
            //假定起始为0速度，进行一次优化，用于刚开始的scan-matching
            //对于速度小于10m/s 角速度小于180度/s 效果都非常好
            //但是这总归是基于0起始速度估计的，起始估计并不是实际情况，因此消除图优化中内容
            optimizer.update(graphFactors, graphValues);
            graphFactors.resize(0);
            graphValues.clear();

            //重置预积分器
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
            imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
            
            //key = 1 表示有一帧
            //系统初始化成功
            key = 1;
            systemInitialized = true;
            return;
        }


        // reset graph for speed
        if (key == 100)
        {
            // get updated noise before reset
            gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise  = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key-1)));
            // reset graph
            resetOptimization();
            // add pose
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
            graphFactors.add(priorPose);
            // add velocity
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
            graphFactors.add(priorVel);
            // add bias
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, updatedBiasNoise);
            graphFactors.add(priorBias);
            // add values
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
            // optimize once
            optimizer.update(graphFactors, graphValues);
            graphFactors.resize(0);
            graphValues.clear();

            key = 1;
        }


        // 1. integrate imu data and optimize
        //初始化完成后就可以估计IMU偏差，机器人位姿，速度
        //计算前一帧与当前帧之间的imu预积分量，用前一帧状态施加预积分量得到当前帧初始状态估计
        //添加来自mapOptimization的当前帧位姿，进行因子图优化，更新当前帧状态
        while (!imuQueOpt.empty())
        {
            // pop and integrate imu data that is between two optimizations
            //提取前一帧与当前帧之间的imu数据，计算预积分
            sensor_msgs::Imu *thisImu = &imuQueOpt.front();
            double imuTime = ROS_TIME(thisImu);
            //判断这一帧是否已经超过了当前激光雷达帧的时间戳
            if (imuTime < currentCorrectionTime - delta_t)
            {
                double dt = (lastImuT_opt < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_opt);
                //此处积分只用到了线加速度和角速度
                //加入的是用来因子图优化的预积分器imuIntegratorOpt_.注意加入了上一步计算出的dt
                //作者要求的9轴imu数据中欧拉角在本程序中没有任何用到，全在地图优化里用到的
                imuIntegratorOpt_->integrateMeasurement(
                        gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                
                lastImuT_opt = imuTime;
                //从队列中删除已经处理的imu数据
                imuQueOpt.pop_front();
            }
            else
                break;
        }
        // add imu factor to graph
        //加imu预积分因子，利用两帧之间的IMU数据完成了预积分后增加imu因子到因子图中
        const gtsam::PreintegratedImuMeasurements& preint_imu = dynamic_cast<const gtsam::PreintegratedImuMeasurements&>(*imuIntegratorOpt_);
        //函数参数：前一帧位姿，前一帧速度，当前帧位姿，当前帧速度，前一帧偏差，预积分量
        gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), preint_imu);
        graphFactors.add(imu_factor);
        // add imu bias between factor
        graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
                         gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));
        // add pose factor
        gtsam::Pose3 curPose = lidarPose.compose(lidar2Imu);
        gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), curPose, correctionNoise);
        //添加位姿因子
        graphFactors.add(pose_factor);
        // insert predicted values
        //用前一帧的状态、偏差、施加imu预积分量来得到当前帧的状态
        gtsam::NavState propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);
        graphValues.insert(X(key), propState_.pose());
        graphValues.insert(V(key), propState_.v());
        graphValues.insert(B(key), prevBias_);
        // optimize
        optimizer.update(graphFactors, graphValues);
        optimizer.update();
        graphFactors.resize(0);
        graphValues.clear();
        // Overwrite the beginning of the preintegration for the next step.
        //优化结果
        gtsam::Values result = optimizer.calculateEstimate();
        //更新当前帧位姿、速度 --> 上一帧
        prevPose_  = result.at<gtsam::Pose3>(X(key));
        prevVel_   = result.at<gtsam::Vector3>(V(key));
        //更新当前帧状态 --> 上一帧
        prevState_ = gtsam::NavState(prevPose_, prevVel_);
        //更新当前帧imu偏差
        prevBias_  = result.at<gtsam::imuBias::ConstantBias>(B(key));
        // Reset the optimization preintegration object.
        //重置预积分器，设置新的偏差，这样下一帧激光里程计进来的时候，预积分量就是两帧之间的增量
        imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
        // check optimization
        //检查是否失效
        if (failureDetection(prevVel_, prevBias_))
        {
            resetParams();
            return;
        }


        // 2. after optiization, re-propagate imu odometry preintegration
        prevStateOdom = prevState_;
        prevBiasOdom  = prevBias_;
        // first pop imu message older than current correction data
        double lastImuQT = -1;
        //从imu队列中删除当前激光里程计时刻之前的imu数据
        while (!imuQueImu.empty() && ROS_TIME(&imuQueImu.front()) < currentCorrectionTime - delta_t)
        {
            //弹出
            lastImuQT = ROS_TIME(&imuQueImu.front());
            imuQueImu.pop_front();
        }
        // repropogate
        //对剩余的imu数据计算预积分
        //因为bias改变了
        if (!imuQueImu.empty())
        {
            // reset bias use the newly optimized bias
            //重置预积分器和最新的偏置，使用最新的偏差更新bias值
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);
            // integrate imu message from the beginning of this optimization
            for (int i = 0; i < (int)imuQueImu.size(); ++i)
            {
                //重新进行预积分
                //与imuHandler中的预积分过程相同
                sensor_msgs::Imu *thisImu = &imuQueImu[i];
                double imuTime = ROS_TIME(thisImu);
                double dt = (lastImuQT < 0) ? (1.0 / 500.0) :(imuTime - lastImuQT);

                imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                                                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                lastImuQT = imuTime;
            }
        }

        //记录帧数
        ++key;
        //优化器开启
        doneFirstOpt = true;
    }

    //速度大于30或者偏差大于0.1，则返回失效，否则没有失效
    bool failureDetection(const gtsam::Vector3& velCur, const gtsam::imuBias::ConstantBias& biasCur)
    {
        Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
        if (vel.norm() > 30)
        {
            ROS_WARN("Large velocity, reset IMU-preintegration!");
            return true;
        }

        Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(), biasCur.accelerometer().z());
        Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(), biasCur.gyroscope().z());
        if (ba.norm() > 0.1 || bg.norm() > 0.1)
        {
            ROS_WARN("Large bias, reset IMU-preintegration!");
            return true;
        }

        return false;
    }

    //header中包含了三个参数，方向协方差，角速度协方差，加速度协方差
    //这个叫做回调函数
    //接受从imu得到的原始数据进行处理，利用时间戳信息在imu与积分器中加入该帧
    //然后利用上一帧中的激光里程计时刻对应的状态和偏差，加入当前帧的预测，得到当前时刻的状态。
    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg)
    {
        //这里的imuConverter函数只有旋转，没有平移
        sensor_msgs::Imu thisImu = imuConverter(*imuMsg);
        //后续与thisImu有关的都是在差一个平移的雷达坐标系中

        // publish static tf
        tfMap2Odom.sendTransform(tf::StampedTransform(map_to_odom, thisImu.header.stamp, "map", "odom"));

        imuQueOpt.push_back(thisImu);
        imuQueImu.push_back(thisImu);

        //这里要求上一次imu因子图优化执行成功，确保更新了上一帧的状态、bias、预积分的信息
        //如果没有执行过odom优化，或者上一次优化失败导致系统重置，则等待一次odom的优化再继续函数流程。
        if (doneFirstOpt == false)
            return;

        //获取当前imu因子图的时间戳
        double imuTime = ROS_TIME(&thisImu);
        //lastImuT_imu初始值为-1
        //如果首次优化，则定义初始时间为1/500秒，否则与上一帧作差
        double dt = (lastImuT_imu < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_imu);
        //更新上一帧imu因子图的时间戳
        lastImuT_imu = imuTime;

        // integrate this single imu message
        //imu预积分器添加一帧imu数据
        imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
                                                gtsam::Vector3(thisImu.angular_velocity.x,    thisImu.angular_velocity.y,    thisImu.angular_velocity.z), dt);

        // predict odometry
        //用上一帧激光里程计时刻对应的信息，加上imu预积分量，得到当前时刻的状态。
        gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);

        // publish odometry
        //发布imu里程计（转到lidar系，与激光里程计同一个系）
        nav_msgs::Odometry odometry;
        odometry.header.stamp = thisImu.header.stamp;
        odometry.header.frame_id = "odom";
        odometry.child_frame_id = "odom_imu";

        // transform imu pose to ldiar
        //预测值curretnState获得imu位姿，再由imu到雷达变换，获得雷达位姿
        gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());
        gtsam::Pose3 lidarPose = imuPose.compose(imu2Lidar);

        //pose表示位置的point 和 quaternion，以及对应的协方差矩阵
        //twist表示速度的linear 和 angular，以及对应的协方差矩阵
        //第一个pose:geometry_msgs/PoseWithCovariance
        //第二个pose:geometry_msgs/Pose,包含一个point 和 一个quaternion
        odometry.pose.pose.position.x = lidarPose.translation().x();
        odometry.pose.pose.position.y = lidarPose.translation().y();
        odometry.pose.pose.position.z = lidarPose.translation().z();
        odometry.pose.pose.orientation.x = lidarPose.rotation().toQuaternion().x();
        odometry.pose.pose.orientation.y = lidarPose.rotation().toQuaternion().y();
        odometry.pose.pose.orientation.z = lidarPose.rotation().toQuaternion().z();
        odometry.pose.pose.orientation.w = lidarPose.rotation().toQuaternion().w();
        
        odometry.twist.twist.linear.x = currentState.velocity().x();
        odometry.twist.twist.linear.y = currentState.velocity().y();
        odometry.twist.twist.linear.z = currentState.velocity().z();
        odometry.twist.twist.angular.x = thisImu.angular_velocity.x + prevBiasOdom.gyroscope().x();
        odometry.twist.twist.angular.y = thisImu.angular_velocity.y + prevBiasOdom.gyroscope().y();
        odometry.twist.twist.angular.z = thisImu.angular_velocity.z + prevBiasOdom.gyroscope().z();
        // information for VINS initialization
        odometry.pose.covariance[0] = double(imuPreintegrationResetId);
        odometry.pose.covariance[1] = prevBiasOdom.accelerometer().x();
        odometry.pose.covariance[2] = prevBiasOdom.accelerometer().y();
        odometry.pose.covariance[3] = prevBiasOdom.accelerometer().z();
        odometry.pose.covariance[4] = prevBiasOdom.gyroscope().x();
        odometry.pose.covariance[5] = prevBiasOdom.gyroscope().y();
        odometry.pose.covariance[6] = prevBiasOdom.gyroscope().z();
        odometry.pose.covariance[7] = imuGravity;
        //发布里程计信息
        pubImuOdometry.publish(odometry);

        // publish imu path
        static nav_msgs::Path imuPath;
        static double last_path_time = -1;
        //当前imu因子图的时间戳
        if (imuTime - last_path_time > 0.1)
        {
            last_path_time = imuTime;
            //定义一个带有时间戳的pose的信息
            geometry_msgs::PoseStamped pose_stamped;
            /*
            把这一帧imu的时间戳、关联的坐标系id、位置相关信息赋值给pose_stamped
            thisImu是在原始的imu数据旋转到雷达坐标系后的数据
            */
            pose_stamped.header.stamp = thisImu.header.stamp;
            pose_stamped.header.frame_id = "odom";
            pose_stamped.pose = odometry.pose.pose;
            //加入path
            imuPath.poses.push_back(pose_stamped);
            while(!imuPath.poses.empty() && abs(imuPath.poses.front().header.stamp.toSec() - imuPath.poses.back().header.stamp.toSec()) > 3.0)
                imuPath.poses.erase(imuPath.poses.begin());
            if (pubImuPath.getNumSubscribers() != 0)
            {
                //给imuPath添加时间戳stamp 和 坐标系id frame_id
                imuPath.header.stamp = thisImu.header.stamp;
                imuPath.header.frame_id = "odom";
                //发布消息
                pubImuPath.publish(imuPath);
            }
        }

        // publish transformation
        //发布odom到base_link的变换，其实也就是到imu的变换
        tf::Transform tCur;
        //pose.pose 是imu里程计的pose信息（位置和方向）
        tf::poseMsgToTF(odometry.pose.pose, tCur);
        tf::StampedTransform odom_2_baselink = tf::StampedTransform(tCur, thisImu.header.stamp, "odom", "base_link");
        tfOdom2BaseLink.sendTransform(odom_2_baselink);
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lidar");
    
    IMUPreintegration ImuP;

    ROS_INFO("\033[1;32m----> Lidar IMU Preintegration Started.\033[0m");

    ros::spin();
    
    return 0;
}

//map：地图坐标系，真实世界的坐标
//odom：里程计坐标系，相当于是根据实际计算得到的坐标和真实坐标之间的变换虚拟出来的一个坐标系
//base_link：机器人本体坐标系，与机器人的中心是重合的
//base_laser：激光雷达的坐标系，与激光雷达的安装位置有关

//odom topic可以通过位姿转换矩阵得到odom->base_link的tf关系。map与odom在运动开始时是重合的，
//但是随着机器的运动，两者之间逐渐出现了偏差，这就是里程计的累计误差。map->odom的tf转化需要矫正