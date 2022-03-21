#include "utility.h"
#include "lvi_sam/cloud_info.h"

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

using namespace gtsam;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose

/*
    * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
    */
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D // xyz + padding(填充的首选方法)
    PCL_ADD_INTENSITY;
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

//注册为PCL点云格式 （PCL内有自己定义的一系列点云格式，用结构体扩展后需要注册）
POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))

typedef PointXYZIRPYT  PointTypePose; //重命名


class mapOptimization : public ParamServer
{

public:

    // gtsam
    NonlinearFactorGraph gtSAMgraph; // 包含非线性因子的因子图
    //Values 用于指定因子图中一组变量的值
    Values initialEstimate;
    Values optimizedEstimate;
    //优化器
    ISAM2 *isam;
    Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance; //位置协方差

    //发布
    ros::Publisher pubLaserCloudSurround; // 点云信息 发布关键帧的map的特征点云
    ros::Publisher pubOdomAftMappedROS; // odometry信息 发布激光里程计
    ros::Publisher pubKeyPoses; // 点云信息 发布关键位姿信息
    ros::Publisher pubPath; // 路径信息 发布路径，主要是给rivz用于展示

    ros::Publisher pubHistoryKeyFrames; // 历史点云信息 发布历史关键帧
    ros::Publisher pubIcpKeyFrames; // 点云信息 icp方法 配准的 对齐的 发布当前关键帧经过闭环优化后的位姿变换之后的特征点云
    ros::Publisher pubRecentKeyFrames; // 点云信息 发布局部map的降采样平面边重合
    ros::Publisher pubRecentKeyFrame; // 点云信息 发布历史帧的角点、平面点降采样集合
    ros::Publisher pubCloudRegisteredRaw; // 点云信息 校准之后的点云信息
    ros::Publisher pubLoopConstraintEdge; // MarkerArray信息 发布闭环边信息

    //输入：激光点云信息、GPS信息、闭环信息
    ros::Subscriber subLaserCloudInfo; // 订阅当前激光帧点云信息，来自FeatureExtraction
    ros::Subscriber subGPS; // 订阅GPS里程计
    ros::Subscriber subLoopInfo; // 订阅来自外部闭环检测程序提供的闭环数据。（本程序没有提供）

    std::deque<nav_msgs::Odometry> gpsQueue;
    lvi_sam::cloud_info cloudInfo;

    //历史所有关键帧的角点集合（降采样）
    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
    //历史所有关键帧的平面点集合（降采样）
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;
    
    //历史关键帧位姿
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast; // corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast; // surf feature set from odoOptimization
    //后面DS是downsampled 降采样的简称 是上面经过降采样后的数据
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; // downsampled corner featuer set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS; // downsampled surf featuer set from odoOptimization

    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    std::vector<PointType> laserCloudOriCornerVec; // corner point holder for parallel computation
    std::vector<PointType> coeffSelCornerVec;
    std::vector<bool> laserCloudOriCornerFlag;
    std::vector<PointType> laserCloudOriSurfVec; // surf point holder for parallel computation
    std::vector<PointType> coeffSelSurfVec;
    std::vector<bool> laserCloudOriSurfFlag;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

    //局部关键帧构建的地图点云，对应kdtree，用于scan-to-map找相邻
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

    pcl::PointCloud<PointType>::Ptr latestKeyFrameCloud;
    pcl::PointCloud<PointType>::Ptr nearHistoryKeyFrameCloud;

    //降采样
    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterICP;
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization
    
    ros::Time timeLaserInfoStamp;
    double timeLaserInfoCur;

    float transformTobeMapped[6];

    std::mutex mtx;

    bool isDegenerate = false;
    cv::Mat matP;

    //当前激光帧角点数量
    int laserCloudCornerLastDSNum = 0;
    //当前激光帧平面点数量
    int laserCloudSurfLastDSNum = 0;

    bool aLoopIsClosed = false;
    int imuPreintegrationResetId = 0;

    //一个关于运动路径的消息
    nav_msgs::Path globalPath;

    //当前帧位姿
    Eigen::Affine3f transPointAssociateToMap;

    map<int, int> loopIndexContainer; // from new to old
    vector<pair<int, int>> loopIndexQueue;
    vector<gtsam::Pose3> loopPoseQueue;
    vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;

    //构造函数
    mapOptimization()
    {
        ISAM2Params parameters; // ISAM2优化器的参数
        //设置上界和下界
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        //初始化优化器
        isam = new ISAM2(parameters);

        //发布历史关键帧里程计
        pubKeyPoses           = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/mapping/trajectory", 1);
        //发布局部关键帧地图的特征点云
        pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/mapping/map_global", 1);
        //发布激光里程计，rviz中表现为坐标轴
        pubOdomAftMappedROS   = nh.advertise<nav_msgs::Odometry>      (PROJECT_NAME + "/lidar/mapping/odometry", 1);
        //发布路径
        pubPath               = nh.advertise<nav_msgs::Path>          (PROJECT_NAME + "/lidar/mapping/path", 1);

        //订阅当前激光帧点云信息，来自featureExtraction
        subLaserCloudInfo     = nh.subscribe<lvi_sam::cloud_info>     (PROJECT_NAME + "/lidar/feature/cloud_info", 5, &mapOptimization::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        //订阅GPS里程计 这个GPS在图优化部分可以先不考虑，这个加入GPS会更精确
        subGPS                = nh.subscribe<nav_msgs::Odometry>      (gpsTopic,                                   50, &mapOptimization::gpsHandler, this, ros::TransportHints().tcpNoDelay());
        //订阅来自外部闭环检测程序提供的闭环数据
        subLoopInfo           = nh.subscribe<std_msgs::Float64MultiArray>(PROJECT_NAME + "/vins/loop/match_frame", 5, &mapOptimization::loopHandler, this, ros::TransportHints().tcpNoDelay());

        //发布闭环匹配关键帧局部地图
        pubHistoryKeyFrames   = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/mapping/loop_closure_history_cloud", 1);
        //发布当前关键帧经过闭环优化后的位姿变换以后的特征点云
        pubIcpKeyFrames       = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/mapping/loop_closure_corrected_cloud", 1);
        //发布闭环边，rviz中变现为闭环帧之间的连线
        pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>(PROJECT_NAME + "/lidar/mapping/loop_closure_constraints", 1);

        //发布局部地图的降采样平面点集合
        pubRecentKeyFrames    = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/mapping/map_local", 1);
        //发布历史帧的角点、平面点降采样集合
        pubRecentKeyFrame     = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/mapping/cloud_registered", 1);
        //发布当前帧原始点云配准之后的点云
        pubCloudRegisteredRaw = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/mapping/cloud_registered_raw", 1);

        //设置每个体素的大小，分别为长宽高
        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        //用于扫描到贴图优化的周围关键姿势
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization

        //初始化变量
        allocateMemory();
    }

    void allocateMemory()
    {
        //对定义的一系列关键帧变量进行初始化
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner featuer set from odoOptimization
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        latestKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        nearHistoryKeyFrameCloud.reset(new pcl::PointCloud<PointType>());

        for (int i = 0; i < 6; ++i)
        {
            transformTobeMapped[i] = 0;
        }

        matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));
    }

    //订阅激光帧点云信息
    void laserCloudInfoHandler(const lvi_sam::cloud_infoConstPtr& msgIn)
    {
        // extract time stamp
        timeLaserInfoStamp = msgIn->header.stamp;
        timeLaserInfoCur = msgIn->header.stamp.toSec();

        // extract info ana feature cloud 提取当前激光帧角点、平面点集合
        cloudInfo = *msgIn;
        pcl::fromROSMsg(msgIn->cloud_corner,  *laserCloudCornerLast);
        pcl::fromROSMsg(msgIn->cloud_surface, *laserCloudSurfLast);

        //设置锁（同步）
        std::lock_guard<std::mutex> lock(mtx);

        static double timeLastProcessing = -1;
        //mapping执行频率控制
        //如果当前激光帧与上一帧激光帧之间的差值大于某个值
        //mappingProcessInterval在utility.h中被初始化为0.15
        if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval) {

            //更新时间戳
            timeLastProcessing = timeLaserInfoCur;

            //当前帧初始化
            //1.如果是第一帧，用原始imu数据的RPY初始化当前帧位姿（旋转部分）
            //2.后续帧，用imu里程计计算两帧之间的增量位姿变换，作用于前一帧的激光位姿，得到当前帧激光位姿
            updateInitialGuess();

            //提取局部角点、平面点云集合，加入局部地图
            //1.对最近的一帧关键帧，搜索时空纬度上相邻的关键帧集合，降采样一下
            //2.对关键帧集合中的每一帧，提取对应的角点、平面点，加入局部地图中
            extractSurroundingKeyFrames();

            //当前激光帧角点、平面点集合降采样
            downsampleCurrentScan();

            //1、要求当前帧特征点数量足够多，且匹配的点数够多，才执行优化
            
            //2、迭代30次优化
            //1）当前激光帧角点寻找局部map匹配点
            //a.更新当前帧位姿，将当前帧角点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成直线，
            //  则认为匹配上了（用距离中心点的协方差矩阵，特征值进行判断）
            //b.计算当前帧角点到直线的距离、垂线的单位向量，存储为角点参数
            //2）当前激光帧平面点寻找局部map匹配点
            //a.更新当前帧位姿，将当前帧平面点坐标交换到map系下，在局部map中查找5个最近点，距离小于1米，且5个点构成平面
            // （最小二乘拟合平面），则认为匹配上了
            //b.计算当前帧平面点到平面的距离、垂线的单位向量，存储为平面点参数
            //3）提取当前帧中与局部map匹配上了的角点、平面点，加入同一集合
            //4）对匹配特征点计算Jacobian矩阵，观测值为特征点到直线、平面的距离，构建高斯牛顿方程，迭代优化当前位姿，存transformTobeMapped

            //3、用imu原始RPY数据与scan-to-map优化后的位姿进行加权融合，更新当前帧位姿的roll pirch 约束z坐标
            scan2MapOptimization();

            //因子图优化，更新所有关键帧位姿
            //设置当前帧为关键帧并执行因子图优化
            //计算当前帧与前一帧位姿变化，如果变化太小，不设为关键帧，反之设为关键帧
            //添加激光里程计因子、GPS因子、闭环因子
            //执行因子图优化
            //得到当前帧优化后位姿，位姿协方差
            //添加cloudKeyPoses3D,cloudKeyPoses6D,更新transformTobeMapped,添加当前关键帧的角点、平面点集合
            //该优化是独立于scan-map的另一个优化
            saveKeyFramesAndFactor();

            //更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿，更新里程计轨迹
            correctPoses();

            //发布激光里程计
            publishOdometry();

            //发布里程计、点云、轨迹
            //1、发布历史关键帧位姿集合
            //2、发布局部地图降采样平面点集合
            //3、发布历史帧（累加的）的角点、平面点降采样集合
            //4、发布里程计轨迹
            publishFrames();
        }
    }

    void gpsHandler(const nav_msgs::Odometry::ConstPtr& gpsMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);
        gpsQueue.push_back(*gpsMsg);
    }

    void pointAssociateToMap(PointType const * const pi, PointType * const po)
    {
        po->x = transPointAssociateToMap(0,0) * pi->x + transPointAssociateToMap(0,1) * pi->y + transPointAssociateToMap(0,2) * pi->z + transPointAssociateToMap(0,3);
        po->y = transPointAssociateToMap(1,0) * pi->x + transPointAssociateToMap(1,1) * pi->y + transPointAssociateToMap(1,2) * pi->z + transPointAssociateToMap(1,3);
        po->z = transPointAssociateToMap(2,0) * pi->x + transPointAssociateToMap(2,1) * pi->y + transPointAssociateToMap(2,2) * pi->z + transPointAssociateToMap(2,3);
        po->intensity = pi->intensity;
    }

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        PointType *pointFrom;

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
        
        for (int i = 0; i < cloudSize; ++i)
        {
            pointFrom = &cloudIn->points[i];
            cloudOut->points[i].x = transCur(0,0) * pointFrom->x + transCur(0,1) * pointFrom->y + transCur(0,2) * pointFrom->z + transCur(0,3);
            cloudOut->points[i].y = transCur(1,0) * pointFrom->x + transCur(1,1) * pointFrom->y + transCur(1,2) * pointFrom->z + transCur(1,3);
            cloudOut->points[i].z = transCur(2,0) * pointFrom->x + transCur(2,1) * pointFrom->y + transCur(2,2) * pointFrom->z + transCur(2,3);
            cloudOut->points[i].intensity = pointFrom->intensity;
        }
        return cloudOut;
    }

    gtsam::Pose3 affine3fTogtsamPose3(const Eigen::Affine3f& thisPose)
    {
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(thisPose, x, y, z, roll, pitch, yaw);
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(roll), double(pitch), double(yaw)),
                                  gtsam::Point3(double(x),    double(y),     double(z)));
    }

    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                                  gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
    }

    gtsam::Pose3 trans2gtsamPose(float transformIn[])
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]), 
                                  gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }

    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    { 
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    Eigen::Affine3f trans2Affine3f(float transformIn[])
    {
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    }

    PointTypePose trans2PointTypePose(float transformIn[])
    {
        PointTypePose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll  = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw   = transformIn[2];
        return thisPose6D;
    }


    void visualizeGlobalMapThread()
    {
        ros::Rate rate(0.2);
        while (ros::ok()){
            rate.sleep();
            publishGlobalMap();
        }

        if (savePCD == false)
            return;

        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files ..." << endl;
        // create directory and remove old files;
        savePCDDirectory = std::getenv("HOME") + savePCDDirectory;
        int unused = system((std::string("exec rm -r ") + savePCDDirectory).c_str());
        unused = system((std::string("mkdir ") + savePCDDirectory).c_str()); ++unused;
        // save key frame transformations
        pcl::io::savePCDFileASCII(savePCDDirectory + "trajectory.pcd", *cloudKeyPoses3D);
        pcl::io::savePCDFileBinary(savePCDDirectory + "transformations.pcd", *cloudKeyPoses6D);
        // extract global point cloud map        
        pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
        for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++) 
        {
            // clip cloud
            // pcl::PointCloud<PointType>::Ptr cornerTemp(new pcl::PointCloud<PointType>());
            // pcl::PointCloud<PointType>::Ptr cornerTemp2(new pcl::PointCloud<PointType>());
            // *cornerTemp = *transformPointCloud(cornerCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
            // for (int j = 0; j < (int)cornerTemp->size(); ++j)
            // {
            //     if (cornerTemp->points[j].z > cloudKeyPoses6D->points[i].z && cornerTemp->points[j].z < cloudKeyPoses6D->points[i].z + 5)
            //         cornerTemp2->push_back(cornerTemp->points[j]);
            // }
            // pcl::PointCloud<PointType>::Ptr surfTemp(new pcl::PointCloud<PointType>());
            // pcl::PointCloud<PointType>::Ptr surfTemp2(new pcl::PointCloud<PointType>());
            // *surfTemp = *transformPointCloud(surfCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
            // for (int j = 0; j < (int)surfTemp->size(); ++j)
            // {
            //     if (surfTemp->points[j].z > cloudKeyPoses6D->points[i].z && surfTemp->points[j].z < cloudKeyPoses6D->points[i].z + 5)
            //         surfTemp2->push_back(surfTemp->points[j]);
            // }
            // *globalCornerCloud += *cornerTemp2;
            // *globalSurfCloud   += *surfTemp2;

            // origin cloud
            *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
            *globalSurfCloud   += *transformPointCloud(surfCloudKeyFrames[i],    &cloudKeyPoses6D->points[i]);
            cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
        }
        // down-sample and save corner cloud
        downSizeFilterCorner.setInputCloud(globalCornerCloud);
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudCorner.pcd", *globalCornerCloud);
        // down-sample and save surf cloud
        downSizeFilterSurf.setInputCloud(globalSurfCloud);
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudSurf.pcd", *globalSurfCloud);
        // down-sample and save global point cloud map
        *globalMapCloud += *globalCornerCloud;
        *globalMapCloud += *globalSurfCloud;
        downSizeFilterSurf.setInputCloud(globalMapCloud);
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudGlobal.pcd", *globalMapCloud);
        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files completed" << endl;
    }

    void publishGlobalMap()
    {
        if (pubLaserCloudSurround.getNumSubscribers() == 0)
            return;

        if (cloudKeyPoses3D->points.empty() == true)
            return;

        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());;
        pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

        // kd-tree to find near key frames to visualize
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        // search near key frames to visualize
        mtx.lock();
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
        kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
            globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
        // downsample near selected key frames
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses; // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);

        // extract visualized and downsampled key frames
        for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i){
            if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                continue;
            int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
            *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
            *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
        }
        // downsample visualized points
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames; // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
        publishCloud(&pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, "odom");    
    }

    void loopHandler(const std_msgs::Float64MultiArray::ConstPtr& loopMsg)
    {
        // control loop closure frequency
        static double last_loop_closure_time = -1;
        {
            // std::lock_guard<std::mutex> lock(mtx);
            if (timeLaserInfoCur - last_loop_closure_time < 5.0)
                return;
            else
                last_loop_closure_time = timeLaserInfoCur;
        }

        performLoopClosure(*loopMsg);
    }

    void performLoopClosure(const std_msgs::Float64MultiArray& loopMsg)
    {
        pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D(new pcl::PointCloud<PointTypePose>());
        {
            std::lock_guard<std::mutex> lock(mtx);
            *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
        }

        // get lidar keyframe id
        int key_cur = -1; // latest lidar keyframe id
        int key_pre = -1; // previous lidar keyframe id
        {
            loopFindKey(loopMsg, copy_cloudKeyPoses6D, key_cur, key_pre);
            if (key_cur == -1 || key_pre == -1 || key_cur == key_pre)// || abs(key_cur - key_pre) < 25)
                return;
        }

        // check if loop added before
        {
            // if image loop closure comes at high frequency, many image loop may point to the same key_cur
            auto it = loopIndexContainer.find(key_cur);
            if (it != loopIndexContainer.end())
                return;
        }
        
        // get lidar keyframe cloud
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
        {
            loopFindNearKeyframes(copy_cloudKeyPoses6D, cureKeyframeCloud, key_cur, 0);
            loopFindNearKeyframes(copy_cloudKeyPoses6D, prevKeyframeCloud, key_pre, historyKeyframeSearchNum);
            if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
                return;
            if (pubHistoryKeyFrames.getNumSubscribers() != 0)
                publishCloud(&pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, "odom");
        }

        // get keyframe pose
        Eigen::Affine3f pose_cur;
        Eigen::Affine3f pose_pre;
        Eigen::Affine3f pose_diff_t; // serves as initial guess
        {
            pose_cur = pclPointToAffine3f(copy_cloudKeyPoses6D->points[key_cur]);
            pose_pre = pclPointToAffine3f(copy_cloudKeyPoses6D->points[key_pre]);

            Eigen::Vector3f t_diff;
            t_diff.x() = - (pose_cur.translation().x() - pose_pre.translation().x());
            t_diff.y() = - (pose_cur.translation().y() - pose_pre.translation().y());
            t_diff.z() = - (pose_cur.translation().z() - pose_pre.translation().z());
            if (t_diff.norm() < historyKeyframeSearchRadius)
                t_diff.setZero();
            pose_diff_t = pcl::getTransformation(t_diff.x(), t_diff.y(), t_diff.z(), 0, 0, 0);
        }

        // transform and rotate cloud for matching
        pcl::IterativeClosestPoint<PointType, PointType> icp;
        // pcl::GeneralizedIterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius * 2);
        icp.setMaximumIterations(100);
        icp.setRANSACIterations(0);
        icp.setTransformationEpsilon(1e-3);
        icp.setEuclideanFitnessEpsilon(1e-3);

        // initial guess cloud
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud_new(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*cureKeyframeCloud, *cureKeyframeCloud_new, pose_diff_t);

        // match using icp
        icp.setInputSource(cureKeyframeCloud_new);
        icp.setInputTarget(prevKeyframeCloud);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        icp.align(*unused_result);

        if (pubIcpKeyFrames.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*cureKeyframeCloud_new, *closed_cloud, icp.getFinalTransformation());
            publishCloud(&pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, "odom");
        }

        // add graph factor
        if (icp.getFitnessScore() < historyKeyframeFitnessScore && icp.hasConverged() == true)
        {
            // get gtsam pose
            gtsam::Pose3 poseFrom = affine3fTogtsamPose3(Eigen::Affine3f(icp.getFinalTransformation()) * pose_diff_t * pose_cur);
            gtsam::Pose3 poseTo   = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[key_pre]);
            // get noise
            float noise = icp.getFitnessScore();
            gtsam::Vector Vector6(6);
            Vector6 << noise, noise, noise, noise, noise, noise;
            noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);
            // save pose constraint
            mtx.lock();
            loopIndexQueue.push_back(make_pair(key_cur, key_pre));
            loopPoseQueue.push_back(poseFrom.between(poseTo));
            loopNoiseQueue.push_back(constraintNoise);
            mtx.unlock();
            // add loop pair to container
            loopIndexContainer[key_cur] = key_pre;
        }

        // visualize loop constraints
        if (!loopIndexContainer.empty())
        {
            visualization_msgs::MarkerArray markerArray;
            // loop nodes
            visualization_msgs::Marker markerNode;
            markerNode.header.frame_id = "odom";
            markerNode.header.stamp = timeLaserInfoStamp;
            markerNode.action = visualization_msgs::Marker::ADD;
            markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
            markerNode.ns = "loop_nodes";
            markerNode.id = 0;
            markerNode.pose.orientation.w = 1;
            markerNode.scale.x = 0.3; markerNode.scale.y = 0.3; markerNode.scale.z = 0.3; 
            markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
            markerNode.color.a = 1;
            // loop edges
            visualization_msgs::Marker markerEdge;
            markerEdge.header.frame_id = "odom";
            markerEdge.header.stamp = timeLaserInfoStamp;
            markerEdge.action = visualization_msgs::Marker::ADD;
            markerEdge.type = visualization_msgs::Marker::LINE_LIST;
            markerEdge.ns = "loop_edges";
            markerEdge.id = 1;
            markerEdge.pose.orientation.w = 1;
            markerEdge.scale.x = 0.1;
            markerEdge.color.r = 0.9; markerEdge.color.g = 0.9; markerEdge.color.b = 0;
            markerEdge.color.a = 1;

            for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
            {
                int key_cur = it->first;
                int key_pre = it->second;
                geometry_msgs::Point p;
                p.x = copy_cloudKeyPoses6D->points[key_cur].x;
                p.y = copy_cloudKeyPoses6D->points[key_cur].y;
                p.z = copy_cloudKeyPoses6D->points[key_cur].z;
                markerNode.points.push_back(p);
                markerEdge.points.push_back(p);
                p.x = copy_cloudKeyPoses6D->points[key_pre].x;
                p.y = copy_cloudKeyPoses6D->points[key_pre].y;
                p.z = copy_cloudKeyPoses6D->points[key_pre].z;
                markerNode.points.push_back(p);
                markerEdge.points.push_back(p);
            }

            markerArray.markers.push_back(markerNode);
            markerArray.markers.push_back(markerEdge);
            pubLoopConstraintEdge.publish(markerArray);
        }
    }

    void loopFindNearKeyframes(const pcl::PointCloud<PointTypePose>::Ptr& copy_cloudKeyPoses6D,
                               pcl::PointCloud<PointType>::Ptr& nearKeyframes, 
                               const int& key, const int& searchNum)
    {
        // extract near keyframes
        nearKeyframes->clear();
        int cloudSize = copy_cloudKeyPoses6D->size();
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            int key_near = key + i;
            if (key_near < 0 || key_near >= cloudSize )
                continue;
            *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[key_near], &copy_cloudKeyPoses6D->points[key_near]);
            *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[key_near],   &copy_cloudKeyPoses6D->points[key_near]);
        }

        if (nearKeyframes->empty())
            return;

        // downsample near keyframes
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }

    void loopFindKey(const std_msgs::Float64MultiArray& loopMsg, 
                     const pcl::PointCloud<PointTypePose>::Ptr& copy_cloudKeyPoses6D,
                     int& key_cur, int& key_pre)
    {
        if (loopMsg.data.size() != 2)
            return;

        double loop_time_cur = loopMsg.data[0];
        double loop_time_pre = loopMsg.data[1];

        if (abs(loop_time_cur - loop_time_pre) < historyKeyframeSearchTimeDiff)
            return;

        int cloudSize = copy_cloudKeyPoses6D->size();
        if (cloudSize < 2)
            return;

        // latest key
        key_cur = cloudSize - 1;
        for (int i = cloudSize - 1; i >= 0; --i)
        {
            if (copy_cloudKeyPoses6D->points[i].time > loop_time_cur)
                key_cur = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        // previous key
        key_pre = 0;
        for (int i = 0; i < cloudSize; ++i)
        {
            if (copy_cloudKeyPoses6D->points[i].time < loop_time_pre)
                key_pre = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }
    }

    void loopClosureThread()
    {
        if (loopClosureEnableFlag == false)
            return;

        ros::Rate rate(0.5);
        while (ros::ok())
        {
            rate.sleep();
            performLoopClosureDetection();
        }
    }

    void performLoopClosureDetection()
    {
        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;

        int key_cur = -1;
        int key_pre = -1;

        double loop_time_cur = -1;
        double loop_time_pre = -1;

        // find latest key and time
        {
            std::lock_guard<std::mutex> lock(mtx);

            if (cloudKeyPoses3D->empty())
                return;
            
            kdtreeHistoryKeyPoses->setInputCloud(cloudKeyPoses3D);
            kdtreeHistoryKeyPoses->radiusSearch(cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);

            key_cur = cloudKeyPoses3D->size() - 1;
            loop_time_cur = cloudKeyPoses6D->points[key_cur].time;
        }

        // find previous key and time
        {
            for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
            {
                int id = pointSearchIndLoop[i];
                if (abs(cloudKeyPoses6D->points[id].time - loop_time_cur) > historyKeyframeSearchTimeDiff)
                {
                    key_pre = id;
                    loop_time_pre = cloudKeyPoses6D->points[key_pre].time;
                    break;
                }
            }
        }

        if (key_cur == -1 || key_pre == -1 || key_pre == key_cur ||
            loop_time_cur < 0 || loop_time_pre < 0)
            return;

        std_msgs::Float64MultiArray match_msg;
        match_msg.data.push_back(loop_time_cur);
        match_msg.data.push_back(loop_time_pre);
        performLoopClosure(match_msg);
    }

    void updateInitialGuess()
    {        
        //前一帧的位姿，这里指lidar的位姿
        //Affine3f 仿射变换矩阵（移向量 + 旋转变换组合而成，可以同时实现旋转，缩放，平移等空间变换）
        static Eigen::Affine3f lastImuTransformation;
        // system initialization
        //如果关键帧集合为空，继续进行初始化
        if (cloudKeyPoses3D->points.empty())
        {
            //大小为6的数组，当前帧位姿的旋转部分，用激光帧信息中的RPY（来自IMU原始数据）初始化
            transformTobeMapped[0] = cloudInfo.imuRollInit;
            transformTobeMapped[1] = cloudInfo.imuPitchInit;
            transformTobeMapped[2] = cloudInfo.imuYawInit;

            //在utility.h中定义
            if (!useImuHeadingInitialization)
                transformTobeMapped[2] = 0;

            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
            return;
        }

        // use VINS odometry estimation for pose guess
        //用当前帧和前一帧对应的imu里程计计算相对位姿变换，再用前一帧的位姿与相对变换，计算当前帧的位姿，存transformTobeMapped
        static int odomResetId = 0;
        static bool lastVinsTransAvailable = false;
        static Eigen::Affine3f lastVinsTransformation;
        if (cloudInfo.odomAvailable == true && cloudInfo.odomResetId == odomResetId)
        {
            // ROS_INFO("Using VINS initial guess");
            if (lastVinsTransAvailable == false)
            {
                // ROS_INFO("Initializing VINS initial guess");
                //如果是首次积分，则将lastVinsTransformation赋值为根据odom的xyz,RPY转换得到的transform
                lastVinsTransformation = pcl::getTransformation(cloudInfo.odomX,    cloudInfo.odomY,     cloudInfo.odomZ, 
                                                                cloudInfo.odomRoll, cloudInfo.odomPitch, cloudInfo.odomYaw);
                lastVinsTransAvailable = true;
            } 
            else 
            {
                // ROS_INFO("Obtaining VINS incremental guess");
                //首先从odom转换成transform，获得当前transform在lastVinsTransformation下的位置
                //当前帧的初始化估计位姿（来自IMU里程计），后面用来计算增量位姿变换
                Eigen::Affine3f transBack = pcl::getTransformation(cloudInfo.odomX,    cloudInfo.odomY,     cloudInfo.odomZ, 
                                                                   cloudInfo.odomRoll, cloudInfo.odomPitch, cloudInfo.odomYaw);
                //inverse 逆矩阵
                //当前帧相对于前一帧的位姿变换---imu里程计计算得到
                Eigen::Affine3f transIncre = lastVinsTransformation.inverse() * transBack;

                //前一帧的位姿
                Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
                //当前帧的位姿
                Eigen::Affine3f transFinal = transTobe * transIncre;
                //更新当前帧位姿transformTobeMapped
                pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                              transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

                //将视觉惯性里程计的transform赋值为odom的位姿
                lastVinsTransformation = pcl::getTransformation(cloudInfo.odomX,    cloudInfo.odomY,     cloudInfo.odomZ, 
                                                                cloudInfo.odomRoll, cloudInfo.odomPitch, cloudInfo.odomYaw);

                //保存当前时态的imu位姿
                lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
                return;
                // transformTobeMapped存储激光雷达位姿
            }
        } 
        else 
        {
            // ROS_WARN("VINS failure detected.");
            //没有odom信息，或者是第一帧进入的时候
            lastVinsTransAvailable = false;
            odomResetId = cloudInfo.odomResetId;
        }

        //当odo不可用 或者 lastVinsTransAvailable == false
        // use imu incremental estimation for pose guess (only rotation)
        //只在第一帧调用（注意上面的return）,用imu累计估计位姿（只估计旋转的）
        if (cloudInfo.imuAvailable == true)
        {
            // ROS_INFO("Using IMU initial guess");
            //首先从imu转换成transform,获得当前transform在lastImuTransformation下的位置
            //当前帧的姿态角（来自原始imu数据）
            Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);
            //当前帧相对于前一帧的姿态变换
            Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;

            //将上一时态的imu的transform点乘两个imu时态间的位姿变换，将其赋值给transformTobeMapped数组
            //前一帧的位姿
            Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
            Eigen::Affine3f transFinal = transTobe * transIncre;
            pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                          transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
            return;
        }
    }

    void extractNearby()
    {
        pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        // extract all the nearby key poses and downsample them 提取所有附近的关键点位姿并对其进行下采样
        //katree的输入，全局关键帧位姿集合（历史所有关键帧集合）
        //kdtree作为一个搜索的工具
        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tree
        //surroundingKeyframeSearchRadius默认值为50.0，scan-to-map优化的距离
        //对最近的一帧关键帧，在半径区域内搜索空间区域上相邻的关键帧集合
        //搜索的结果存数在 pointSearchInd 和 pointSearchSqDis中
        //对于back就是点云里的最后一个点
        //这个函数的第一个参数 传入一个点在kd树中，搜索这个所有到这个点的距离小于某个值的点
        kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
        for (int i = 0; i < (int)pointSearchInd.size(); ++i)
        {
            //将搜索到的每个点，通过id在cloudKeyPose3D中找到对应的关键帧（点云）并加入
            int id = pointSearchInd[i];
            //加入相邻关键帧位姿集合中
            surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
        }

        //降采样
        downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
        //降采样后的结果存储在DS中
        downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);

        // also extract some latest key frames in case the robot rotates in one position 同时提取一些最新的关键帧，防止机器人在一个位置旋转
        //加入时间上相邻的一些关键帧，比如当载体在原地转圈，这些帧加进来是合理的
        //Question : 原地旋转为什么不算作空间上相邻的关键帧
        /*
        暂时的答案（3.7）
        3D点云集合中只保存位置信息（xyz），6D点云中保存位置 + 旋转信息（xyz + rpy）
        当原地旋转时，关键帧的xyz应该是保持不变的，所以连续的若干帧会在3D点云集合中只保存一个坐标（是不是这样需要后续看cloudKeyPose3D是怎么添加帧的）
        所以使用3D点云集合搜索空间领域时，会丢失旋转前后的帧（xyz相同，rpy不同），所以需要重新对事件领域进行添加
        这便解释下面if语句中使用的是6D的时间戳
        */
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)
                surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        //将相邻关键帧集合对应的角点、平面点，加入到局部map中，作为scan-to-map匹配的局部点云地图
        extractCloud(surroundingKeyPosesDS);
    }

    //相邻关键帧集合对应的角点、平面点，加入到map中
    //称之为局部map，后面进行scan-to-map匹配，所用到的map就是这里的相邻关键帧对应点云集合
    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
    {
        std::vector<pcl::PointCloud<PointType>> laserCloudCornerSurroundingVec;
        std::vector<pcl::PointCloud<PointType>> laserCloudSurfSurroundingVec;

        laserCloudCornerSurroundingVec.resize(cloudToExtract->size());
        laserCloudSurfSurroundingVec.resize(cloudToExtract->size());

        // extract surrounding map
        //遍历当前帧（实际是取最近的一个关键帧来找它相邻的关键帧集合）时空唯独上相邻的关键帧集合
        #pragma omp parallel for num_threads(numberOfCores) //并行编程
        for (int i = 0; i < (int)cloudToExtract->size(); ++i)
        {
            //相邻关键帧索引
            int thisKeyInd = (int)cloudToExtract->points[i].intensity;
            //提取50米范围内的点，transformPointCloud作用是返回输入点云乘以输入的transform
            //自定义函数，计算两个点云图之间的距离
            if (pointDistance(cloudKeyPoses3D->points[thisKeyInd], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
                continue;
            //相邻关键帧对应的角点、平面点云，通过6D位姿变换到世界坐标系下
            laserCloudCornerSurroundingVec[i]  = *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
            laserCloudSurfSurroundingVec[i]    = *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
        }

        // fuse the map
        //赋值线和面特征点击，并且进行下采样
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear(); 
        for (int i = 0; i < (int)cloudToExtract->size(); ++i)
        {
            //加入局部map
            *laserCloudCornerFromMap += laserCloudCornerSurroundingVec[i];
            *laserCloudSurfFromMap   += laserCloudSurfSurroundingVec[i];
        }

        // Downsample the surrounding corner key frames (or map) 降采样 DS后缀downsample
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        // Downsample the surrounding surf key frames (or map)
        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
    }

    void extractSurroundingKeyFrames()
    {
        //关键帧集合为空，直接返回
        if (cloudKeyPoses3D->points.empty() == true)
            return; 
        
        extractNearby();
    }

    //当前激光帧角点，平面点集合降采样
    void downsampleCurrentScan()
    {
        // Downsample cloud from current scan
        //对第二部分得到的角点和平面点集合进行进一步降采样
        //3.7实验，这个只能写两个，多一个少一个都不行，直接图上的xyz飞了
        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();

        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();

        // laserCloudSurfLastDS->clear();
        // downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        // downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        // laserCloudSurfLastDSNum = laserCloudCornerLastDS->size();
    }

    void updatePointAssociateToMap()
    {
        transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
    }

    void cornerOptimization()
    {
        //放射变换，更新当前位姿与地图间位姿变换
        updatePointAssociateToMap();

        #pragma omp parallel for num_threads(numberOfCores) // 并行计算
        //遍历点云，构建点到直线的约束
        for (int i = 0; i < laserCloudCornerLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            //角点（lidar系的）
            pointOri = laserCloudCornerLastDS->points[i];
            //将点从lidar系转换到map坐标系
            pointAssociateToMap(&pointOri, &pointSel);
            //在局部角点map中查找当前角点相邻的5个角点 这里的5可以做个对比实验
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            //cv库中来存储矩阵的
            cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));
                    
            //只有最近的点都在一定的阈值内（这里是1m）才进行计算
            if (pointSearchSqDis[4] < 1.0)
            {
                float cx = 0, cy = 0, cz = 0;
                for (int j = 0; j < 5; j++) 
                {
                    //计算5个点的均值坐标，记为中心点
                    cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                }
                cx /= 5; cy /= 5;  cz /= 5;

                //根据均值计算协方差（得到一个协方差矩阵 3*3的）
                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++)
                {
                    //计算点与中心点之间的距离
                    float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                    a22 += ay * ay; a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

                //构建协方差矩阵
                matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;

                //eigen库，进行特征值分解，求协方差矩阵的特征值（matD1）和特征向量（matV1）
                cv::eigen(matA1, matD1, matV1);
                //如果最大的特征值相比次大特征值 大很多 则认为构成了线，角点是合格的

                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) 
                {
                    //当前帧角点坐标（map系下）
                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    //局部map对应中心角点，沿着特征向量的方向，前后各取一个点
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                    //这个就是外积，表示三角形的面积
                    float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                    + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                                    + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)) * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

                    //计算1和2点构成线段的长度
                    float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                    float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                              + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                               - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                               + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    //三角形的高，点到直线的距离
                    float ld2 = a012 / l12;

                    //涉及到一个鲁棒核函数，距离越大，s越小，使用距离惩罚因子（权重）
                    float s = 1 - 0.9 * fabs(ld2);

                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    coeff.intensity = s * ld2;
                    //距离越小，这个因子的影响就越小（距离越大，加权后的影响就越大）
                    if (s > 0.1)
                    {
                        //距离小于1 当前激光帧角点，加入匹配集合中
                        laserCloudOriCornerVec[i] = pointOri;
                        //角点的参数
                        coeffSelCornerVec[i] = coeff;
                        laserCloudOriCornerFlag[i] = true;
                    }
                }
            }
        }
    }

    //当前激光帧平面点寻找局部map匹配点
    void surfOptimization()
    {
        updatePointAssociateToMap();

        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudSurfLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudSurfLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel); 
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            Eigen::Matrix<float, 5, 3> matA0;
            Eigen::Matrix<float, 5, 1> matB0;
            Eigen::Vector3f matX0;

            matA0.setZero();
            matB0.fill(-1);
            matX0.setZero();

            if (pointSearchSqDis[4] < 1.0) 
            {
                for (int j = 0; j < 5; j++) 
                {
                    //将5个点存入matA
                    matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }

                //Ax=B，根据这五个点求解平面方程，进行QR分解，获得平面方程解
                matX0 = matA0.colPivHouseholderQr().solve(matB0);

                //平面方程的系数，也是法向量的分量
                float pa = matX0(0, 0);
                float pb = matX0(1, 0);
                float pc = matX0(2, 0);
                float pd = 1; // 这个就是平面方程的常数项

                //将matx归一化，得到单位法向量
                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps; pb /= ps; pc /= ps; pd /= ps;

                bool planeValid = true;
                //检查平面是否合格，如果5个点中有点到平面的距离超过0.2米，那么认为这些点太分散了，不构成平面
                for (int j = 0; j < 5; j++)
                {
                    if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                             pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                             pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2)
                    {
                        planeValid = false;
                        break;
                    }
                }

                //平面合格
                if (planeValid)
                {
                    //当前激光帧点到平面距离
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
                            + pointSel.y * pointSel.y + pointSel.z * pointSel.z));

                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;

                    if (s > 0.1)
                    {
                        laserCloudOriSurfVec[i] = pointOri;
                        coeffSelSurfVec[i] = coeff;
                        laserCloudOriSurfFlag[i] = true;
                    }
                }
            }
        }
    }

    //提取当前帧中与局部map匹配上了的角点、平面点，加入同一集合
    void combineOptimizationCoeffs()
    {
        // combine corner coeffs
        //遍历当前帧角点集合，提取出与局部map匹配上了的角点
        for (int i = 0; i < laserCloudCornerLastDSNum; ++i)
        {
            if (laserCloudOriCornerFlag[i] == true)
            {
                laserCloudOri->push_back(laserCloudOriCornerVec[i]);
                coeffSel->push_back(coeffSelCornerVec[i]);
            }
        }
        // combine surf coeffs
        //遍历当前帧平面点集合，提取出与局部map匹配上了的平面点
        for (int i = 0; i < laserCloudSurfLastDSNum; ++i)
        {
            if (laserCloudOriSurfFlag[i] == true)
            {
                laserCloudOri->push_back(laserCloudOriSurfVec[i]);
                coeffSel->push_back(coeffSelSurfVec[i]);
            }
        }
        // reset flag for next iteration
        //清空标记
        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
    }

    //scan-to-map优化
    bool LMOptimization(int iterCount)
    {
        // This optimization is from the original loam_velodyne, need to cope with coordinate transformation
        // lidar <- camera      ---     camera <- lidar
        // x = z                ---     x = y
        // y = x                ---     y = z
        // z = y                ---     z = x
        // roll = yaw           ---     roll = pitch
        // pitch = roll         ---     pitch = yaw
        // yaw = pitch          ---     yaw = roll
        //这个是相机到雷达系的转换          这个是雷达到相机系的转换
        //就是后面的值 赋值给前面 举个例子：相机的x赋值给雷达的y 那么雷达的y就要赋值给相机的x 这样就能对应起来了
        //这个在此代码中是这样的 其他的代码 再检查就可以。

        // lidar -> camera
        //雷达到相机变换的三轴方向的正弦 和 余弦
        float srx = sin(transformTobeMapped[1]);
        float crx = cos(transformTobeMapped[1]);
        float sry = sin(transformTobeMapped[2]);
        float cry = cos(transformTobeMapped[2]);
        float srz = sin(transformTobeMapped[0]);
        float crz = cos(transformTobeMapped[0]);

        int laserCloudSelNum = laserCloudOri->size();
        //当前帧匹配特征点数太少
        if (laserCloudSelNum < 50)
        {
            return false;
        }

        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));

        PointType pointOri, coeff;

        //遍历匹配特征点，构建Jacobian矩阵
        for (int i = 0; i < laserCloudSelNum; i++)
        {
            // lidar -> camera
            pointOri.x = laserCloudOri->points[i].y;
            pointOri.y = laserCloudOri->points[i].z;
            pointOri.z = laserCloudOri->points[i].x;
            // lidar -> camera
            //coeff为点到直线/平面的距离
            coeff.x = coeffSel->points[i].y;
            coeff.y = coeffSel->points[i].z;
            coeff.z = coeffSel->points[i].x;
            coeff.intensity = coeffSel->points[i].intensity;
            // in camera
            float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                      + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                      + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;

            float ary = ((cry*srx*srz - crz*sry)*pointOri.x 
                      + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                      + ((-cry*crz - srx*sry*srz)*pointOri.x 
                      + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;

            float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                      + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                      + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;
            // lidar -> camera
            matA.at<float>(i, 0) = arz;
            matA.at<float>(i, 1) = arx;
            matA.at<float>(i, 2) = ary;
            matA.at<float>(i, 3) = coeff.z;
            matA.at<float>(i, 4) = coeff.x;
            matA.at<float>(i, 5) = coeff.y;
            //点到直线距离、平面距离，作为观测值
            matB.at<float>(i, 0) = -coeff.intensity;
        }

        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        //高斯牛顿方程，进行解决
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        //首次迭代，检查近似Hession矩阵，看是否退化，或者称为奇异，也就是它的行列式的值为0
        if (iterCount == 0) 
        {
            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[6] = {100, 100, 100, 100, 100, 100};
            for (int i = 5; i >= 0; i--)
            {
                if (matE.at<float>(0, i) < eignThre[i])
                {
                    for (int j = 0; j < 6; j++)
                    {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                }
                else
                {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }

        if (isDegenerate)
        {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);

        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));

        if (deltaR < 0.05 && deltaT < 0.05)
        {
            return true; // converged
        }
        return false; // keep optimizing
    }

    void scan2MapOptimization()
    {
        //如果关键帧点云数据为空，则直接返回
        if (cloudKeyPoses3D->points.empty())
            return;

        //关键点数量大于阈值，边为10，面为100
        if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum)
        {
            //输入为局部map点云
            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

            //设置的一个迭代次数 30
            for (int iterCount = 0; iterCount < 30; iterCount++)
            {
                //每次迭代清空特征点集合
                //当前帧与局部map匹配上了的角点，平面点，加入同一集合，后面是对应点的参数
                laserCloudOri->clear();
                coeffSel->clear();

                //当前激光帧角点寻找局部map匹配点 这个要看函数内部定义
                cornerOptimization();
                //当前激光帧平面点寻找局部map匹配点
                surfOptimization();

                //提取当前帧中与局部map匹配上了的角点、平面点，加入同一集合
                combineOptimizationCoeffs();

                //scan-to-map优化
                //对匹配特征点计算雅克比矩阵，观测值为特征点到直线、平面的距离，构建高斯牛顿方程，迭代优化当前位姿，存transformTobeMapped
                if (LMOptimization(iterCount) == true)
                    break;              
            }

            transformUpdate();
        } else {
            ROS_WARN("Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
        }
    }

    //用imu原始RPY数据与scan-to-map优化后的位姿进行加权融合，更新当前帧位姿的roll pitch 约束z坐标
    void transformUpdate()
    {
        if (cloudInfo.imuAvailable == true)
        {
            //俯仰角小于1.4
            if (std::abs(cloudInfo.imuPitchInit) < 1.4)
            {
                double imuWeight = 0.01;
                tf::Quaternion imuQuaternion;
                tf::Quaternion transformQuaternion;
                double rollMid, pitchMid, yawMid;

                // slerp roll
                //roll角求加权均值，用scan-to-map优化得到的位姿与imu原始RPY数据，进行加权平均
                transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
                imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
                //slerp函数，利用求面差值得到介于两个已知旋转之间的近似值
                //转到函数的定义，简要返回四元数，它是该四元数和另一个四元数之间的球面线性插值的结果
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[0] = rollMid;

                // slerp pitch
                transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
                imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[1] = pitchMid;
            }
        }

        transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);
        transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);
        transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);
    }

    float constraintTransformation(float value, float limit)
    {
        if (value < -limit)
            value = -limit;
        if (value > limit)
            value = limit;

        return value;
    }

    bool saveFrame()
    {
        if (cloudKeyPoses3D->points.empty())
            return true;

        Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
        Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

        if (abs(roll)  < surroundingkeyframeAddingAngleThreshold &&
            abs(pitch) < surroundingkeyframeAddingAngleThreshold && 
            abs(yaw)   < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingDistThreshold)
            return false;

        return true;
    }

    //激光里程计因子
    void addOdomFactor()
    {
        //检查是否有关键帧点
        if (cloudKeyPoses3D->points.empty())
        {
            //第一帧初始化先验因子
            noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
            gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
            //变量节点设置初始值
            initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
        }
        else
        {
            //添加激光里程计因子
            noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
            gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
            //添加关键帧最近的两个位姿
            //参数：前一帧id，当前帧id，第一帧与当前帧的位姿变换（作为观测值），噪声协方差
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
            //变量节点设置初始值
            //下标
            initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
            // if (isDegenerate)
            // {
            //     adding VINS constraints is deleted as benefits are not obvious, disable for now
            //     gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), vinsPoseFrom.between(vinsPoseTo), odometryNoise));
            // }
        }
    }

    void addGPSFactor()
    {
        if (gpsQueue.empty())
            return;

        // wait for system initialized and settles down
        if (cloudKeyPoses3D->points.empty())
            return;
        else if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 5.0)
                return;

        // pose covariance small, no need to correct
        if (poseCovariance(3,3) < poseCovThreshold && poseCovariance(4,4) < poseCovThreshold)
            return;

        // last gps position
        static PointType lastGPSPoint;

        while (!gpsQueue.empty())
        {
            if (gpsQueue.front().header.stamp.toSec() < timeLaserInfoCur - 0.2)
            {
                // message too old
                gpsQueue.pop_front();
            }
            else if (gpsQueue.front().header.stamp.toSec() > timeLaserInfoCur + 0.2)
            {
                // message too new
                break;
            }
            else
            {
                nav_msgs::Odometry thisGPS = gpsQueue.front();
                gpsQueue.pop_front();

                // GPS too noisy, skip
                float noise_x = thisGPS.pose.covariance[0];
                float noise_y = thisGPS.pose.covariance[7];
                float noise_z = thisGPS.pose.covariance[14];
                if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)
                    continue;

                float gps_x = thisGPS.pose.pose.position.x;
                float gps_y = thisGPS.pose.pose.position.y;
                float gps_z = thisGPS.pose.pose.position.z;
                if (!useGpsElevation)
                {
                    gps_z = transformTobeMapped[5];
                    noise_z = 0.01;
                }

                // GPS not properly initialized (0,0,0)
                if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
                    continue;

                // Add GPS every a few meters
                PointType curGPSPoint;
                curGPSPoint.x = gps_x;
                curGPSPoint.y = gps_y;
                curGPSPoint.z = gps_z;
                if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0)
                    continue;
                else
                    lastGPSPoint = curGPSPoint;

                gtsam::Vector Vector3(3);
                Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);
                noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Variances(Vector3);
                gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
                gtSAMgraph.add(gps_factor);

                aLoopIsClosed = true;

                break;
            }
        }
    }

    //闭环因子
    void addLoopFactor()
    {
        if (loopIndexQueue.empty())
            return;

        for (size_t i = 0; i < loopIndexQueue.size(); ++i)
        {
            //添加相邻两个回环位姿
            int indexFrom = loopIndexQueue[i].first;
            int indexTo = loopIndexQueue[i].second;
            gtsam::Pose3 poseBetween = loopPoseQueue[i];
            gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
            //添加到因子图中
            gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
        }

        //清空所有的回环数据队列
        loopIndexQueue.clear();
        loopPoseQueue.clear();
        loopNoiseQueue.clear();
        aLoopIsClosed = true;
    }

    //因子图优化，更新所有关键帧位姿
    void saveKeyFramesAndFactor()
    {
        //计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
        if (saveFrame() == false)
            return;

        // odom factor
        //激光里程计因子
        addOdomFactor();

        //这个gps因子没有加入，这个地方以后研究
        // gps factor
        //addGPSFactor();

        // loop factor
        //闭环检测因子，这个需要VIS先看见，然后用LIS来进一步优化，论文中是这么说的
        addLoopFactor();

        // update iSAM
        //iSAM就是来优化的
        isam->update(gtSAMgraph, initialEstimate);
        isam->update();
        
        //update以后要清空一下保存的因子图，note！历史数据不会清掉，ISAM保存起来了
        gtSAMgraph.resize(0);
        initialEstimate.clear();

        //save key poses
        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;

        //优化结果
        isamCurrentEstimate = isam->calculateEstimate();
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);
        // cout << "****************************************************" << endl;
        // isamCurrentEstimate.print("Current estimate: ");

        //cloudKeyPoses3D 加入当前帧位姿
        thisPose3D.x = latestEstimate.translation().x();
        thisPose3D.y = latestEstimate.translation().y();
        thisPose3D.z = latestEstimate.translation().z();
        thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index 这个强度不用来定位，而是用来分类，可以百度
        cloudKeyPoses3D->push_back(thisPose3D);

        //cloudKeyPoses6D 加入当前帧位姿
        //6D中存有时间戳
        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
        thisPose6D.roll  = latestEstimate.rotation().roll();
        thisPose6D.pitch = latestEstimate.rotation().pitch();
        thisPose6D.yaw   = latestEstimate.rotation().yaw();
        thisPose6D.time = timeLaserInfoCur;
        cloudKeyPoses6D->push_back(thisPose6D); //加入到cloudKeyPoses6D中

        // cout << "****************************************************" << endl;
        // cout << "Pose covariance:" << endl;
        // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
        //位姿协方差
        poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);

        // save updated transform
        //transformTobeMapped中更新当前帧位姿，注意每个位置对应的RPY 和 xyz
        transformTobeMapped[0] = latestEstimate.rotation().roll();
        transformTobeMapped[1] = latestEstimate.rotation().pitch();
        transformTobeMapped[2] = latestEstimate.rotation().yaw();
        transformTobeMapped[3] = latestEstimate.translation().x();
        transformTobeMapped[4] = latestEstimate.translation().y();
        transformTobeMapped[5] = latestEstimate.translation().z();

        // save all the received edge and surf points
        //当前帧激光角点、平面点，降采样集合
        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);

        // save key frame cloud
        //保存特征点降采样集合
        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);

        // save path for visualization
        //更新里程计轨迹，将thisPose6D的位姿加入到/lidar/mapping/path中去
        updatePath(thisPose6D);
    }

    //更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿，更新里程计轨迹
    void correctPoses()
    {
        if (cloudKeyPoses3D->points.empty())
            return;

        if (aLoopIsClosed == true)
        {
            // clear path
            //清空里程计轨迹
            globalPath.poses.clear();

            // update key poses
            int numPoses = isamCurrentEstimate.size();
            //更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿
            for (int i = 0; i < numPoses; ++i)
            {
                //这里3d表示xyz，6d表示xyz+rpy，代码一目了然
                cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
                cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
                cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
                cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();

                //将修正好的位姿逐一再添加到globalPath中去
                updatePath(cloudKeyPoses6D->points[i]);
            }

            aLoopIsClosed = false;
            // ID for reseting IMU pre-integration
            ++imuPreintegrationResetId;
        }
    }

    //发布激光里程计
    void publishOdometry()
    {
        // Publish odometry for ROS
        nav_msgs::Odometry laserOdometryROS;
        laserOdometryROS.header.stamp = timeLaserInfoStamp;
        laserOdometryROS.header.frame_id = "odom";
        laserOdometryROS.child_frame_id = "odom_mapping";
        //猜测transformTobeMapped的123为rpy，事实论证是对的
        laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
        laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
        laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
        laserOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        laserOdometryROS.pose.covariance[0] = double(imuPreintegrationResetId);
        pubOdomAftMappedROS.publish(laserOdometryROS);
        // Publish TF
        //发布TF，odom->lidar
        static tf::TransformBroadcaster br;
        tf::Transform t_odom_to_lidar = tf::Transform(tf::createQuaternionFromRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]),
                                                      tf::Vector3(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]));
        tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(t_odom_to_lidar, timeLaserInfoStamp, "odom", "lidar_link");
        br.sendTransform(trans_odom_to_lidar);
    }

    //更新里程计轨迹，将thisPose6D的位姿加入到/lidar/mapping/path 中去
    //更新路径（位置坐标 + 旋转四元数）
    void updatePath(const PointTypePose& pose_in)
    {
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);
        pose_stamped.header.frame_id = "odom";
        pose_stamped.pose.position.x = pose_in.x;
        pose_stamped.pose.position.y = pose_in.y;
        pose_stamped.pose.position.z = pose_in.z;
        tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();

        globalPath.poses.push_back(pose_stamped);
    }

    //发布里程计、点云、轨迹
    void publishFrames()
    {
        if (cloudKeyPoses3D->points.empty())
            return;
        // publish key poses
        //发布历史关键帧位姿集合
        publishCloud(&pubKeyPoses, cloudKeyPoses6D, timeLaserInfoStamp, "odom");
        // Publish surrounding key frames
        //发布局部map的降采样特征点集合
        publishCloud(&pubRecentKeyFrames, laserCloudSurfFromMapDS, timeLaserInfoStamp, "odom");
        // publish registered key frame
        //发布历史帧（累加的）的角点，平面点降采样集合
        if (pubRecentKeyFrame.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            //transformPointCloud作用是返回输入点云乘以输入的transform
            *cloudOut += *transformPointCloud(laserCloudCornerLastDS,  &thisPose6D);
            *cloudOut += *transformPointCloud(laserCloudSurfLastDS,    &thisPose6D);
            publishCloud(&pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, "odom");
        }
        // publish registered high-res raw cloud
        if (pubCloudRegisteredRaw.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut = *transformPointCloud(cloudOut,  &thisPose6D);
            publishCloud(&pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, "odom");
        }
        // publish path
        //发布路径
        if (pubPath.getNumSubscribers() != 0)
        {
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = "odom";
            pubPath.publish(globalPath);
        }
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lidar");

    mapOptimization MO;

    ROS_INFO("\033[1;32m----> Lidar Map Optimization Started.\033[0m");
    
    std::thread loopDetectionthread(&mapOptimization::loopClosureThread, &MO);
    std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);

    ros::spin();

    loopDetectionthread.join();
    visualizeMapThread.join();

    return 0;
}