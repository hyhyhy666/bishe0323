#include "utility.h" // 包含了本系统中所需要的各种初始定义，位于本地文件夹下
#include "lvi_sam/cloud_info.h" // 点云的基本信息，也包含了imu 和 odom的成员变量

struct smoothness_t
{ 
    float value;
    size_t ind;
};

struct by_value
{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right)
    { 
        return left.value < right.value;
    }
};

class FeatureExtraction : public ParamServer
{

public:

    ros::Subscriber subLaserCloudInfo; // 输入：点云信息

    // 输出：点云信息
    ros::Publisher pubLaserCloudInfo; // 用来完成发布不同的点云信息，激光帧提取特征之后的点云信息
    ros::Publisher pubCornerPoints; // 激光帧提取到的角点点云
    ros::Publisher pubSurfacePoints; // 激光帧提取到的平面点点云

    pcl::PointCloud<PointType>::Ptr extractedCloud;
    pcl::PointCloud<PointType>::Ptr cornerCloud;
    pcl::PointCloud<PointType>::Ptr surfaceCloud;

    pcl::VoxelGrid<PointType> downSizeFilter;

    //用来存储当前激光帧的全部信息，包括所有的历史数据
    lvi_sam::cloud_info cloudInfo;
    std_msgs::Header cloudHeader; // Header是std_msgs包下用来传递数据的点云信息

    //用来存储激光帧点云的曲率
    std::vector<smoothness_t> cloudSmoothness;
    float *cloudCurvature;
    
    //有待确定，一个用来表示是否提取，一个用来表示提取的类型：角点还是平面点
    int *cloudNeighborPicked;
    int *cloudLabel;

    //构造函数，用来初始化
    //它们初始化了订阅者和发布者的信息，并且给上文所定义的各个变量进行初始化。
    FeatureExtraction()
    {
        //订阅当前激光帧运动畸变校正后的点云信息
        subLaserCloudInfo = nh.subscribe<lvi_sam::cloud_info>(PROJECT_NAME + "/lidar/deskew/cloud_info", 5, &FeatureExtraction::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());

        //发布当前激光帧提取特征之后的点云信息
        pubLaserCloudInfo = nh.advertise<lvi_sam::cloud_info> (PROJECT_NAME + "/lidar/feature/cloud_info", 5);
        //发布当前激光帧的角点点云
        pubCornerPoints = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/feature/cloud_corner", 5);
        //发布当前激光帧的平面点点云
        pubSurfacePoints = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/feature/cloud_surface", 5);
        
        //调用初始化函数
        initializationValue();
    }

    //初始化各个变量信息
    void initializationValue()
    {
        cloudSmoothness.resize(N_SCAN*Horizon_SCAN);

        downSizeFilter.setLeafSize(odometrySurfLeafSize, odometrySurfLeafSize, odometrySurfLeafSize);

        extractedCloud.reset(new pcl::PointCloud<PointType>());
        cornerCloud.reset(new pcl::PointCloud<PointType>());
        surfaceCloud.reset(new pcl::PointCloud<PointType>());

        cloudCurvature = new float[N_SCAN*Horizon_SCAN];
        cloudNeighborPicked = new int[N_SCAN*Horizon_SCAN];
        cloudLabel = new int[N_SCAN*Horizon_SCAN];
    }

    void laserCloudInfoHandler(const lvi_sam::cloud_infoConstPtr& msgIn)
    {
        cloudInfo = *msgIn; // new cloud info
        cloudHeader = msgIn->header; // new cloud header
        pcl::fromROSMsg(msgIn->cloud_deskewed, *extractedCloud); // new cloud for extraction

        calculateSmoothness();

        markOccludedPoints();

        extractFeatures();

        publishFeatureCloud();
    }

    void calculateSmoothness()
    {
        int cloudSize = extractedCloud->points.size();
        //遍历所有的有效点云
        for (int i = 5; i < cloudSize - 5; i++)
        {
            //用当前激光点前后5个点计算当前点的曲率
            float diffRange = cloudInfo.pointRange[i-5] + cloudInfo.pointRange[i-4]
                            + cloudInfo.pointRange[i-3] + cloudInfo.pointRange[i-2]
                            + cloudInfo.pointRange[i-1] - cloudInfo.pointRange[i] * 10
                            + cloudInfo.pointRange[i+1] + cloudInfo.pointRange[i+2]
                            + cloudInfo.pointRange[i+3] + cloudInfo.pointRange[i+4]
                            + cloudInfo.pointRange[i+5];            

            cloudCurvature[i] = diffRange*diffRange;//diffX * diffX + diffY * diffY + diffZ * diffZ;

            cloudNeighborPicked[i] = 0;
            cloudLabel[i] = 0;
            // cloudSmoothness for sorting
            //存储该点的曲率 和 激光点的一维索引
            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind = i;
        }
    }

    //用来标记两种不同的点，分别是被遮挡的 和 被标记的。
    void markOccludedPoints()
    {
        int cloudSize = extractedCloud->points.size();
        // mark occluded points and parallel beam points 标记遮挡点和平行光束点
        for (int i = 5; i < cloudSize - 6; ++i)
        {
            // occluded points
            float depth1 = cloudInfo.pointRange[i];
            float depth2 = cloudInfo.pointRange[i+1];
            int columnDiff = std::abs(int(cloudInfo.pointColInd[i+1] - cloudInfo.pointColInd[i]));

            if (columnDiff < 10)
            {
                // 10 pixel diff in range image
                if (depth1 - depth2 > 0.3)
                {
                    cloudNeighborPicked[i - 5] = 1;
                    cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1;
                }
                else if (depth2 - depth1 > 0.3)
                {
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1;
                }
            }
            // parallel beam
            float diff1 = std::abs(float(cloudInfo.pointRange[i-1] - cloudInfo.pointRange[i]));
            float diff2 = std::abs(float(cloudInfo.pointRange[i+1] - cloudInfo.pointRange[i]));

            if (diff1 > 0.02 * cloudInfo.pointRange[i] && diff2 > 0.02 * cloudInfo.pointRange[i])
                cloudNeighborPicked[i] = 1;
        }
    }

    //特征提取，最核心的函数
    void extractFeatures()
    {
        //首先清除原来的信息，并创建指针，用来存放平面点 和 降采样以后的平面点
        cornerCloud->clear();
        surfaceCloud->clear();

        pcl::PointCloud<PointType>::Ptr surfaceCloudScan(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointType>());

        for (int i = 0; i < N_SCAN; i++)
        {
            //N_SCAN 是在utility.h中定义的变量，这个是用来存储雷达中的数据：扫描线
            //因此第一个for循环的作用是遍历所有的扫描线。
            surfaceCloudScan->clear();

            for (int j = 0; j < 6; j++)
            {
                //这里的6是一个常量，是作者用来分段的数目，将每一个SCAN分为六段来进行分析

                //用线性插值对SCAN进行等分，取得sp和ep
                int sp = (cloudInfo.startRingIndex[i] * (6 - j) + cloudInfo.endRingIndex[i] * j) / 6;
                int ep = (cloudInfo.startRingIndex[i] * (5 - j) + cloudInfo.endRingIndex[i] * (j + 1)) / 6 - 1;

                if (sp >= ep)
                    continue;

                std::sort(cloudSmoothness.begin()+sp, cloudSmoothness.begin()+ep, by_value());

                int largestPickedNum = 0;
                for (int k = ep; k >= sp; k--)
                {
                    int ind = cloudSmoothness[k].ind; // 是一个标签，为了避免重复访问点云
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold)
                    {
                        //判断条件：当前激光点还未被处理，且曲率大于阈值，则认为是角点
                        //这里的edgeThreshold默认值为0.1
                        largestPickedNum++;
                        if (largestPickedNum <= 20)
                        {
                            //在每一段中只提取出来20个点，然后将其标记为角点，加入角点点云
                            cloudLabel[ind] = 1;
                            cornerCloud->push_back(extractedCloud->points[ind]);
                        }
                        else
                        {
                            break;
                        }

                        cloudNeighborPicked[ind] = 1;
                        for (int l = 1; l <= 5; l++)
                        {
                            //同一条扫描线上前5个点标记一下，不再处理，避免特征聚集
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--)
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                for (int k = sp; k <= ep; k++)
                {
                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold)
                    {

                        cloudLabel[ind] = -1;
                        cloudNeighborPicked[ind] = 1;

                        for (int l = 1; l <= 5; l++) {

                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {

                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                for (int k = sp; k <= ep; k++)
                {
                    if (cloudLabel[k] <= 0)
                    {
                        surfaceCloudScan->push_back(extractedCloud->points[k]);
                    }
                }
            }

            surfaceCloudScanDS->clear();
            downSizeFilter.setInputCloud(surfaceCloudScan); // 降采样
            downSizeFilter.filter(*surfaceCloudScanDS);

            *surfaceCloud += *surfaceCloudScanDS;
        }
    }

    void freeCloudInfoMemory() // 清理点云信息
    {
        cloudInfo.startRingIndex.clear();
        cloudInfo.startRingIndex.shrink_to_fit();
        cloudInfo.endRingIndex.clear();
        cloudInfo.endRingIndex.shrink_to_fit();
        cloudInfo.pointColInd.clear();
        cloudInfo.pointColInd.shrink_to_fit();
        cloudInfo.pointRange.clear();
        cloudInfo.pointRange.shrink_to_fit();
    }

    void publishFeatureCloud() // 发布点云信息
    {
        //先清理点云信息，再保存好新提取到的点云信息，然后传送给图优化函数mapOptimization
        // free cloud info memory
        freeCloudInfoMemory();
        // save newly extracted features
        cloudInfo.cloud_corner  = publishCloud(&pubCornerPoints,  cornerCloud,  cloudHeader.stamp, "base_link");
        cloudInfo.cloud_surface = publishCloud(&pubSurfacePoints, surfaceCloud, cloudHeader.stamp, "base_link");
        // publish to mapOptimization
        pubLaserCloudInfo.publish(cloudInfo);
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lidar");

    FeatureExtraction FE;

    ROS_INFO("\033[1;32m----> Lidar Feature Extraction Started.\033[0m");
   
    ros::spin();

    return 0;
}