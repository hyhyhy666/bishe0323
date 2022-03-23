#include "feature_tracker.h"

#define SHOW_UNDISTORTION 0


// mtx lock for two threads
std::mutex mtx_lidar;

// global variable for saving the depthCloud shared between two threads
pcl::PointCloud<PointType>::Ptr depthCloud(new pcl::PointCloud<PointType>());

// global variables saving the lidar point cloud
deque<pcl::PointCloud<PointType>> cloudQueue;
deque<double> timeQueue;

// global depth register for obtaining depth of a feature
DepthRegister *depthRegister;

// feature publisher for VINS estimator
ros::Publisher pub_feature;
ros::Publisher pub_match;
ros::Publisher pub_restart;

// feature tracker variables
FeatureTracker trackerData[NUM_OF_CAM];
double first_image_time;
int pub_count = 1;
bool first_image_flag = true;
double last_image_time = 0;
bool init_pub = 0;



// 对于新来的图像进行特征点的追踪
void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    double cur_img_time = img_msg->header.stamp.toSec();

    // 处理第一帧图像
    if(first_image_flag)
    {
        first_image_flag = false;
        first_image_time = cur_img_time;
        last_image_time = cur_img_time;
        return;
    }
    // detect unstable camera stream
    // 处理不稳定数据流
    if (cur_img_time - last_image_time > 1.0 || cur_img_time < last_image_time)
    {
        ROS_WARN("image discontinue! reset the feature tracker!");
        // 重置
        first_image_flag = true; 
        last_image_time = 0;
        pub_count = 1;
        std_msgs::Bool restart_flag;
        restart_flag.data = true;
        pub_restart.publish(restart_flag);
        return;
    }
    last_image_time = cur_img_time;
    // frequency control
    // 控制发布频率
    // pub_count: 发布图像的个数
    // FREQ: 20 控制图像光流跟踪的频率，这里为20HZ
    // PUB_THIS_FRAME: 是否需要发布特征点的标志
    if (round(1.0 * pub_count / (cur_img_time - first_image_time)) <= FREQ)
    {
        PUB_THIS_FRAME = true;
        // reset the frequency control
        if (abs(1.0 * pub_count / (cur_img_time - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = cur_img_time;
            pub_count = 0;
        }
    }
    else
    {
        PUB_THIS_FRAME = false;
    }

    // 图像的格式调整和图像读取
    // CvBrigde是ROS库，提供ROS和OpenCV之间的接口
    // 8UC1 是 8bit的单色灰度图
    // mono8 是 8UC1的一个格式
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
        // toCvCopy()函数将ROS图像消息转换为OpenCV图像
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    // 对最新帧特征点的提取和光流追踪(核心)
    // readImage()实现了特征的处理和光流的追踪，基本调用了feature_tracker.cpp的全部函数
    // 如果是单目摄像头，调用readImage()
    // 如果是双目摄像头，需要自适应直方图均衡化处理
    // 这里的EQUALIZE是 如果光太亮 或 太暗 则为1，用来进行直方图均衡化
    cv::Mat show_img = ptr->image;
    // img_msg 或 img都是sensor_msg格式的，需要一个桥梁，转换为CV::Mat格式的数据，以供后续图像处理
    TicToc t_r; // 计算时间的类
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ROS_DEBUG("processing camera %d", i);
        if (i != 1 || !STEREO_TRACK)
            trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), cur_img_time);
        else
        {
            if (EQUALIZE)
            {
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
            }
            else
                trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
        }

        #if SHOW_UNDISTORTION
            trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
        #endif
    }

    // 对新加入的特征点更新全局id
    for (unsigned int i = 0;; i++)
    {
        bool completed = false;
        for (int j = 0; j < NUM_OF_CAM; j++)
            if (j != 1 || !STEREO_TRACK)
                completed |= trackerData[j].updateID(i);
        if (!completed)
            break;
    }

   // 校正、封装并发布特征点到pub_feature
   // 将特征点id，校正后归一化平面的3D点（x，y，z=1），像素2D点（u，v），像素的速度（vx，vy）
   // 封装成sensor_msgs::PointCloudPtr类型的feature_points实例中，发布到pub_img
   if (PUB_THIS_FRAME)
   {
        pub_count++; // 发布数量+1
        // 用于封装发布的信息
        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
        sensor_msgs::ChannelFloat32 id_of_point;
        sensor_msgs::ChannelFloat32 u_of_point; // 像素坐标x
        sensor_msgs::ChannelFloat32 v_of_point; // 像素坐标y
        sensor_msgs::ChannelFloat32 velocity_x_of_point; // 速度x
        sensor_msgs::ChannelFloat32 velocity_y_of_point; // 速度y

        feature_points->header.stamp = img_msg->header.stamp;
        feature_points->header.frame_id = "vins_body";

        vector<set<int>> hash_ids(NUM_OF_CAM); // 哈希表id
        for (int i = 0; i < NUM_OF_CAM; i++) // 循环相机数量
        {
            auto &un_pts = trackerData[i].cur_un_pts;
            auto &cur_pts = trackerData[i].cur_pts;
            auto &ids = trackerData[i].ids;
            auto &pts_velocity = trackerData[i].pts_velocity;
            for (unsigned int j = 0; j < ids.size(); j++) // 特征点的数量
            {
                if (trackerData[i].track_cnt[j] > 1) // 只发布追踪次数大于1的特征点
                {
                    int p_id = ids[j];
                    hash_ids[i].insert(p_id);
                    geometry_msgs::Point32 p;
                    p.x = un_pts[j].x;
                    p.y = un_pts[j].y;
                    p.z = 1;

                    feature_points->points.push_back(p);
                    id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
                    u_of_point.values.push_back(cur_pts[j].x);
                    v_of_point.values.push_back(cur_pts[j].y);
                    velocity_x_of_point.values.push_back(pts_velocity[j].x);
                    velocity_y_of_point.values.push_back(pts_velocity[j].y);
                }
            }
        }

        // 封装信息，准备发布
        feature_points->channels.push_back(id_of_point);
        feature_points->channels.push_back(u_of_point);
        feature_points->channels.push_back(v_of_point);
        feature_points->channels.push_back(velocity_x_of_point);
        feature_points->channels.push_back(velocity_y_of_point);

        // get feature depth from lidar point cloud
        // 从lidar点云数据中获取深度信息
        // 从共享内存中获得深度信息
        pcl::PointCloud<PointType>::Ptr depth_cloud_temp(new pcl::PointCloud<PointType>());
        mtx_lidar.lock();
        *depth_cloud_temp = *depthCloud;
        mtx_lidar.unlock();

        // 获取深度
        sensor_msgs::ChannelFloat32 depth_of_points = depthRegister->get_depth(img_msg->header.stamp, show_img, depth_cloud_temp, trackerData[0].m_camera, feature_points->points);
        feature_points->channels.push_back(depth_of_points);
        
        // skip the first image; since no optical speed on frist image
        // 跳过第一帧图像，因为没有光流信息
        if (!init_pub)
        {
            init_pub = 1;
        }
        else
            pub_feature.publish(feature_points);

        // publish features in image
        // 在图像中发布特征
        if (pub_match.getNumSubscribers() != 0)
        {
            ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::RGB8);
            //cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
            cv::Mat stereo_img = ptr->image;

            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
                cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB); // show_img灰度图转RGB(tmp_img)

                // 显示追踪状态，越红越好，越蓝越不行--cv::Scalar决定的
                for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
                {
                    if (SHOW_TRACK)
                    {
                        // track count
                        // 计算跟踪的特征点数量
                        double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
                        cv::circle(tmp_img, trackerData[i].cur_pts[j], 4, cv::Scalar(255 * (1 - len), 255 * len, 0), 4);
                    }
                    else
                    {
                        // depth
                        // 结合深度进行计算 
                        if(j < depth_of_points.values.size())
                        {
                            if (depth_of_points.values[j] > 0)
                                cv::circle(tmp_img, trackerData[i].cur_pts[j], 4, cv::Scalar(0, 255, 0), 4);
                            else
                                cv::circle(tmp_img, trackerData[i].cur_pts[j], 4, cv::Scalar(0, 0, 255), 4);
                        }
                    }
                }
            }

            pub_match.publish(ptr->toImageMsg());
        }
    }
}


void lidar_callback(const sensor_msgs::PointCloud2ConstPtr& laser_msg)
{
    // LIDAR_SKIP的值为3
    static int lidar_count = -1;
    // 每4个就会跳过一次处理
    if (++lidar_count % (LIDAR_SKIP+1) != 0)
        return;

    // 0. listen to transform
    // 接受位姿变换信息，降采样通过之后，从laser_msg中，获取位姿的信息
    static tf::TransformListener listener;
    static tf::StampedTransform transform;
    try
    {
        listener.waitForTransform("vins_world", "vins_body_ros", laser_msg->header.stamp, ros::Duration(0.01));
        listener.lookupTransform("vins_world", "vins_body_ros", laser_msg->header.stamp, transform);
    } 
    catch (tf::TransformException ex)
    {
        ROS_ERROR("lidar no tf");
        return;
    }

    double xCur, yCur, zCur, rollCur, pitchCur, yawCur;
    xCur = transform.getOrigin().x();
    yCur = transform.getOrigin().y();
    zCur = transform.getOrigin().z();
    tf::Matrix3x3 m(transform.getRotation());
    m.getRPY(rollCur, pitchCur, yawCur);
    Eigen::Affine3f transNow = pcl::getTransformation(xCur, yCur, zCur, rollCur, pitchCur, yawCur);

    // 1. convert laser cloud message to pcl
    // 将点云转换成PCL格式
    pcl::PointCloud<PointType>::Ptr laser_cloud_in(new pcl::PointCloud<PointType>());
    pcl::fromROSMsg(*laser_msg, *laser_cloud_in);

    // 2. downsample new cloud (save memory)
    // 调用PCL的降采样算法
    // 使用PCL内置额过滤器是的数据规模下降，节约内存空间
    
    // 生成新的PCL格式的点云容器
    pcl::PointCloud<PointType>::Ptr laser_cloud_in_ds(new pcl::PointCloud<PointType>());
    // 生成过滤器
    static pcl::VoxelGrid<PointType> downSizeFilter;
    // 设置过滤的大小，设置采样体素大小
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    // 设置输入的点源
    downSizeFilter.setInputCloud(laser_cloud_in);
    // 开始过滤，并将输出放到刚生成的PCL的容器里
    downSizeFilter.filter(*laser_cloud_in_ds);
    // 把过滤好的数据覆盖原本的数据
    *laser_cloud_in = *laser_cloud_in_ds;

    // 3. filter lidar points (only keep points in camera view)
    // 保证当前点云的点在当前相机视角内
    // 生成新的PCL格式的点云容器
    //pcl::PointCloud<PointType>::Ptr laser_cloud_in_filter(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr laser_cloud_in_filter(new pcl::PointCloud<PointType>());
    // 遍历所有的点
    for (int i = 0; i < (int)laser_cloud_in->size(); ++i)
    {
        PointType p = laser_cloud_in->points[i];
        // 符合条件的数据，放入laser_cloud_in_filter中
        if (p.x >= 0 && abs(p.y / p.x) <= 10 && abs(p.z / p.x) <= 10)
            laser_cloud_in_filter->push_back(p);
    }
    // 把过滤好的数据覆盖原本的数据
    *laser_cloud_in = *laser_cloud_in_filter;

    // TODO: transform to IMU body frame
    // 4. offset T_lidar -> T_camera 
    // 将点云从激光雷达坐标系变成相机坐标系
    // 通过这一部分，可以为后面的图像信息中的特征点生成深度信息

    pcl::PointCloud<PointType>::Ptr laser_cloud_offset(new pcl::PointCloud<PointType>());
    // 获得tf变换信息，生成变换矩阵
    Eigen::Affine3f transOffset = pcl::getTransformation(L_C_TX, L_C_TY, L_C_TZ, L_C_RX, L_C_RY, L_C_RZ);
    // 从lidar坐标系转变到相机坐标系
    pcl::transformPointCloud(*laser_cloud_in, *laser_cloud_offset, transOffset);
    // 覆盖原有数据
    *laser_cloud_in = *laser_cloud_offset;

    // 5. transform new cloud into global odom frame
    // 再把点云变换到世界坐标系
    // 把第0步获取到的变换矩阵，用来将坐标系变换到世界坐标系
    // 经过前面的降采样，将数据量降低，提高程序的效率
    pcl::PointCloud<PointType>::Ptr laser_cloud_global(new pcl::PointCloud<PointType>());
    pcl::transformPointCloud(*laser_cloud_in, *laser_cloud_global, transNow);

    // 6. save new cloud
    // 最后把变换完成的点云存储到待处理队列
    double timeScanCur = laser_msg->header.stamp.toSec();
    cloudQueue.push_back(*laser_cloud_global);
    timeQueue.push_back(timeScanCur);

    // 7. pop old cloud
    // 保持队列的时效
    while (!timeQueue.empty())
    {
        if (timeScanCur - timeQueue.front() > 5.0)
        {
            cloudQueue.pop_front();
            timeQueue.pop_front();
        }
        else
        {
            break;
        }
    }

    std::lock_guard<std::mutex> lock(mtx_lidar); // 需要访问到共享内存，访问前拿到共享内存的锁
    // 8. fuse global cloud
    // 将队列里的点云输入作为总体的待处理深度图
    depthCloud->clear();
    for (int i = 0; i < (int)cloudQueue.size(); ++i)
        *depthCloud += cloudQueue[i];

    // 9. downsample global cloud
    // 降采样总体的深度图
    pcl::PointCloud<PointType>::Ptr depthCloudDS(new pcl::PointCloud<PointType>());
    // 设置过滤的大小，设置采样体素的大小
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    downSizeFilter.setInputCloud(depthCloud);
    downSizeFilter.filter(*depthCloudDS);
    *depthCloud = *depthCloudDS;

    // pcl::PointCloud<PointType>::Ptr depthCloudDS(new pcl::PointCloud<PointType>());
    // downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    // downSizeFilter.setInputCloud(depthClooud);
    // downSizeFilter.filter(*depthCloudDS);
    // *depthCloud = *depthCloudDS;
}

int main(int argc, char **argv)
{
    // initialize ROS node
    // 这里虽然有vins这个名字，但是被launch文件中覆盖了
    ros::init(argc, argv, "vins");
    // 然后获取操作这个节点的句柄，并且控制日志的记录的等级
    ros::NodeHandle n;
    ROS_INFO("\033[1;32m----> Visual Feature Tracker Started.\033[0m");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Warn);
    // 读入这个节点所需要的参数
    readParameters(n);

    // read camera params
    for (int i = 0; i < NUM_OF_CAM; i++)
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);

    // load fisheye mask to remove features on the boundry
    // 加载鱼眼mask来去除边界上的特征
    if(FISHEYE)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
            if(!trackerData[i].fisheye_mask.data)
            {
                ROS_ERROR("load fisheye mask fail");
                ROS_BREAK();
            }
            else
                ROS_INFO("load mask success");
        }
    }

    // initialize depthRegister (after readParameters())
    depthRegister = new DepthRegister(n);
    
    // subscriber to image and lidar
    // 输入为原始图像信息 和 上一个imageProjection节点变换到世界坐标系的当前点云信息
    // 订阅原始的图像 和 世界坐标系下的lidar点云
    ros::Subscriber sub_img   = n.subscribe(IMAGE_TOPIC,       5,    img_callback);
    ros::Subscriber sub_lidar = n.subscribe(POINT_CLOUD_TOPIC, 5,    lidar_callback);
    if (!USE_LIDAR)
        sub_lidar.shutdown();

    // messages to vins estimator
    // 输出是给出一个带有深度的特征点 和 带有特征点的图片 和 是否重启的信号
    // 给vins estimator的消息
    pub_feature = n.advertise<sensor_msgs::PointCloud>(PROJECT_NAME + "/vins/feature/feature",     5);
    pub_match   = n.advertise<sensor_msgs::Image>     (PROJECT_NAME + "/vins/feature/feature_img", 5);
    pub_restart = n.advertise<std_msgs::Bool>         (PROJECT_NAME + "/vins/feature/restart",     5);

    // two ROS spinners for parallel processing (image and lidar)
    // 多线程，申请2个线程来运行当有图片消息和lidar消息到来的时候的回调函数
    ros::MultiThreadedSpinner spinner(2);
    spinner.spin();

    return 0;
}