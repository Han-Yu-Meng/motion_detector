/*******************************************************************************
 * Copyright (c) 2025. IWIN-FINS Lab, Shanghai Jiao Tong University.
 ******************************************************************************/

#include <fins/node.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <omp.h> // 引入 OpenMP

#if __has_include(<tf2_eigen/tf2_eigen.hpp>)
  #include <tf2_eigen/tf2_eigen.hpp>
#else
  #include <tf2_eigen/tf2_eigen.h>
#endif

#include <Eigen/Dense>
#include <mutex>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <cstring>

namespace m_detector {

struct __attribute__((packed)) PointXYZI {
    float x; float y; float z; float intensity;
};

struct Pixel {
    float min_depth = 9999.0f;
    float max_depth = -1.0f;
    inline void update(float d) {
        if (d < min_depth) min_depth = d;
        if (d > max_depth) max_depth = d;
    }
    inline void reset() { min_depth = 9999.0f; max_depth = -1.0f; }
};

struct DepthImage {
    Eigen::Isometry3d T_l_w = Eigen::Isometry3d::Identity();
    std::vector<Pixel> data;
    int cols, rows;
    DepthImage(int r, int c) : data(r * c), cols(c), rows(r) {}
    inline void reset() { for (auto& p : data) p.reset(); }
};

class DetectorCore {
public:
    struct Params {
        float min_v = -7.0f * M_PI / 180.0f;
        float max_v = 52.0f * M_PI / 180.0f;
        int grid_h = 360; // 降低分辨率以平衡性能
        int grid_v = 60;
        float eps = 0.3f;  // 增加基础阈值
    };

    DetectorCore(const Params& p, size_t history_size = 7) 
        : p_(p), history_size_(history_size), head_(0) {
        for (size_t i = 0; i < history_size_; ++i)
            history_.push_back(std::make_unique<DepthImage>(p_.grid_v, p_.grid_h));
    }

    // 初始检测逻辑：增加深度比例系数
    bool initialCheck(const Eigen::Vector3d& pt_global, const Eigen::Isometry3d& T_l_w_cur) const {
        Eigen::Vector3d pt_l = T_l_w_cur.inverse() * pt_global;
        float d = pt_l.norm();
        if (d < 0.3f || d > 60.0f) return false;

        // 动态阈值：随距离增加，容忍度变大
        float dynamic_eps = p_.eps + (d * 0.01f); 

        int occluded = 0;
        for (const auto& img : history_) {
            Eigen::Vector3d pt_h = img->T_l_w.inverse() * pt_global;
            float dh = pt_h.norm();
            int u, v;
            if (project(pt_h, dh, u, v)) {
                if (dh > img->data[v * img->cols + u].max_depth + dynamic_eps) occluded++;
            }
        }
        return (occluded >= (int)history_size_ / 2 + 1);
    }

    // 更新当前帧点云（非并行部分，由主线程顺序调用）
    void addPoint(const Eigen::Vector3d& pt_global) {
        Eigen::Vector3d pt_l = history_[head_]->T_l_w.inverse() * pt_global;
        float d = pt_l.norm();
        int u, v;
        if (project(pt_l, d, u, v)) history_[head_]->data[v * p_.grid_h + u].update(d);
    }

    void nextFrame(const Eigen::Isometry3d& T_l_w) {
        head_ = (head_ + 1) % history_size_;
        history_[head_]->reset();
        history_[head_]->T_l_w = T_l_w;
    }

private:
    inline bool project(const Eigen::Vector3d& pt, float d, int& u, int& v) const {
        float phi = std::atan2(pt.y(), pt.x());
        float theta = std::asin(pt.z() / d);
        if (theta < p_.min_v || theta > p_.max_v) return false;
        u = static_cast<int>((phi + M_PI) / (2.0f * M_PI) * (p_.grid_h - 1));
        v = static_cast<int>((theta - p_.min_v) / (p_.max_v - p_.min_v) * (p_.grid_v - 1));
        return (u >= 0 && u < p_.grid_h && v >= 0 && v < p_.grid_v);
    }
    Params p_;
    size_t history_size_, head_;
    std::vector<std::unique_ptr<DepthImage>> history_;
};

class VoxelMap {
public:
    struct Voxel { int dyn_pts = 0; bool active = false; };
    VoxelMap(float res) : res_(res) {}

    // 优化：增加正向偏移量，防止负数索引导致哈希碰撞
    inline int64_t key(float x, float y, float z) const {
        int64_t ix = static_cast<int64_t>(std::floor(x / res_) + 10000);
        int64_t iy = static_cast<int64_t>(std::floor(y / res_) + 10000);
        int64_t iz = static_cast<int64_t>(std::floor(z / res_) + 10000);
        return (ix << 40) | (iy << 20) | iz;
    }

    // 顺序添加点（非多线程安全）
    void add(float x, float y, float z, bool dyn) {
        auto& v = voxels_[key(x, y, z)];
        if (dyn) v.dyn_pts++;
    }

    void bake(int threshold) {
        for (auto& kv : voxels_) if (kv.second.dyn_pts >= threshold) kv.second.active = true;
    }

    // 膨胀检查：仅检查 6 邻域以减少计算量，同时保持空间连通
    bool isDynamic(float x, float y, float z) const {
        int64_t ix = static_cast<int64_t>(std::floor(x / res_) + 10000);
        int64_t iy = static_cast<int64_t>(std::floor(y / res_) + 10000);
        int64_t iz = static_cast<int64_t>(std::floor(z / res_) + 10000);
        
        static const int dx[] = {0, 0, 0, 0, 0, 1, -1};
        static const int dy[] = {0, 0, 0, 1, -1, 0, 0};
        static const int dz[] = {0, 1, -1, 0, 0, 0, 0};

        for (int i = 0; i < 7; ++i) {
            int64_t k = ((ix + dx[i]) << 40) | ((iy + dy[i]) << 20) | (iz + dz[i]);
            auto it = voxels_.find(k);
            if (it != voxels_.end() && it->second.active) return true;
        }
        return false;
    }
    void clear() { voxels_.clear(); }

private:
    float res_;
    std::unordered_map<int64_t, Voxel> voxels_;
};

} // namespace m_detector

class MovingEventDetectorNode : public fins::Node {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void define() override {
        set_name("MovingEventDetector");
        register_input<sensor_msgs::msg::PointCloud2>("lidar_in", &MovingEventDetectorNode::on_lidar);
        register_input<geometry_msgs::msg::TransformStamped>("$T_{odom}^{base}$", &MovingEventDetectorNode::on_odom);
        register_input<geometry_msgs::msg::TransformStamped>("$T_{base}^{lidar}$", &MovingEventDetectorNode::on_extrinsic);
        register_output<sensor_msgs::msg::PointCloud2>("dynamic_cloud");
        register_output<sensor_msgs::msg::PointCloud2>("static_cloud");
    }

    void initialize() override {
        fins::ParamLoader cfg("MovingEventDetector");
        m_detector::DetectorCore::Params p;
        p.eps = cfg.get("epsilon_d", 0.3f);
        p.grid_h = cfg.get("grid_h", 360);
        p.grid_v = cfg.get("grid_v", 60);
        
        voxel_res_ = cfg.get("voxel_res", 0.2f);
        voxel_th_ = cfg.get("voxel_min_pts", 5); // 提高阈值，过滤噪声
        
        detector_ = std::make_unique<m_detector::DetectorCore>(p, cfg.get("history_size", 7));
        logger->info("M-Detector Parallel & Robust Version Initialized.");
    }

private:
    void on_odom(const fins::Msg<geometry_msgs::msg::TransformStamped>& m) {
        T_w_b_ = tf2::transformToEigen(*m);
        received_odom_ = true;
    }

    void on_extrinsic(const fins::Msg<geometry_msgs::msg::TransformStamped>& m) {
        T_b_l_ = tf2::transformToEigen(*m);
        received_ext_ = true;
    }

    void on_lidar(const fins::Msg<sensor_msgs::msg::PointCloud2>& msg) {
        if (!received_odom_ || !received_ext_) return;

        Eigen::Isometry3d T_w_l = T_w_b_ * T_b_l_;
        detector_->nextFrame(T_w_l);

        int x_off = -1, y_off = -1, z_off = -1, i_off = -1;
        for (const auto& f : msg->fields) {
            if (f.name == "x") x_off = f.offset;
            else if (f.name == "y") y_off = f.offset;
            else if (f.name == "z") z_off = f.offset;
            else if (f.name == "intensity") i_off = f.offset;
        }
        if (x_off == -1) return;

        size_t count = msg->width * msg->height;
        m_detector::VoxelMap vmap(voxel_res_);
        
        struct Temp { m_detector::PointXYZI p; Eigen::Vector3d pg; bool is_dyn; };
        std::vector<Temp> pts(count);

        // Pass 1: 并行计算变换与初始检测
        #pragma omp parallel for schedule(dynamic, 512)
        for (size_t i = 0; i < count; ++i) {
            const uint8_t* d = &msg->data[i * msg->point_step];
            float x = *reinterpret_cast<const float*>(d + x_off);
            float y = *reinterpret_cast<const float*>(d + y_off);
            float z = *reinterpret_cast<const float*>(d + z_off);
            float intensity = (i_off != -1) ? *reinterpret_cast<const float*>(d + i_off) : 0.0f;

            Eigen::Vector3d p_global = T_w_l * Eigen::Vector3d(x, y, z);
            // 只读操作，线程安全
            bool is_dyn = detector_->initialCheck(p_global, T_w_l);
            
            pts[i] = {{x, y, z, intensity}, p_global, is_dyn};
        }

        // Pass 2: 顺序填充体素地图与更新历史 (Map写入非线程安全)
        for (size_t i = 0; i < count; ++i) {
            vmap.add(pts[i].p.x, pts[i].p.y, pts[i].p.z, pts[i].is_dyn);
            detector_->addPoint(pts[i].pg);
        }

        vmap.bake(voxel_th_);

        // Pass 3: 并行收集动态点
        std::vector<m_detector::PointXYZI> dyn_pts, sta_pts;
        dyn_pts.reserve(count / 10);
        sta_pts.reserve(count);

        // 为了避免在并行中使用 push_back 导致的锁竞争，先用标记位
        std::vector<int8_t> final_is_dyn(count);
        #pragma omp parallel for schedule(dynamic, 512)
        for (size_t i = 0; i < count; ++i) {
            final_is_dyn[i] = vmap.isDynamic(pts[i].p.x, pts[i].p.y, pts[i].p.z) ? 1 : 0;
        }

        for (size_t i = 0; i < count; ++i) {
            if (final_is_dyn[i]) dyn_pts.push_back(pts[i].p);
            else sta_pts.push_back(pts[i].p);
        }

        publish(dyn_pts, "dynamic_cloud", msg);
        publish(sta_pts, "static_cloud", msg);
    }

    void publish(const std::vector<m_detector::PointXYZI>& pts, const std::string& chan, 
                 const fins::Msg<sensor_msgs::msg::PointCloud2>& orig) {
        if (pts.empty()) return;
        sensor_msgs::msg::PointCloud2 m;
        m.header = orig->header;
        m.height = 1; m.width = pts.size();
        m.point_step = sizeof(m_detector::PointXYZI);
        m.row_step = m.point_step * m.width;
        m.is_dense = true;
        
        static const std::vector<std::string> names = {"x", "y", "z", "intensity"};
        for (size_t i = 0; i < names.size(); ++i) {
            sensor_msgs::msg::PointField f;
            f.name = names[i]; f.offset = i * 4;
            f.datatype = sensor_msgs::msg::PointField::FLOAT32; f.count = 1;
            m.fields.push_back(f);
        }
        m.data.resize(pts.size() * sizeof(m_detector::PointXYZI));
        std::memcpy(m.data.data(), pts.data(), m.data.size());
        send(chan, m, orig.acq_time);
    }

    Eigen::Isometry3d T_w_b_ = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d T_b_l_ = Eigen::Isometry3d::Identity();
    bool received_odom_ = false, received_ext_ = false;
    float voxel_res_ = 0.2f;
    int voxel_th_ = 5;
    std::unique_ptr<m_detector::DetectorCore> detector_;
};

EXPORT_NODE(MovingEventDetectorNode)
DEFINE_PLUGIN_ENTRY(fins::STATELESS)