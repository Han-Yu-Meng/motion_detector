/*******************************************************************************
 * Copyright (c) 2025.
 * IWIN-FINS Lab, Shanghai Jiao Tong University, Shanghai, China.
 * All rights reserved.
 ******************************************************************************/

#include <fins/node.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

#if __has_include(<tf2_eigen/tf2_eigen.hpp>)
  #include <tf2_eigen/tf2_eigen.hpp>
#else
  #include <tf2_eigen/tf2_eigen.h>
#endif

#include <Eigen/Dense>
#include <mutex>
#include <vector>
#include <cmath>
#include <limits>
#include <memory>
#include <algorithm>
#include <cstring>
#include <chrono>

namespace m_detector {

inline float fast_atan2(float y, float x) {
    if (x == 0.0f) {
        if (y > 0.0f) return M_PI / 2.0f;
        if (y < 0.0f) return -M_PI / 2.0f;
        return 0.0f;
    }
    float atan;
    float z = y / x;
    if (std::abs(z) < 1.0f) {
        atan = z / (1.0f + 0.28f * z * z);
        if (x < 0.0f) {
            if (y < 0.0f) return atan - M_PI;
            return atan + M_PI;
        }
    } else {
        atan = M_PI / 2.0f - z / (z * z + 0.28f);
        if (y < 0.0f) return atan - M_PI;
    }
    return atan;
}

inline float fast_asin(float x) {
    float negate = static_cast<float>(x < 0.0f);
    x = std::abs(x);
    if (x >= 1.0f) return (negate != 0.0f) ? -M_PI / 2.0f : M_PI / 2.0f;
    float ret = -0.0187293f;
    ret *= x;
    ret += 0.0742610f;
    ret *= x;
    ret -= 0.2121144f;
    ret *= x;
    ret += 1.5707288f;
    ret = M_PI / 2.0f - std::sqrt(1.0f - x) * ret;
    return ret - 2.0f * negate * ret;
}

struct Pixel {
    float min_depth = std::numeric_limits<float>::max();
    float max_depth = -std::numeric_limits<float>::max();
    
    inline void update(float depth) {
        if (depth < min_depth) min_depth = depth;
        if (depth > max_depth) max_depth = depth;
    }
    
    inline void reset() {
        min_depth = std::numeric_limits<float>::max();
        max_depth = -std::numeric_limits<float>::max();
    }
};

struct DepthImage {
    Eigen::Isometry3d T_world_lidar = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d T_lidar_world = Eigen::Isometry3d::Identity(); 
    std::vector<Pixel> data;
    int cols = 0;
    int rows = 0;

    DepthImage(int r, int c) : cols(c), rows(r), data(r * c) {}

    inline void reset() {
        for (auto& pix : data) pix.reset();
    }

    inline Pixel& at(int u, int v) {
        return data[v * cols + u];
    }
};

class DetectorCore {
public:
    struct Params {
        float min_angle_h = -M_PI;
        float max_angle_h = M_PI;
        float min_angle_v = -7.0f * M_PI / 180.0f; 
        float max_angle_v = 52.0f * M_PI / 180.0f; 
        int grid_h = 360;
        int grid_v = 60;
        float epsilon_d = 0.15f; 
        int neighborhood_size = 1;
    };

    DetectorCore(const Params& params, size_t history_size = 5) 
        : params_(params), history_size_(history_size), head_(0) {
        for (size_t i = 0; i < history_size_; ++i) {
            history_images_.push_back(std::make_unique<DepthImage>(params_.grid_v, params_.grid_h));
        }
        static_background_ = std::make_unique<DepthImage>(params_.grid_v, params_.grid_h);
    }

    void updatePose(const Eigen::Isometry3d& T_world_lidar, const Eigen::Isometry3d& T_lidar_world) {
        current_pose_ = T_world_lidar;
        current_pose_inv_ = T_lidar_world;
    }

    bool detectPoint(const Eigen::Vector3d& pt_global, float& out_depth) {
        Eigen::Vector3d pt_local = current_pose_inv_ * pt_global;
        float depth = pt_local.norm();
        out_depth = depth;

        if (depth < 0.1f) return false;

        int u, v;
        if (!projectToGrid(pt_local, depth, u, v)) return false;

        if (checkMapConsistency(u, v, depth)) return false;

        bool test_perpendicular = false;
        bool test_parallel_away = true;
        bool test_parallel_toward = true;

        int occluded_count = 0;
        int occluding_count = 0;

        for (size_t i = 0; i < history_size_; ++i) {
            const auto& img = history_images_[i];
            Eigen::Vector3d pt_hist = img->T_lidar_world * pt_global; 
            float d_hist = pt_hist.norm();
            
            int uh, vh;
            if (!projectToGrid(pt_hist, d_hist, uh, vh)) {
                test_parallel_away = false;
                test_parallel_toward = false;
                continue;
            }

            float min_d_neighbor, max_d_neighbor;
            getNeighborhoodMinMax(*img, uh, vh, min_d_neighbor, max_d_neighbor);

            if (d_hist > max_d_neighbor + params_.epsilon_d) {
                occluded_count++;
            } else {
                test_parallel_away = false;
            }

            if (d_hist < min_d_neighbor - params_.epsilon_d) {
                occluding_count++;
            } else {
                test_parallel_toward = false;
            }
        }

        if (occluding_count >= static_cast<int>(history_size_ / 2) + 1) {
            test_perpendicular = true;
        }

        bool is_moving_event = (test_perpendicular || test_parallel_away || test_parallel_toward);

        if (!is_moving_event) {
            updateStaticMap(u, v, depth);
        }

        return is_moving_event;
    }

    void commitFrame() {
        head_ = (head_ + 1) % history_size_;
        history_images_[head_]->reset();
        history_images_[head_]->T_world_lidar = current_pose_;
        history_images_[head_]->T_lidar_world = current_pose_inv_;
    }

    void addPointToCurrentFrame(const Eigen::Vector3d& pt_global) {
        Eigen::Vector3d pt_local = current_pose_inv_ * pt_global;
        float depth = pt_local.norm();
        int u, v;
        if (projectToGrid(pt_local, depth, u, v)) {
            history_images_[head_]->at(u, v).update(depth);
        }
    }

private:
    inline bool projectToGrid(const Eigen::Vector3d& pt, float depth, int& u, int& v) const {
        float phi = fast_atan2(pt.y(), pt.x());
        float theta = fast_asin(pt.z() / depth);

        if (phi < params_.min_angle_h || phi > params_.max_angle_h ||
            theta < params_.min_angle_v || theta > params_.max_angle_v) {
            return false;
        }

        u = static_cast<int>((phi - params_.min_angle_h) / (params_.max_angle_h - params_.min_angle_h) * (params_.grid_h - 1));
        v = static_cast<int>((theta - params_.min_angle_v) / (params_.max_angle_v - params_.min_angle_v) * (params_.grid_v - 1));
        return true;
    }

    void getNeighborhoodMinMax(const DepthImage& img, int u, int v, float& min_d, float& max_d) const {
        min_d = std::numeric_limits<float>::max();
        max_d = -std::numeric_limits<float>::max();
        int half_w = params_.neighborhood_size;

        for (int dv = -half_w; dv <= half_w; ++dv) {
            int nv = v + dv;
            if (nv < 0 || nv >= params_.grid_v) continue;
            for (int du = -half_w; du <= half_w; ++du) {
                int nu = (u + du + params_.grid_h) % params_.grid_h;
                
                const auto& pix = img.data[nv * img.cols + nu];
                if (pix.min_depth < min_d) min_d = pix.min_depth;
                if (pix.max_depth > max_d) max_d = pix.max_depth;
            }
        }
    }

    bool checkMapConsistency(int u, int v, float depth) {
        float min_d, max_d;
        getNeighborhoodMinMax(*static_background_, u, v, min_d, max_d);
        return (depth >= min_d - params_.epsilon_d && depth <= max_d + params_.epsilon_d);
    }

    void updateStaticMap(int u, int v, float depth) {
        auto& pix = static_background_->at(u, v);
        if (pix.min_depth > 9999.0f) {
            pix.min_depth = depth;
            pix.max_depth = depth;
        } else {
            pix.min_depth = 0.98f * pix.min_depth + 0.02f * depth;
            pix.max_depth = 0.98f * pix.max_depth + 0.02f * depth;
        }
    }

    Params params_;
    size_t history_size_;
    size_t head_;
    
    Eigen::Isometry3d current_pose_ = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d current_pose_inv_ = Eigen::Isometry3d::Identity();

    std::vector<std::unique_ptr<DepthImage>> history_images_;
    std::unique_ptr<DepthImage> static_background_;
};

struct __attribute__((packed)) PointXYZI {
    float x;
    float y;
    float z;
    float intensity;
};

} // namespace m_detector

class MovingEventDetectorNode : public fins::Node {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct Config {
        double epsilon_d = 0.15;
        int history_size = 5;
        int neighborhood_size = 1;
        int grid_h = 360;
        int grid_v = 60;
        double min_angle_v_deg = -7.0;
        double max_angle_v_deg = 52.0;
        double blind_zone_radius = 0.15;
        std::string base_frame = "base_link";
        std::string lidar_frame = "base_lidar";
    };

    void define() override {
        set_name("MovingEventDetector");
        set_category("Perception");

        register_input<sensor_msgs::msg::PointCloud2>("lidar_in", &MovingEventDetectorNode::on_lidar_callback);
        register_input<geometry_msgs::msg::TransformStamped>("$T_{odom}^{base}$", &MovingEventDetectorNode::on_tf_world_to_base_callback);
        register_input<geometry_msgs::msg::TransformStamped>("$T_{base}^{lidar}$", &MovingEventDetectorNode::on_tf_base_to_lidar_callback);

        register_output<sensor_msgs::msg::PointCloud2>("dynamic_cloud");
        register_output<sensor_msgs::msg::PointCloud2>("static_cloud");
    }

    void initialize() override {
        fins::ParamLoader config("MovingEventDetector");

        config_.epsilon_d = config.get("epsilon_d", 0.15);
        config_.history_size = config.get("history_size", 5);
        config_.neighborhood_size = config.get("neighborhood_size", 1);
        config_.grid_h = config.get("grid_h", 360);
        config_.grid_v = config.get("grid_v", 60);
        config_.min_angle_v_deg = config.get("min_angle_v", -7.0);
        config_.max_angle_v_deg = config.get("max_angle_v", 52.0);
        config_.blind_zone_radius = config.get("blind_zone_radius", 0.15);
        config_.base_frame = config.get("base_frame", "base_link");
        config_.lidar_frame = config.get("lidar_frame", "base_lidar");

        m_detector::DetectorCore::Params params;
        params.grid_h = config_.grid_h;
        params.grid_v = config_.grid_v;
        params.epsilon_d = static_cast<float>(config_.epsilon_d);
        params.neighborhood_size = config_.neighborhood_size;
        params.min_angle_v = static_cast<float>(config_.min_angle_v_deg * M_PI / 180.0);
        params.max_angle_v = static_cast<float>(config_.max_angle_v_deg * M_PI / 180.0);

        detector_ = std::make_unique<m_detector::DetectorCore>(params, static_cast<size_t>(config_.history_size));
    }

private:
    void on_tf_world_to_base_callback(const fins::Msg<geometry_msgs::msg::TransformStamped>& msg) {
        std::lock_guard<std::mutex> lock(pose_mtx_);
        tf_world_to_base_ = tf2::transformToEigen(*msg);
        tf_world_to_base_inv_ = tf_world_to_base_.inverse();
        tf_world_to_base_received_ = true;
    }

    void on_tf_base_to_lidar_callback(const fins::Msg<geometry_msgs::msg::TransformStamped>& msg) {
        std::lock_guard<std::mutex> lock(pose_mtx_);
        tf_base_to_lidar_ = tf2::transformToEigen(*msg);
        tf_base_to_lidar_inv_ = tf_base_to_lidar_.inverse();
        tf_base_to_lidar_received_ = true;
    }

    void on_lidar_callback(const fins::Msg<sensor_msgs::msg::PointCloud2>& msg) {
        auto t_start = std::chrono::high_resolution_clock::now();
        
        Eigen::Isometry3d T_W_B; // World -> Base
        Eigen::Isometry3d T_B_W; // Base -> World
        Eigen::Isometry3d T_B_L; // Base -> Lidar
        Eigen::Isometry3d T_L_B; // Lidar -> Base
        bool poses_received = false;

        {
            std::lock_guard<std::mutex> lock(pose_mtx_);
            if (tf_world_to_base_received_ && tf_base_to_lidar_received_) {
                T_W_B = tf_world_to_base_;
                T_B_W = tf_world_to_base_inv_;
                T_B_L = tf_base_to_lidar_;
                T_L_B = tf_base_to_lidar_inv_;
                poses_received = true;
            }
        }
        
        auto t_after_lock = std::chrono::high_resolution_clock::now();
        double lock_ms = std::chrono::duration<double, std::milli>(t_after_lock - t_start).count();
        logger->info("[DIAG 1] Wait for pose lock took {} ms.", lock_ms);

        if (!poses_received) {
            logger->warn("[DIAG 1] Lidar callback ignored: Poses not yet received.");
            return;
        }
        if (msg->data.empty()) {
            logger->warn("[DIAG 1] Lidar callback ignored: Message data is empty.");
            return;
        }

        // [诊断日志段 2]：字段解析耗时
        int x_offset = -1, y_offset = -1, z_offset = -1, intensity_offset = -1;
        for (const auto& field : msg->fields) {
            if (field.name == "x") x_offset = field.offset;
            else if (field.name == "y") y_offset = field.offset;
            else if (field.name == "z") z_offset = field.offset;
            else if (field.name == "intensity") intensity_offset = field.offset;
        }

        auto t_after_fields = std::chrono::high_resolution_clock::now();
        double fields_ms = std::chrono::duration<double, std::milli>(t_after_fields - t_after_lock).count();
        logger->info("[DIAG 2] Field offsets parsing took {} ms.", fields_ms);

        if (x_offset == -1 || y_offset == -1 || z_offset == -1) {
            logger->error("[DIAG 2] Critical fields x/y/z not found in PointCloud2!");
            return;
        }

        std::vector<m_detector::PointXYZI> dynamic_points;
        std::vector<m_detector::PointXYZI> static_points;

        size_t total_points = msg->width * msg->height;
        
        // 边界安全防御检查
        if (msg->data.size() < total_points * msg->point_step) {
            logger->error("[DIAG 2] PointCloud2 data size mismatch! Expected at least {} bytes, got {} bytes.", 
                          total_points * msg->point_step, msg->data.size());
            return;
        }

        dynamic_points.reserve(total_points / 4); 
        static_points.reserve(total_points);

        // 核心双向位姿变换组合：
        // T_world_lidar (T_W_L) = T_world_base * T_base_lidar
        Eigen::Isometry3d T_W_L = T_W_B * T_B_L;
        // T_lidar_world (T_L_W) = T_lidar_base * T_base_world
        Eigen::Isometry3d T_L_W = T_L_B * T_B_W;

        detector_->updatePose(T_W_L, T_L_W);

        const double blind_sq = config_.blind_zone_radius * config_.blind_zone_radius;

        // [诊断日志段 3]：点云遍历耗时与追踪
        logger->info("[DIAG 3] Start processing {} points...", total_points);
        auto t_loop_start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < total_points; ++i) {
            if (i > 0 && i % 5000 == 0) {
                auto t_now = std::chrono::high_resolution_clock::now();
                double partial_ms = std::chrono::duration<double, std::milli>(t_now - t_loop_start).count();
                logger->info("[DIAG 3-Progress] Processed {} / {} points (Time elapsed: {} ms)", i, total_points, partial_ms);
            }

            const uint8_t* ptr = &msg->data[i * msg->point_step];
            
            // 输入点云已经处于 base_link 坐标系下
            float x = *reinterpret_cast<const float*>(ptr + x_offset);
            float y = *reinterpret_cast<const float*>(ptr + y_offset);
            float z = *reinterpret_cast<const float*>(ptr + z_offset);
            
            if ((x*x + y*y + z*z) < blind_sq) {
                continue;
            }

            float intensity = 0.0f;
            if (intensity_offset != -1) {
                intensity = *reinterpret_cast<const float*>(ptr + intensity_offset);
            }

            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
                continue;
            }

            // 将点从 base_link 映射到 odom 全局坐标系
            Eigen::Vector3d pt_global = T_W_B * Eigen::Vector3d(x, y, z);
            float depth = 0.0f;

            m_detector::PointXYZI out_pt;
            out_pt.x = x; // 保持输出坐标系为 base_link
            out_pt.y = y;
            out_pt.z = z;
            out_pt.intensity = intensity;

            if (detector_->detectPoint(pt_global, depth)) {
                dynamic_points.push_back(out_pt);
            } else {
                static_points.push_back(out_pt);
            }

            detector_->addPointToCurrentFrame(pt_global);
        }

        detector_->commitFrame();

        auto t_loop_end = std::chrono::high_resolution_clock::now();
        double loop_ms = std::chrono::duration<double, std::milli>(t_loop_end - t_loop_start).count();
        logger->info("[DIAG 3] Point cloud processing loop completed. Duration: {} ms. Dynamic: {}, Static: {}", 
                     loop_ms, dynamic_points.size(), static_points.size());

        // [诊断日志段 4]：转换与序列化耗时
        sensor_msgs::msg::PointCloud2 dynamic_msg;
        sensor_msgs::msg::PointCloud2 static_msg;

        populate_point_cloud(dynamic_msg, dynamic_points, msg->header);
        populate_point_cloud(static_msg, static_points, msg->header);

        auto t_populate_end = std::chrono::high_resolution_clock::now();
        double populate_ms = std::chrono::duration<double, std::milli>(t_populate_end - t_loop_end).count();
        logger->info("[DIAG 4] PointCloud2 message generation took {} ms.", populate_ms);

        // [诊断日志段 5]：发送投递耗时
        send("dynamic_cloud", dynamic_msg, msg.acq_time);
        send("static_cloud", static_msg, msg.acq_time);

        auto t_send_end = std::chrono::high_resolution_clock::now();
        double send_ms = std::chrono::duration<double, std::milli>(t_send_end - t_populate_end).count();
        double total_cb_ms = std::chrono::duration<double, std::milli>(t_send_end - t_start).count();
        
        logger->info("[DIAG 5] PointCloud2 publish took {} ms. [TOTAL CALLBACK TIME: {} ms]", send_ms, total_cb_ms);
        logger->info("------------------------------------------------------------------");
    }

    void populate_point_cloud(sensor_msgs::msg::PointCloud2& msg, 
                              const std::vector<m_detector::PointXYZI>& points, 
                              const std_msgs::msg::Header& header) {
        msg.header = header;
        msg.header.frame_id = config_.base_frame; // 强制修改为指定的 base_frame
        msg.height = 1;
        msg.width = points.size();
        msg.is_bigendian = false;
        msg.point_step = sizeof(m_detector::PointXYZI);
        msg.row_step = msg.point_step * msg.width;
        msg.is_dense = true;

        msg.fields.clear();
        msg.fields.resize(4);

        msg.fields[0].name = "x";
        msg.fields[0].offset = 0;
        msg.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
        msg.fields[0].count = 1;

        msg.fields[1].name = "y";
        msg.fields[1].offset = 4;
        msg.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
        msg.fields[1].count = 1;

        msg.fields[2].name = "z";
        msg.fields[2].offset = 8;
        msg.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
        msg.fields[2].count = 1;

        msg.fields[3].name = "intensity";
        msg.fields[3].offset = 12;
        msg.fields[3].datatype = sensor_msgs::msg::PointField::FLOAT32;
        msg.fields[3].count = 1;

        msg.data.resize(points.size() * sizeof(m_detector::PointXYZI));
        std::memcpy(msg.data.data(), points.data(), msg.data.size());
    }

    Config config_;
    std::mutex pose_mtx_;
    
    std::unique_ptr<m_detector::DetectorCore> detector_;
    Eigen::Isometry3d tf_world_to_lidar_{Eigen::Isometry3d::Identity()};
    Eigen::Isometry3d tf_world_to_lidar_inv_{Eigen::Isometry3d::Identity()};
    
    Eigen::Isometry3d tf_world_to_base_{Eigen::Isometry3d::Identity()};
    Eigen::Isometry3d tf_world_to_base_inv_{Eigen::Isometry3d::Identity()};
    Eigen::Isometry3d tf_base_to_lidar_{Eigen::Isometry3d::Identity()};
    Eigen::Isometry3d tf_base_to_lidar_inv_{Eigen::Isometry3d::Identity()};
    
    bool tf_received_{false};
    bool tf_world_to_base_received_{false};
    bool tf_base_to_lidar_received_{false};
};

EXPORT_NODE(MovingEventDetectorNode)
DEFINE_PLUGIN_ENTRY(fins::STATELESS)