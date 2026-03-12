#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>

// ==========================================
// 1. 数据结构定义
// ==========================================
struct GyroSample {
    Eigen::Vector3d data;   // rad/s (gyro) 或 m/s² (accel)
    double timestamp_ms;    // 毫秒时间戳
};

struct QuatSample {
    Eigen::Quaterniond quat;  // 姿态四元数 (w, x, y, z)
    double timestamp_ms;
};

// ==========================================
// 2. Madgwick AHRS 实现
// ==========================================
class MadgwickFilter {
public:
    MadgwickFilter(double beta = 0.1) : beta_(beta), q_(1, 0, 0, 0) {}
    
    // 单步更新（陀螺仪 + 加速度计）
    void update(const Eigen::Vector3d& gyro,    // rad/s
                const Eigen::Vector3d& accel,   // m/s² (不需要归一化)
                double dt) {                    // 秒
        
        Eigen::Quaterniond q = q_;
        
        // 归一化加速度计
        Eigen::Vector3d a = accel.normalized();
        
        // 梯度下降算法（Madgwick 论文公式 30-33）
        Eigen::Vector3d f;
        f << 2*(q.x()*q.z() - q.w()*q.y()) - a.x(),
             2*(q.w()*q.x() + q.y()*q.z()) - a.y(),
             2*(0.5 - q.x()*q.x() - q.y()*q.y()) - a.z();
        
        Eigen::Matrix<double, 3, 4> J;
        J << -2*q.y(),  2*q.z(), -2*q.w(), 2*q.x(),
              2*q.x(),  2*q.w(),  2*q.z(), 2*q.y(),
              0,       -4*q.x(), -4*q.y(), 0;
        
        Eigen::Vector4d gradient = J.transpose() * f;
        gradient.normalize();
        
        // 应用陀螺仪偏差修正（公式 12）
        Eigen::Vector3d gyro_corrected = gyro - beta_ * Eigen::Vector3d(gradient[1], gradient[2], gradient[3]);
        
        // 积分四元数导数（公式 13）
        Eigen::Quaterniond q_dot;
        q_dot.w() = 0.5 * (-q.x()*gyro_corrected.x() - q.y()*gyro_corrected.y() - q.z()*gyro_corrected.z());
        q_dot.x() = 0.5 * ( q.w()*gyro_corrected.x() + q.y()*gyro_corrected.z() - q.z()*gyro_corrected.y());
        q_dot.y() = 0.5 * ( q.w()*gyro_corrected.y() - q.x()*gyro_corrected.z() + q.z()*gyro_corrected.x());
        q_dot.z() = 0.5 * ( q.w()*gyro_corrected.z() + q.x()*gyro_corrected.y() - q.y()*gyro_corrected.x());
        
        // 积分
        q.w() += q_dot.w() * dt;
        q.x() += q_dot.x() * dt;
        q.y() += q_dot.y() * dt;
        q.z() += q_dot.z() * dt;
        q.normalize();
        
        q_ = q;
    }
    
    Eigen::Quaterniond getQuaternion() const { return q_; }
    void reset() { q_ = Eigen::Quaterniond(1, 0, 0, 0); }

private:
    double beta_;           // 算法增益，越大对加速度计信任越高（0.01-0.5）
    Eigen::Quaterniond q_;  // 当前姿态
};

// ==========================================
// 3. 视频稳定器（核心类）
// ==========================================
class VideoStabilizer {
public:
    struct SmoothParams {
        double base_alpha = 0.1;        // 基础平滑系数 (0-1, 越小越平滑)
        double roll_threshold = 5.0;    // Roll 轴速度阈值 (deg/s)，超过此值减少平滑
        double pitch_threshold = 5.0;   // Pitch 轴速度阈值
        double yaw_threshold = 20.0;    // Yaw 轴速度阈值（通常设置更高，保留转向）
        bool use_velocity_dampened = true;  // 是否启用速度阻尼
    };

    // 主处理流程
    bool process(const std::vector<GyroSample>& gyro_data,
                 const std::vector<GyroSample>& accel_data,
                 const SmoothParams& params = SmoothParams()) {
        
        // 1. Madgwick 积分
        if (!integrateIMU(gyro_data, accel_data)) {
            return false;
        }
        
        // 2. 双向指数平滑 + 速度阻尼
        smoothTrajectory(params);
        
        return true;
    }
    
    // 获取原始姿态（用于对比）
    Eigen::Quaterniond getRawOrientation(double timestamp_ms) const {
        return interpolateQuat(raw_quats_, timestamp_ms);
    }
    
    // 获取平滑后姿态（用于渲染）
    Eigen::Quaterniond getSmoothOrientation(double timestamp_ms) const {
        return interpolateQuat(smooth_quats_, timestamp_ms);
    }
    
    // 计算矫正矩阵：将原始画面旋转到平滑视角
    // 使用方法：corrected_frame = correction_matrix * original_frame
    Eigen::Matrix3d computeCorrectionMatrix(double timestamp_ms) const {
        Eigen::Quaterniond q_raw = getRawOrientation(timestamp_ms);
        Eigen::Quaterniond q_smooth = getSmoothOrientation(timestamp_ms);
        
        // 关键：矫正矩阵 = 平滑姿态 * 原始姿态的逆
        // 这样可以将原始相机坐标系对齐到平滑后的坐标系
        Eigen::Quaterniond q_correction = q_smooth * q_raw.inverse();
        
        return q_correction.toRotationMatrix();
    }
    
    // 获取完整轨迹（用于可视化，类似 GYROflow 底部波形图）
    const std::vector<QuatSample>& getRawTrajectory() const { return raw_quats_; }
    const std::vector<QuatSample>& getSmoothTrajectory() const { return smooth_quats_; }

private:
    std::vector<QuatSample> raw_quats_;     // Madgwick 原始输出
    std::vector<QuatSample> smooth_quats_;  // 平滑后输出
    
    // ==========================================
    // 1. Madgwick 积分（时间戳对齐）
    // ==========================================
    bool integrateIMU(const std::vector<GyroSample>& gyro_data,
                      const std::vector<GyroSample>& accel_data) {
        
        if (gyro_data.empty() || accel_data.empty()) return false;
        
        raw_quats_.clear();
        MadgwickFilter madgwick(0.1);  // beta=0.1，平衡陀螺仪和加速度计
        
        size_t accel_idx = 0;
        
        for (size_t i = 0; i < gyro_data.size(); ++i) {
            const auto& gyro = gyro_data[i];
            double ts = gyro.timestamp_ms;
            
            // 找到对应时间戳的加速度计数据（线性插值）
            Eigen::Vector3d accel = interpolateAccel(accel_data, ts, accel_idx);
            
            // 计算 dt（毫秒转秒）
            double dt = (i == 0) ? 0.005 : (ts - gyro_data[i-1].timestamp_ms) / 1000.0;
            dt = std::max(dt, 0.001);  // 防止除零，最小 1ms
            dt = std::min(dt, 0.1);    // 防止异常跳变，最大 100ms
            
            // 更新 Madgwick
            madgwick.update(gyro.data, accel, dt);
            
            // 保存结果
            QuatSample sample;
            sample.quat = madgwick.getQuaternion();
            sample.timestamp_ms = ts;
            raw_quats_.push_back(sample);
        }
        
        return true;
    }
    
    // 加速度计插值（时间戳对齐）
    Eigen::Vector3d interpolateAccel(const std::vector<GyroSample>& accel_data,
                                     double timestamp_ms,
                                     size_t& start_idx) const {
        // 找到相邻的两个样本
        while (start_idx < accel_data.size() - 1 && 
               accel_data[start_idx + 1].timestamp_ms < timestamp_ms) {
            start_idx++;
        }
        
        if (start_idx >= accel_data.size() - 1) {
            return accel_data.back().data;
        }
        
        const auto& s1 = accel_data[start_idx];
        const auto& s2 = accel_data[start_idx + 1];
        
        if (s2.timestamp_ms == s1.timestamp_ms) return s1.data;
        
        double t = (timestamp_ms - s1.timestamp_ms) / (s2.timestamp_ms - s1.timestamp_ms);
        t = std::max(0.0, std::min(1.0, t));  // clamp 到 [0,1]
        
        return s1.data * (1 - t) + s2.data * t;
    }
    
    // ==========================================
    // 2. 双向指数平滑 + 速度阻尼（核心）
    // ==========================================
    void smoothTrajectory(const SmoothParams& params) {
        if (raw_quats_.empty()) return;
        
        smooth_quats_ = raw_quats_;  // 复制时间戳结构
        
        // 第一阶段：前向平滑（Forward pass）
        for (size_t i = 1; i < raw_quats_.size(); ++i) {
            double alpha = computeAdaptiveAlpha(i, params);
            
            // 四元数 SLERP 平滑
            Eigen::Quaterniond q_curr = raw_quats_[i].quat;
            Eigen::Quaterniond q_prev = smooth_quats_[i-1].quat;
            
            // 确保最短路径（四元数双覆盖）
            if (q_curr.dot(q_prev) < 0) q_curr = Eigen::Quaterniond(-q_curr.coeffs());
            
            smooth_quats_[i].quat = q_curr.slerp(alpha, q_prev);
        }
        
        // 第二阶段：后向平滑（Backward pass）- 消除相位延迟！
        // 这是 GYROflow 丝滑的关键：离线算法可以"看到未来"
        for (int i = smooth_quats_.size() - 2; i >= 0; --i) {
            double alpha = params.base_alpha;  // 后向通常使用固定系数
            
            Eigen::Quaterniond q_curr = smooth_quats_[i].quat;
            Eigen::Quaterniond q_next = smooth_quats_[i+1].quat;
            
            if (q_curr.dot(q_next) < 0) q_curr = Eigen::Quaterniond(-q_curr.coeffs());
            
            smooth_quats_[i].quat = q_curr.slerp(alpha, q_next);
        }
    }
    
    // 速度阻尼：根据各轴角速度动态调整平滑系数
    double computeAdaptiveAlpha(size_t idx, const SmoothParams& params) {
        if (idx == 0 || !params.use_velocity_dampened) {
            return params.base_alpha;
        }
        
        // 计算当前角速度（四元数微分近似）
        const Eigen::Quaterniond& q_curr = raw_quats_[idx].quat;
        const Eigen::Quaterniond& q_prev = raw_quats_[idx-1].quat;
        double dt = (raw_quats_[idx].timestamp_ms - raw_quats_[idx-1].timestamp_ms) / 1000.0;
        
        // 四元数差分转角速度
        Eigen::Quaterniond q_diff = q_curr * q_prev.inverse();
        if (q_diff.w() < 0) q_diff = Eigen::Quaterniond(-q_diff.coeffs());
        
        // 转换为旋转矢量（轴角表示）
        double angle = 2.0 * std::acos(std::min(1.0, std::abs(q_diff.w())));
        Eigen::Vector3d axis(q_diff.x(), q_diff.y(), q_diff.z());
        if (axis.norm() > 1e-6) {
            axis.normalize();
        }
        Eigen::Vector3d angular_vel = axis * angle / dt;  // rad/s
        
        // 转为 deg/s 便于调试
        Eigen::Vector3d vel_deg = angular_vel * 180.0 / M_PI;
        
        // 分轴速度阻尼逻辑
        double damping = 1.0;
        
        // Roll 轴（X轴）：始终高平滑，抑制高频抖动
        if (std::abs(vel_deg.x()) > params.roll_threshold) {
            damping = std::min(damping, 0.8);  // 即使高速也保持 80% 平滑
        }
        
        // Pitch 轴（Y轴）：中等平滑
        if (std::abs(vel_deg.y()) > params.pitch_threshold) {
            damping = std::min(damping, 0.5);
        }
        
        // Yaw 轴（Z轴）：低速时平滑，高速时保留（避免转向滞后）
        if (std::abs(vel_deg.z()) > params.yaw_threshold) {
            damping = std::min(damping, 0.1);  // 快速转向时几乎不平滑
        }
        
        return params.base_alpha * damping;
    }
    
    // ==========================================
    // 3. 四元数插值查询（视频时间戳 → 姿态）
    // ==========================================
    static Eigen::Quaterniond interpolateQuat(const std::vector<QuatSample>& samples,
                                               double timestamp_ms) {
        if (samples.empty()) return Eigen::Quaterniond::Identity();
        if (samples.size() == 1) return samples[0].quat;
        
        // 边界处理
        if (timestamp_ms <= samples.front().timestamp_ms) return samples.front().quat;
        if (timestamp_ms >= samples.back().timestamp_ms) return samples.back().quat;
        
        // 二分查找
        auto it = std::lower_bound(samples.begin(), samples.end(), timestamp_ms,
            [](const QuatSample& s, double ts) { return s.timestamp_ms < ts; });
        
        if (it == samples.begin()) return it->quat;
        
        auto it_next = it;
        auto it_prev = std::prev(it);
        
        // SLERP 插值
        double t = (timestamp_ms - it_prev->timestamp_ms) / 
                   (it_next->timestamp_ms - it_prev->timestamp_ms);
        t = std::max(0.0, std::min(1.0, t));
        
        Eigen::Quaterniond q1 = it_prev->quat;
        Eigen::Quaterniond q2 = it_next->quat;
        
        // 确保最短路径
        if (q1.dot(q2) < 0) q2 = Eigen::Quaterniond(-q2.coeffs());
        
        return q1.slerp(t, q2);
    }
};

// ==========================================
// 4. 使用示例（伪代码）
// ==========================================
/*
int main() {
    // 1. 读取 GoPro 元数据（你需要自己解析 GPMD 或 JSON）
    std::vector<GyroSample> gyro_samples = loadGyroFromVideo("Hero6.mp4");
    std::vector<GyroSample> accel_samples = loadAccelFromVideo("Hero6.mp4");
    
    // 2. 初始化稳定器
    VideoStabilizer stabilizer;
    VideoStabilizer::SmoothParams params;
    params.base_alpha = 0.05;           // 强平滑（0.01-0.1 范围）
    params.roll_threshold = 10.0;       // Roll 轴 10 deg/s 以上开始减少平滑
    params.yaw_threshold = 30.0;        // Yaw 轴 30 deg/s 以上几乎不平滑（保留转向）
    params.use_velocity_dampened = true;
    
    // 3. 处理
    stabilizer.process(gyro_samples, accel_samples, params);
    
    // 4. 视频渲染循环（每帧调用）
    for (double video_ts = 0; video_ts < duration_ms; video_ts += 33.33) {  // 30fps
        // 获取矫正矩阵
        Eigen::Matrix3d R_corr = stabilizer.computeCorrectionMatrix(video_ts);
        
        // 应用到视频帧（OpenCV 或自定义渲染）
        // frame = warp(frame, R_corr, camera_matrix);
        
        // 调试：获取姿态对比
        Eigen::Quaterniond q_raw = stabilizer.getRawOrientation(video_ts);
        Eigen::Quaterniond q_smooth = stabilizer.getSmoothOrientation(video_ts);
        
        // 可以绘制波形图类似 GYROflow 底部界面
    }
    
    return 0;
}
*/