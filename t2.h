// ==========================================
// 真正的 时间窗口平滑（FIR 卷积版）
// ==========================================
class WindowedSmoother {
public:
    struct Config {
        double window_sec = 1.0;        // 窗口半宽：±1.0秒（总窗口2秒）
        double tau_slow = 1.0;          // 低速时衰减系数（高斯sigma或指数tau）
        double tau_fast = 0.1;          // 高速时衰减系数
        double speed_threshold = 30.0;  // deg/s
    };

    std::vector<QuatSample> process(const std::vector<QuatSample>& raw, 
                                    const Config& config) {
        int n = raw.size();
        std::vector<QuatSample> result(n);
        
        // 预计算每帧的速度和时间常数（用于自适应）
        std::vector<double> taus(n);
        for (int i = 0; i < n; ++i) {
            Eigen::Vector3d vel = computeVelocity(raw, i);  // deg/s
            double speed = vel.norm();
            taus[i] = adaptiveTau(speed, config);
        }
        
        // 对每一帧，收集窗口内所有数据做加权
        for (int i = 0; i < n; ++i) {
            double t_center = raw[i].timestamp_ms;
            double tau = taus[i];  // 当前帧根据速度选择的时间常数
            
            // 1. 收集窗口内所有帧（±window_sec）
            std::vector<WeightedSample> window;
            double total_weight = 0;
            
            for (int j = 0; j < n; ++j) {
                double dt_sec = std::abs(raw[j].timestamp_ms - t_center) / 1000.0;
                if (dt_sec > config.window_sec) continue;  // 超出1秒窗口，跳过
                
                // 计算权重：指数衰减或高斯
                // 距离中心越远，权重越小
                double w = std::exp(-dt_sec / tau);  // 指数核
                // 或 double w = std::exp(-dt_sec*dt_sec / (2*tau*tau));  // 高斯核
                
                if (w > 0.01) {  // 忽略极小权重（优化）
                    window.push_back({j, w});
                    total_weight += w;
                }
            }
            
            // 2. 四元数加权平均（在李代数 so3 空间进行，避免万向锁）
            // 方法：以当前帧为参考，将所有帧表示为相对旋转，平均后再映射回去
            result[i].quat = weightedAverageSO3(raw, i, window, total_weight);
            result[i].timestamp_ms = t_center;
        }
        
        return result;
    }

private:
    struct WeightedSample {
        int idx;
        double weight;
    };

    // 计算第 i 帧的瞬时角速度（deg/s）
    Eigen::Vector3d computeVelocity(const std::vector<QuatSample>& data, int i) {
        if (i == 0 || i >= data.size()-1) return Eigen::Vector3d::Zero();
        
        // 中央差分
        Eigen::Quaterniond q_prev = data[i-1].quat;
        Eigen::Quaterniond q_next = data[i+1].quat;
        double dt = (data[i+1].timestamp_ms - data[i-1].timestamp_ms) / 1000.0;
        
        Eigen::Quaterniond q_rel = q_next * q_prev.inverse();
        if (q_rel.w() < 0) q_rel.coeffs() *= -1;
        
        // 转轴角
        double angle = 2.0 * std::atan2(q_rel.vec().norm(), std::abs(q_rel.w()));
        if (std::abs(angle) < 1e-6) return Eigen::Vector3d::Zero();
        
        Eigen::Vector3d axis = q_rel.vec().normalized();
        return axis * (angle / dt * 180.0 / M_PI);  // deg/s
    }

    double adaptiveTau(double speed, const Config& cfg) {
        // 速度阻尼：速度越大，tau越小（权重集中在中心，窗口等效变窄）
        double t = (speed - cfg.speed_threshold) / 20.0;
        t = std::max(0.0, std::min(1.0, t));
        t = t * t * (3.0 - 2.0 * t);
        return cfg.tau_slow * (1.0 - t) + cfg.tau_fast * t;
    }

    // ==========================================
    // 核心：在 so3 空间做加权平均（避免直接在四元数4D空间平均导致的归一化问题）
    // ==========================================
    Eigen::Quaterniond weightedAverageSO3(const std::vector<QuatSample>& raw,
                                         int center_idx,
                                         const std::vector<WeightedSample>& window,
                                         double total_weight) {
        const Eigen::Quaterniond& q_ref = raw[center_idx].quat;
        Eigen::Vector3d avg_so3 = Eigen::Vector3d::Zero();
        
        // 1. 将所有样本转换到参考帧的局部坐标系（相对旋转），取对数（so3）
        for (const auto& ws : window) {
            const auto& q_j = raw[ws.idx].quat;
            
            // 相对旋转：q_j * q_ref^{-1}（表示从参考帧到当前帧的旋转）
            Eigen::Quaterniond q_rel = q_j * q_ref.inverse();
            if (q_rel.w() < 0) q_rel.coeffs() *= -1;  // 确保最短路径
            
            // 对数映射：四元数 -> 轴角向量（so3 李代数）
            Eigen::Vector3d so3 = quaternionToSO3(q_rel);
            
            // 加权累加（在向量空间，这是合法的）
            double w = ws.weight / total_weight;
            avg_so3 += w * so3;
        }
        
        // 2. 指数映射回四元数：so3 -> 四元数
        Eigen::Quaterniond q_rel_avg = so3ToQuaternion(avg_so3);
        
        // 3. 转换回全局坐标系：相对旋转 * 参考帧
        return (q_rel_avg * q_ref).normalized();
    }

    // 四元数对数映射（Q -> so3）
    Eigen::Vector3d quaternionToSO3(const Eigen::Quaterniond& q) {
        double w = q.w();
        Eigen::Vector3d v = q.vec();
        double v_norm = v.norm();
        
        if (v_norm < 1e-6) return Eigen::Vector3d::Zero();
        
        // angle = 2 * atan(|v|/w)
        double angle = 2.0 * std::atan2(v_norm, w);
        // so3 = axis * angle = v/|v| * angle = v * (angle/|v|)
        return v * (angle / v_norm);
    }

    // so3 指数映射（so3 -> Q）
    Eigen::Quaterniond so3ToQuaternion(const Eigen::Vector3d& so3) {
        double angle = so3.norm();
        if (angle < 1e-6) return Eigen::Quaterniond::Identity();
        
        Eigen::Vector3d axis = so3 / angle;
        double half = angle * 0.5;
        
        return Eigen::Quaterniond(
            std::cos(half),
            axis.x() * std::sin(half),
            axis.y() * std::sin(half),
            axis.z() * std::sin(half)
        );
    }
};