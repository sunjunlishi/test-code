/**
* Gyroflow-Level Video Stabilization - 步行场景专用修复版
* 关键修复：
* 1. 正确的补偿方向：R_smooth * R_current^T（将当前帧旋转到平滑位置）
* 2. 移除地平线锁定（步行需要自然转向）
* 3. 改进的 FOV 缩放计算
* 4. 正确的 GoPro 坐标系转换
* 5. 延迟平滑（使用未来数据）
*/

#include <iostream>
#include <vector>
#include <deque>
#include <cmath>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#define M_PI 3.14159265358979323846

extern "C" {
#include "GPMF_parser.h"
#include "GPMF_utils.h"
#include "GPMF_mp4reader.h"
}

namespace Gyroflow {

	template<typename T>
	inline T clamp_val(T value, T min_val, T max_val) {
		return std::max(min_val, std::min(value, max_val));
	}

	struct Vec3 {
		double x, y, z;
		Vec3(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}
		Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
		Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
		Vec3 operator*(double s) const { return Vec3(x*s, y*s, z*s); }
		Vec3 operator-() const { return Vec3(-x, -y, -z); }
		double length() const { return std::sqrt(x*x + y*y + z*z); }
		Vec3 normalized() const {
			double len = length();
			if (len < 1e-10) return Vec3(0, 0, 0);
			return Vec3(x / len, y / len, z / len);
		}
	};

	struct Quaternion {
		double w, x, y, z;
		Quaternion(double w = 1, double x = 0, double y = 0, double z = 0) : w(w), x(x), y(y), z(z) {}

		static Quaternion identity() { return Quaternion(1, 0, 0, 0); }

		static Quaternion fromAxisAngle(const Vec3& axis, double angle) {
			if (std::abs(angle) < 1e-10) return identity();
			double half = angle * 0.5;
			double s = std::sin(half);
			double c = std::cos(half);
			Vec3 a = axis.normalized();
			return Quaternion(c, a.x*s, a.y*s, a.z*s);
		}

		static Quaternion fromEulerAngles(double roll, double pitch, double yaw) {
			double cr = std::cos(roll*0.5), sr = std::sin(roll*0.5);
			double cp = std::cos(pitch*0.5), sp = std::sin(pitch*0.5);
			double cy = std::cos(yaw*0.5), sy = std::sin(yaw*0.5);
			return Quaternion(
				cr*cp*cy + sr*sp*sy,
				sr*cp*cy - cr*sp*sy,
				cr*sp*cy + sr*cp*sy,
				cr*cp*sy - sr*sp*cy
				);
		}

		void normalize() {
			double len = std::sqrt(w*w + x*x + y*y + z*z);
			if (len > 1e-10) { w /= len; x /= len; y /= len; z /= len; }
		}

		Quaternion conjugate() const { return Quaternion(w, -x, -y, -z); }

		Quaternion inverse() const {
			double norm = w*w + x*x + y*y + z*z;
			if (norm < 1e-10) return identity();
			return Quaternion(w / norm, -x / norm, -y / norm, -z / norm);
		}

		Quaternion operator*(const Quaternion& q) const {
			return Quaternion(
				w*q.w - x*q.x - y*q.y - z*q.z,
				w*q.x + x*q.w + y*q.z - z*q.y,
				w*q.y - x*q.z + y*q.w + z*q.x,
				w*q.z + x*q.y - y*q.x + z*q.w
				);
		}

		Vec3 rotate(const Vec3& v) const {
			Quaternion qv(0, v.x, v.y, v.z);
			Quaternion t = (*this) * qv * conjugate();
			return Vec3(t.x, t.y, t.z);
		}

		Vec3 toEulerAngles() const {
			double roll = std::atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y));
			double pitch = std::asin(clamp_val(2 * (w*y - z*x), -1.0, 1.0));
			double yaw = std::atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z));
			return Vec3(roll, pitch, yaw);
		}

		cv::Mat toRotationMatrix() const {
			double xx = x*x, yy = y*y, zz = z*z;
			double xy = x*y, xz = x*z, yz = y*z;
			double wx = w*x, wy = w*y, wz = w*z;
			cv::Mat R = (cv::Mat_<double>(3, 3) <<
				1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
				2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
				2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy));
			return R;
		}

		static double dot(const Quaternion& a, const Quaternion& b) {
			return a.w*b.w + a.x*b.x + a.y*b.y + a.z*b.z;
		}

		// 确保四元数连续性（选择最短路径）
		static Quaternion makeContinuous(const Quaternion& q, const Quaternion& ref) {
			if (dot(q, ref) < 0) {
				return Quaternion(-q.w, -q.x, -q.y, -q.z);
			}
			return q;
		}


		// ========== 新增：SLERP 球面线性插值 ==========
		static Quaternion slerp(const Quaternion& a, const Quaternion& b, double t) {
			if (t <= 0.0) return a;
			if (t >= 1.0) return b;

			Quaternion q1 = a, q2 = b;

			// 确保最短路径
			if (dot(q1, q2) < 0) {
				q2 = Quaternion(-q2.w, -q2.x, -q2.y, -q2.z);
			}

			double cosTheta = dot(q1, q2);
			cosTheta = clamp_val(cosTheta, -1.0, 1.0);

			// 角度很小则用线性插值避免数值问题
			if (cosTheta > 0.9995) {
				Quaternion result = q1 + (q2 - q1) * t;
				result.normalize();
				return result;
			}

			double theta = std::acos(cosTheta);
			double sinTheta = std::sin(theta);
			if (std::abs(sinTheta) < 1e-10) return q1;

			double ratioA = std::sin((1 - t) * theta) / sinTheta;
			double ratioB = std::sin(t * theta) / sinTheta;

			Quaternion result(
				q1.w * ratioA + q2.w * ratioB,
				q1.x * ratioA + q2.x * ratioB,
				q1.y * ratioA + q2.y * ratioB,
				q1.z * ratioA + q2.z * ratioB
				);
			result.normalize();
			return result;
		}

		// ========== 新增：计算旋转角度（弧度）==========
		double angle() const {
			double cosHalf = clamp_val(w, -1.0, 1.0);
			return 2.0 * std::acos(std::abs(cosHalf));
		}

		// 向量减法运算符（用于速度计算）
		Quaternion operator-(const Quaternion& q) const {
			return Quaternion(w - q.w, x - q.x, y - q.y, z - q.z);
		}

		// 向量加法运算符
		Quaternion operator+(const Quaternion& q) const {
			return Quaternion(w + q.w, x + q.x, y + q.y, z + q.z);
		}
	};

	// ==================== 镜头配置 ====================
	struct LensProfile {
		cv::Mat camera_matrix;
		cv::Mat optimal_matrix;
		cv::Mat dist_coeffs;
		cv::Mat map1, map2;
		double fov_scale = 1.0;
		int width = 0, height = 0;

		void init(int w, int h, const cv::Mat& K, const cv::Mat& D, bool fisheye = true) {
			width = w; height = h;
			camera_matrix = K.clone();
			dist_coeffs = D.clone();

			cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
			cv::Rect roi;
			cv::Mat new_cam = cv::getOptimalNewCameraMatrix(camera_matrix, dist_coeffs,
				cv::Size(w, h), 0.0, cv::Size(w, h), &roi);
			optimal_matrix = new_cam.clone();

			fov_scale = std::min((double)roi.width / w, (double)roi.height / h);

			if (fisheye) {
				cv::fisheye::initUndistortRectifyMap(camera_matrix, dist_coeffs, R, optimal_matrix,
					cv::Size(w, h), CV_32FC1, map1, map2);
			}
			else {
				cv::initUndistortRectifyMap(camera_matrix, dist_coeffs, R, optimal_matrix,
					cv::Size(w, h), CV_32FC1, map1, map2);
			}
		}

		void loadHero6Wide(int w, int h) {
			// 使用你提供的精确参数
			double fx = 1186.540692279175;
			double fy = 1186.470033858650;
			double cx = 1355.38895089212;
			double cy = 1020.317054525362;

			camera_matrix = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

			// 畸变系数 k1, k2, k3, k4
			dist_coeffs = (cv::Mat_<double>(4, 1) <<
				0.0444046577769409,
				0.0194678995117994,
				-0.0044766975393439,
				-0.0020429128777408);

			init(w, h, camera_matrix, dist_coeffs, true);
		}

		void loadHero6Linear(int w, int h) {
			double fx = w * 0.9, fy = h * 0.9;
			double cx = w / 2.0, cy = h / 2.0;
			camera_matrix = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
			dist_coeffs = (cv::Mat_<double>(5, 1) << -0.05, 0.01, 0, 0, 0);

			init(w, h, camera_matrix, dist_coeffs, false);
		}

		cv::Mat undistort(const cv::Mat& frame) const {
			cv::Mat result;
			cv::remap(frame, result, map1, map2, cv::INTER_LINEAR);
			return result;
		}
	};

	// ==================== GPMF 提取器 ====================
	class GyroExtractor {
	public:
		struct GyroSample {
			Vec3 data;              // rad/s
			double timestamp_ms;    // 毫秒
		};

		std::vector<GyroSample> samples;
		double sample_rate = 400.0;
		double time_offset_ms = 0;

		bool extract(const std::string& path) {
			size_t handle = OpenMP4Source(const_cast<char*>(path.c_str()),
				MOV_GPMF_TRAK_TYPE, MOV_GPMF_TRAK_SUBTYPE, 0);
			if (!handle) return false;

			uint32_t payloads = GetNumberPayloads(handle);
			std::cout << "GPMF payloads: " << payloads << std::endl;

			double first_ts = -1;

			for (uint32_t i = 0; i < payloads; i++) {
				uint32_t size = GetPayloadSize(handle, i);
				if (!size) continue;

				size_t res = 0;
				res = GetPayloadResource(handle, res, size);
				uint32_t* payload = GetPayload(handle, res, i);
				if (!payload) { FreePayloadResource(handle, res); continue; }

				double in = 0, out = 0;
				GetPayloadTime(handle, i, &in, &out);

				// === 关键修复 ===
				// 1. GetPayloadTime 返回的是 绝对时间 (秒)
				// 2. 不需要累加 offset
				// 3. 必须转换为 毫秒
				double abs_in_ms = in * 1000.0;
				double abs_out_ms = out * 1000.0;

				// 调试：打印前 3 个和最后 1 个
				if (i < 3 || i == payloads - 1) {
					std::cout << "[PAYLOAD " << i << "] Time: " << in << "s ~ " << out << "s ("
						<< abs_in_ms << " ~ " << abs_out_ms << " ms)" << std::endl;
				}

				GPMF_stream ms;
				if (GPMF_OK == GPMF_Init(&ms, payload, size)) {
					if (GPMF_OK == GPMF_FindNext(&ms, STR2FOURCC("GYRO"), GPMF_RECURSE_LEVELS)) {
						extractSamples(ms, abs_in_ms, abs_out_ms, first_ts);
					}
				}
				GPMF_Free(&ms);
				FreePayloadResource(handle, res);

				// === 删除之前的累加逻辑 ===
			}
			CloseSource(handle);

			// 统计
			if (!samples.empty()) {
				double total_duration_sec = (samples.back().timestamp_ms - samples.front().timestamp_ms) / 1000.0;
				std::cout << "[GYRO-STAT] Total samples: " << samples.size() << std::endl;
				std::cout << "[GYRO-STAT] Duration: " << total_duration_sec << " seconds" << std::endl;
				std::cout << "[GYRO-STAT] Avg sample rate: " << samples.size() / total_duration_sec << " Hz" << std::endl;
			}

			return !samples.empty();
		}

		// 在 GyroExtractor 类中 public 部分添加：
		// 获取时间区间 [t_start, t_end] 内的所有陀螺仪样本（Gyroflow 风格）
		std::vector<GyroSample> getRange(double t_start_ms, double t_end_ms) const {

			
			std::vector<GyroSample> result;
			if (samples.empty() || t_start_ms >= t_end_ms) return result;

			// 找到起始位置（lower_bound 返回第一个 >= t_start_ms 的元素）
			auto it_start = std::lower_bound(samples.begin(), samples.end(), t_start_ms,
				[](const GyroSample& s, double val) { return s.timestamp_ms < val; });

			// 找到结束位置（upper_bound 返回第一个 > t_end_ms 的元素）
			auto it_end = std::upper_bound(samples.begin(), samples.end(), t_end_ms,
				[](double val, const GyroSample& s) { return val < s.timestamp_ms; });

			// 如果起始位置不是第一个，且前一个点更接近 t_start_ms，则插入插值起点
			if (it_start != samples.begin()) {
				auto it_prev = std::prev(it_start);
				if (it_prev->timestamp_ms < t_start_ms) {
					// 线性插值起点
					double t0 = it_prev->timestamp_ms;
					double t1 = it_start->timestamp_ms;
					double alpha = (t_start_ms - t0) / (t1 - t0);
					alpha = clamp_val(alpha, 0.0, 1.0);

					Vec3 g_interp(
						it_prev->data.x + alpha * (it_start->data.x - it_prev->data.x),
						it_prev->data.y + alpha * (it_start->data.y - it_prev->data.y),
						it_prev->data.z + alpha * (it_start->data.z - it_prev->data.z)
						);
					result.push_back({ g_interp, t_start_ms });
				}
			}

			// 添加区间内的所有样本
			for (auto it = it_start; it != it_end; ++it) {
				result.push_back(*it);
			}

			// 如果结束位置不是最后一个，且需要插值终点
			if (it_end != samples.end() && !result.empty() && result.back().timestamp_ms < t_end_ms) {
				auto it_last = std::prev(it_end);
				if (it_last->timestamp_ms < t_end_ms) {
					// 需要下一个点来插值
					auto it_next = it_end;
					if (it_next != samples.end()) {
						double t0 = it_last->timestamp_ms;
						double t1 = it_next->timestamp_ms;
						double alpha = (t_end_ms - t0) / (t1 - t0);
						alpha = clamp_val(alpha, 0.0, 1.0);

						Vec3 g_interp(
							it_last->data.x + alpha * (it_next->data.x - it_last->data.x),
							it_last->data.y + alpha * (it_next->data.y - it_last->data.y),
							it_last->data.z + alpha * (it_next->data.z - it_last->data.z)
							);
						result.push_back({ g_interp, t_end_ms });
					}
					else {
						// 没有下一个点，使用最后一个已知点
						result.push_back({ it_last->data, t_end_ms });
					}
				}
			}

			return result;
		}

		Vec3 getAt(double ts_ms) const {
			if (samples.empty()) return Vec3();
			if (ts_ms <= samples.front().timestamp_ms) return samples.front().data;
			if (ts_ms >= samples.back().timestamp_ms) return samples.back().data;

			// 二分查找
			size_t left = 0, right = samples.size() - 1;
			while (left < right - 1) {
				size_t mid = (left + right) / 2;
				if (samples[mid].timestamp_ms < ts_ms) left = mid;
				else right = mid;
			}

			double t1 = samples[left].timestamp_ms;
			double t2 = samples[right].timestamp_ms;
			double alpha = (ts_ms - t1) / (t2 - t1);
			if (alpha < 0) alpha = 0;
			if (alpha > 1) alpha = 1;

			const Vec3& g1 = samples[left].data;
			const Vec3& g2 = samples[right].data;
			return Vec3(
				g1.x + alpha * (g2.x - g1.x),
				g1.y + alpha * (g2.y - g1.y),
				g1.z + alpha * (g2.z - g1.z)
				);
		}

	private:
		void extractSamples(GPMF_stream& ms, double in_ms, double out_ms, double& first_ts) {
			uint32_t sample_count = GPMF_Repeat(&ms);
			uint32_t elements = GPMF_ElementsInStruct(&ms);
			if (sample_count == 0 || elements != 3) return;

			double scale = 1.0;
			GPMF_stream find;
			GPMF_CopyState(&ms, &find);
			if (GPMF_OK == GPMF_FindPrev(&find, GPMF_KEY_SCALE, GPMF_CURRENT_LEVEL)) {
				float s = 1.0f;
				GPMF_FormattedData(&find, &s, sizeof(float), 0, 1);
				scale = s;
			}

			std::cout << "GYRO Scale: " << scale << " deg/s per count" << std::endl;
	

			std::vector<int16_t> buffer(sample_count * elements);
			if (GPMF_OK != GPMF_FormattedData(&ms, buffer.data(),
				buffer.size()*sizeof(int16_t), 0, sample_count)) return;

			double step_ms = (out_ms - in_ms) / sample_count;

			double sample_rate = sample_count * 1000.0 / (out_ms - in_ms);
			std::cout << "Sample rate: " << sample_rate << " Hz" << std::endl;

			for (uint32_t i = 0; i < sample_count; i++) {
				// GoPro Hero 6 GYRO 格式: Z, X, Y (单位：deg/s)
				double raw_z = buffer[i * 3 + 0];
				double raw_x = buffer[i * 3 + 1];
				double raw_y = buffer[i * 3 + 2];

				// ========== 关键修正：始终转换为 rad/s ==========
				// GoPro 的 scale 单位是 deg/s，必须转换为 rad/s
				double gx = raw_x * scale * (M_PI / 180.0);
				double gy = raw_y * scale * (M_PI / 180.0);
				double gz = raw_z * scale * (M_PI / 180.0);

				// ========== 坐标系转换（参考 Gyroflow 官方）==========
				// GoPro 坐标系 -> 标准相机坐标系
				// X' = -Y, Y' = X, Z' = Z
				double gx_std = -gy;
				double gy_std = gx;
				double gz_std = gz;

				// 每 100 个样本打印一次，避免刷屏
				if (i%30 == 0) {
					std::cout << "[GYRO] Sample#" << i
						<< " | Raw(Z,X,Y): (" << raw_z << ", " << raw_x << ", " << raw_y << ") deg/s"
						<< " | Converted(X,Y,Z): (" << gx_std << ", " << gy_std << ", " << gz_std << ") rad/s"
						<< " | Magnitude: " << std::sqrt(gx_std*gx_std + gy_std*gy_std + gz_std*gz_std) * 180 / M_PI << " deg/s"
						<< std::endl;
				}

				Vec3 gyro(gx_std, gy_std, gz_std);

				double ts = in_ms + i * step_ms;
				if (first_ts < 0) first_ts = ts;
				//std::cout << "samples timeinfo .................." << samples.size() << " time " << ts<<std::endl;
				samples.push_back({ gyro, ts });
			}
		}

	};
	// ==================== Gyroflow Default 平滑器（跑步优化版）====================
	class GyroflowSmoother {
	public:
		struct OrientationSample {
			Quaternion orientation;
			double timestamp_ms;
			Vec3 gyro;  // rad/s，用于速度计算
		};

		std::deque<OrientationSample> history;

		// 核心参数（参考 Gyroflow Default 算法）
		double smoothness = 0.9;              // [0.001, 1.0] 平滑度
		double max_smoothness_sec = 2.0;      // 低速时最大平滑时间（秒）
		double max_smoothness_high_vel_sec = 0.1;  // 高速时最大平滑时间（秒）
		bool second_pass = true;              // 启用二阶平滑
		double gyro_sample_rate = 200.0;      // 陀螺仪采样率

											  // 内部状态
		double lookahead_ms = 0.0;
		double time_constant_ms = 1000.0;

		void setParams(double smooth, double sample_rate, double delay_frames, double fps) {
			smoothness = clamp_val(smooth, 0.001, 1.0);
			gyro_sample_rate = sample_rate;
			lookahead_ms = (delay_frames / fps) * 1000.0;

			// Gyroflow 公式：时间窗口 = 0.5 + smoothness * 2.5 秒
			time_constant_ms = (0.5 + smoothness * 2.5) * 1000.0;

			std::cout << "Smoother params: smooth=" << smoothness
				<< ", time_const=" << time_constant_ms << "ms"
				<< ", lookahead=" << lookahead_ms << "ms" << std::endl;
		}

		void add(const Quaternion& q, double ts_ms, const Vec3& gyro) {
			// 确保四元数连续性
			Quaternion q_fixed = q;
			if (!history.empty()) {
				q_fixed = Quaternion::makeContinuous(q, history.back().orientation);
			}

			history.push_back({ q_fixed, ts_ms, gyro });

			// 清理旧数据（保留 5 倍时间窗口）
			double cutoff = ts_ms - time_constant_ms * 5;
			while (!history.empty() && history.front().timestamp_ms < cutoff) {
				history.pop_front();
			}
		}

		// 获取平滑后的姿态（核心算法 - 修复版）
		Quaternion getSmoothed(double current_ts) {
			if (history.empty()) return Quaternion::identity();
			if (history.size() < 2) return history.front().orientation;

			// 转换为 vector 方便处理
			std::vector<OrientationSample> samples(history.begin(), history.end());

			// ========== 修复：先提取 Quaternion 向量 ==========
			std::vector<Quaternion> quats;
			quats.reserve(samples.size());
			for (const auto& s : samples) {
				quats.push_back(s.orientation); 
			}

			// 1. 计算角速度比率（用于速度阻尼）
			std::vector<double> vel_ratios = computeVelocityRatios(samples);

			// 2. 第一轮 SLERP 平滑（正向）- 传入 Quaternion 向量
			std::vector<Quaternion> pass1 = slerpPass(quats, vel_ratios, true);

			// 3. 第二轮 SLERP 平滑（反向，抵消延迟）
			std::vector<Quaternion> pass2 = slerpPass(pass1, vel_ratios, false);

			// 4. 应用 lookahead 延迟补偿
			return applyLookahead(pass2, samples, current_ts);
		}

	private:
		// 计算归一化的速度比率 [0, 1]
		std::vector<double> computeVelocityRatios(const std::vector<OrientationSample>& samples) {
			std::vector<double> ratios(samples.size(), 0.0);
			if (samples.size() < 2) return ratios;

			// 计算相邻帧的旋转角度（度/秒）
			std::vector<double> angular_vels;
			angular_vels.reserve(samples.size());
			angular_vels.push_back(0.0);

			for (size_t i = 1; i < samples.size(); i++) {
				// 计算两个四元数之间的旋转角度
				Quaternion diff = samples[i - 1].orientation.inverse() * samples[i].orientation;
				double angle_rad = diff.angle();  // 使用 Quaternion::angle()
				double angle_deg = angle_rad * 180.0 / M_PI;
				double dt_sec = (samples[i].timestamp_ms - samples[i - 1].timestamp_ms) / 1000.0;
				double vel = (dt_sec > 1e-6) ? angle_deg / dt_sec : 0.0;
				angular_vels.push_back(vel);
			}

			// 指数平滑速度（100ms 时间常数）
			double alpha_vel = computeAlpha(0.1, gyro_sample_rate);
			for (size_t i = 1; i < angular_vels.size(); i++) {
				angular_vels[i] = angular_vels[i - 1] * (1.0 - alpha_vel) + angular_vels[i] * alpha_vel;
			}

			// 归一化：MAX_VELOCITY = 1000 * smoothness (deg/s)
			double max_vel = 1000.0 * smoothness;
			for (size_t i = 0; i < samples.size(); i++) {
				ratios[i] = clamp_val(angular_vels[i] / max_vel, 0.0, 1.0);
			}

			return ratios;
		}

		// 计算 SLERP 的 alpha 值（速度阻尼）
		double computeAlphaForVel(double vel_ratio) {
			double alpha_slow = computeAlpha(max_smoothness_sec, gyro_sample_rate);
			double alpha_fast = computeAlpha(max_smoothness_high_vel_sec, gyro_sample_rate);
			// 速度越快，alpha 越小（平滑度越低）
			return alpha_slow * (1.0 - vel_ratio) + alpha_fast * vel_ratio;
		}

		double computeAlpha(double time_const_sec, double sample_rate) {
			double dt = 1.0 / sample_rate;
			// 时间常数越大，alpha 越小（平滑度越高）
			return 1.0 - std::exp(-dt / time_const_sec);
		}

		// SLERP 平滑通道（正向或反向）
		std::vector<Quaternion> slerpPass(const std::vector<Quaternion>& quats,
			const std::vector<double>& vel_ratios,
			bool forward) {
			std::vector<Quaternion> result(quats.size());

			if (forward) {
				result[0] = quats[0];
				for (size_t i = 1; i < quats.size(); i++) {
					double alpha = computeAlphaForVel(vel_ratios[i]);
					result[i] = Quaternion::slerp(result[i - 1], quats[i], std::min(alpha, 1.0));
				}
			}
			else {
				result.back() = quats.back();
				for (int i = (int)quats.size() - 2; i >= 0; i--) {
					double alpha = computeAlphaForVel(vel_ratios[i]);
					result[i] = Quaternion::slerp(result[i + 1], quats[i], std::min(alpha, 1.0));
				}
			}
			return result;
		}

		// 应用 lookahead 延迟补偿（从平滑结果中取未来帧）
		Quaternion applyLookahead(const std::vector<Quaternion>& smoothed,
			const std::vector<OrientationSample>& samples,
			double current_ts) {
			double target_ts = current_ts + lookahead_ms;

			// 二分查找最接近 target_ts 的帧
			size_t left = 0, right = samples.size() - 1;
			while (left < right - 1) {
				size_t mid = (left + right) / 2;
				if (samples[mid].timestamp_ms < target_ts) left = mid;
				else right = mid;
			}

			// 线性插值
			double t1 = samples[left].timestamp_ms;
			double t2 = samples[right].timestamp_ms;
			if (t2 <= t1) return smoothed[left];

			double alpha = (target_ts - t1) / (t2 - t1);
			alpha = clamp_val(alpha, 0.0, 1.0);

			return Quaternion::slerp(smoothed[left], smoothed[right], alpha);
		}
	};
	// ==================== 延迟平滑器（Gyroflow 核心算法）====================
	class DelayedSmoother {
	public:
		struct OrientationSample {
			Quaternion orientation;
			double timestamp_ms;
		};

		std::deque<OrientationSample> history;
		double smoothness = 0.8;
		double time_constant_ms = 3000.0;
		double lookahead_ms = 200.0;

		void setSmoothness(double s, double gyro_sample_rate) {
			smoothness = clamp_val(s, 0.0, 1.0);
			// Gyroflow: 时间窗口 = (0.5 + smoothness * 2.5) 秒
			time_constant_ms = (0.5 + smoothness * 2.5) * 1000.0;
			std::cout << "Smoothing time constant: " << time_constant_ms << " ms" << std::endl;
		}

		void setDelay(double delay_frames, double fps) {
			lookahead_ms = (delay_frames / fps) * 1000.0;
			std::cout << "Lookahead delay: " << lookahead_ms << " ms" << std::endl;
		}

		void add(const Quaternion& q, double ts_ms) {
			// 确保四元数连续性
			Quaternion q_fixed = q;
			if (!history.empty()) {
				q_fixed = Quaternion::makeContinuous(q, history.back().orientation);
			}

			history.push_back({ q_fixed, ts_ms });

			// 清理旧数据
			double cutoff = ts_ms - time_constant_ms * 5;
			while (!history.empty() && history.front().timestamp_ms < cutoff) {
				history.pop_front();
			}
		}

		// 获取延迟补偿后的平滑姿态（关键：使用未来数据）
		Quaternion getSmoothed(double current_ts) const {
			double target_ts = current_ts + lookahead_ms;

			if (history.empty()) return Quaternion::identity();
			if (history.size() < 2) return history.front().orientation;

			// 高斯加权平滑
			double sum_w = 0;
			double sum_qw = 0, sum_qx = 0, sum_qy = 0, sum_qz = 0;
			double sigma = time_constant_ms / 3.0;

			for (const auto& s : history) {
				double dt = (s.timestamp_ms - target_ts) / sigma;
				double w = std::exp(-0.5 * dt * dt);

				sum_qw += s.orientation.w * w;
				sum_qx += s.orientation.x * w;
				sum_qy += s.orientation.y * w;
				sum_qz += s.orientation.z * w;
				sum_w += w;
			}

			if (sum_w < 1e-10) return history.back().orientation;

			Quaternion result(sum_qw / sum_w, sum_qx / sum_w, sum_qy / sum_w, sum_qz / sum_w);
			result.normalize();
			return result;
		}
	};


	class Stabilizer {
	public:
		GyroExtractor gyro;
		LensProfile lens;
		GyroflowSmoother smoother;  // 替换 DelayedSmoother

		Quaternion current_orientation;
		double last_frame_ts = -1;
		double fps = 30.0;
		int frame_count = 0;

		double user_fov_scale = 1.0;
		double delay_frames = 0.0;
		std::deque<std::pair<double, double>> zoom_history;

		bool init(const std::string& mp4_path, const std::string& lens_mode) {
			if (!gyro.extract(mp4_path)) {
				std::cerr << "Failed to extract gyro data!" << std::endl;
				return false;
			}

			cv::VideoCapture cap(mp4_path);
			if (!cap.isOpened()) return false;

			int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
			int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
			fps = cap.get(cv::CAP_PROP_FPS);
			if (fps < 1) fps = 30.0;  // 默认值
			cap.release();

			if (lens_mode == "linear") lens.loadHero6Linear(w, h);
			else lens.loadHero6Wide(w, h);

			std::cout << "Video:................... " << w << "x" << h << " @ " << fps << "fps" << std::endl;
			std::cout << "Lens FOV scale: " << lens.fov_scale << lens_mode<< std::endl;
			return true;
		}

		void setParams(double smoothness, double fov_scale, double delay = 0.0) {
			smoother.setParams(smoothness, gyro.sample_rate, delay, fps);
			user_fov_scale = fov_scale;
			delay_frames = delay;
		}

		void reset() {
			current_orientation = Quaternion::identity();
			last_frame_ts = -1;
			frame_count = 0;
			smoother.history.clear();
			zoom_history.clear();
		}

		
		cv::Mat process(const cv::Mat& frame, double timestamp_ms) {
			// 1. 畸变校正
			cv::Mat undistorted = lens.undistort(frame);

			// 2. 积分得到当前帧姿态
			integrateGyro(timestamp_ms);

			// 3. 添加到平滑器
			Vec3 gyro_now = gyro.getAt(timestamp_ms);
			smoother.add(current_orientation, timestamp_ms, gyro_now);

			// 4. 获取平滑后的姿态
			Quaternion smoothed = smoother.getSmoothed(timestamp_ms);

			// 5. 计算补偿旋转: R_comp = R_smoothed * R_current^T  ? 修正顺序
			Quaternion current_inv = current_orientation.conjugate();
			Quaternion compensation = smoothed * current_inv;  // ? 修正：smoothed 在前
			compensation.normalize();

			// 调试输出...
			Vec3 comp_euler = compensation.toEulerAngles();
			Vec3 curr_euler = current_orientation.toEulerAngles();
			Vec3 smooth_euler = smoothed.toEulerAngles();

			std::cout << "[DEBUG] Frame#" << frame_count << std::endl;
			std::cout << "frame_count ...." << frame_count << std::endl;;
			// 调试：打印关键姿态的欧拉角（度）
			//if (frame_count % 10 == 0) 
			// 6. 安全检查：补偿角度太大则跳过

			std::cout << "[DEBUG] Frame#" << frame_count << std::endl;
			std::cout << "  Current : R=" << curr_euler.x * 180 / M_PI << " P=" << curr_euler.y * 180 / M_PI << " Y=" << curr_euler.z * 180 / M_PI << std::endl;
			std::cout << "  Smoothed: R=" << smooth_euler.x * 180 / M_PI << " P=" << smooth_euler.y * 180 / M_PI << " Y=" << smooth_euler.z * 180 / M_PI << std::endl;
			std::cout << "  Comp    : R=" << comp_euler.x * 180 / M_PI << " P=" << comp_euler.y * 180 / M_PI << " Y=" << comp_euler.z * 180 / M_PI << std::endl;

			// 6. 安全检查
			double max_comp_angle = std::max({ std::abs(comp_euler.x),
				std::abs(comp_euler.y),
				std::abs(comp_euler.z) });

			// 修正：frame_count 必须先递增，或移到前面
			frame_count++;  // 移到前面，确保计数正确

			if (max_comp_angle > 0.52) {  // 30度
					//last_frame_ts = timestamp_ms;  // 别忘了更新时间戳
				   //return undistorted.clone();
			}

			// 7-9. 变换应用...
			cv::Mat R = compensation.toRotationMatrix();
			cv::Mat K = lens.camera_matrix;
			cv::Mat transform = K * R * K.inv();

			cv::Mat stabilized;
			cv::warpPerspective(undistorted, stabilized, transform, frame.size(),
				cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

			stabilized = applyAdaptiveZoom(stabilized, compensation);

			last_frame_ts = timestamp_ms;
			return stabilized;
		}

	private:
		// 零偏估计（静止时校准）
		Vec3 gyro_bias_ = Vec3(0, 0, 0);
		bool bias_estimated_ = false;

		// 参数配置
		static constexpr double MAX_GYRO_NORM = 8.0;        // rad/s (约 458 deg/s)
		static constexpr double MAX_SINGLE_ROTATION = 0.5;  // rad (约 28度/帧)
		static constexpr double MAX_TIME_GAP_SEC = 0.5;     // 500ms，超过则重置

		void integrateGyro(double target_ts_ms) {
			// 第一帧初始化
			if (last_frame_ts < 0) {
				last_frame_ts = target_ts_ms;

				// 首次运行时估计零偏（使用前 500ms 数据）
				if (!bias_estimated_) {
					estimateGyroBias();
				}
				return;
			}

			double dt_total = (target_ts_ms - last_frame_ts)/ 1000.0;  // 转为秒													// 安全检查：时间间隔异常
			if (dt_total <= 0) {
				std::cout << "[WARN] Non-positive dt: " << dt_total << "s, skipping frame" << std::endl;
				last_frame_ts = target_ts_ms;
				return;
			}
			if (dt_total > MAX_TIME_GAP_SEC) {
				std::cout << "[WARN] Large time gap: " << dt_total * 1000 << " ms, reset integration" << std::endl;
				last_frame_ts = target_ts_ms;
				return;
			}

			// 关键改进：获取区间内所有陀螺仪样本（不是单点！）
			std::vector<GyroExtractor::GyroSample> samples = gyro.getRange(last_frame_ts, target_ts_ms);

			if (samples.size() < 2) {
				// 样本不足，退化为单点采样（但警告）
				std::cout << "[WARN] Insufficient gyro samples: " << samples.size()
					<< " in [" << last_frame_ts << ", " << target_ts_ms << "] ms" << std::endl;

				// 退化方案：使用单点
				Vec3 g = gyro.getAt(target_ts_ms) - gyro_bias_;
				integrateSingleStep(g, dt_total);
				last_frame_ts = target_ts_ms;
				return;
			}

			//  Gyroflow 风格：梯形积分遍历所有相邻样本对
			Quaternion delta_total = Quaternion::identity();
			double integrated_time = 0.0;

			for (size_t i = 1; i < samples.size(); i++) {
				const auto& s0 = samples[i - 1];
				const auto& s1 = samples[i];

				double dt = (s1.timestamp_ms - s0.timestamp_ms) / 1000.0;  // 秒
				if (dt <= 0 || dt > 0.1) continue;  // 跳过异常间隔

													// 零偏校正
				Vec3 g0 = s0.data - gyro_bias_;
				Vec3 g1 = s1.data - gyro_bias_;

				//  梯形法则：使用两个端点的平均值
				Vec3 g_avg = (g0 + g1) * 0.5;

				// 限制最大角速度（异常值抑制）
				double g_norm = g_avg.length();
				if (g_norm > MAX_GYRO_NORM) {
					g_avg = g_avg * (MAX_GYRO_NORM / g_norm);
					g_norm = MAX_GYRO_NORM;
				}

				double angle = g_norm * dt;

				// 安全检查：单步旋转过大
				if (angle > MAX_SINGLE_ROTATION) {
					std::cout << "[WARN] Large step rotation: " << angle * 180 / M_PI
						<< " deg in " << dt * 1000 << " ms, clamped" << std::endl;
					angle = MAX_SINGLE_ROTATION;
				}

				if (angle > 1e-10) {
					Vec3 axis = g_avg.normalized();
					Quaternion delta = Quaternion::fromAxisAngle(axis, angle);
					delta_total = delta_total * delta;  // 累积到局部 delta
					integrated_time += dt;
				}
			}

			// 应用总旋转（右乘积分）
			delta_total.normalize();
			current_orientation = current_orientation * delta_total;
			current_orientation.normalize();

			// 调试输出（每 30 帧或异常时）
			/*if (frame_count % 30 == 0 || integrated_time < dt_total * 0.5) {
				Vec3 euler = current_orientation.toEulerAngles();
				std::cout << "[INTEGRATE] Frame " << frame_count
					<< " | Samples: " << samples.size()
					<< " | Integrated: " << integrated_time * 1000 << "/" << dt_total * 1000 << " ms"
					<< " | Current RPY: (" << euler.x * 180 / M_PI << ", "
					<< euler.y * 180 / M_PI << ", " << euler.z * 180 / M_PI << ") deg" << std::endl;
			}*/

			// 打印每帧的 delta 旋转角度
			double delta_angle = delta_total.angle() * 180 / M_PI;
			Vec3 delta_axis = (delta_total.angle() > 0.001) ?
				Vec3(delta_total.x, delta_total.y, delta_total.z).normalized() : Vec3(0, 0, 0);

			if (frame_count % 30 == 0) {
				std::cout << "[DELTA] Frame " << frame_count
					<< " | Angle: " << delta_angle << "°"
					<< " | Axis: (" << delta_axis.x << "," << delta_axis.y << "," << delta_axis.z << ")"
					<< std::endl;
			}

			last_frame_ts = target_ts_ms;
		}

		// 单步积分（退化方案）
		void integrateSingleStep(const Vec3& g, double dt) {
			double g_norm = g.length();
			if (g_norm < 1e-10) return;

			// 限制角速度
			Vec3 g_clamped = g;
			if (g_norm > MAX_GYRO_NORM) {
				g_clamped = g * (MAX_GYRO_NORM / g_norm);
				g_norm = MAX_GYRO_NORM;
			}

			double angle = g_norm * dt;
			if (angle > MAX_SINGLE_ROTATION) {
				angle = MAX_SINGLE_ROTATION;
			}

			Vec3 axis = g_clamped.normalized();
			Quaternion delta = Quaternion::fromAxisAngle(axis, angle);
			current_orientation = current_orientation * delta;
			current_orientation.normalize();
		}

		// 估计陀螺仪零偏（使用前 500ms 的静止数据）
		void estimateGyroBias() {
			if (gyro.samples.empty()) return;

			double duration_ms = 500.0;  // 前 500ms
			double start_ts = gyro.samples.front().timestamp_ms;
			double end_ts = start_ts + duration_ms;

			Vec3 sum;
			size_t count = 0;

			for (const auto& s : gyro.samples) {
				if (s.timestamp_ms > end_ts) break;
				sum = sum + s.data;
				count++;
			}

			if (count > 0) {
				gyro_bias_ = sum * (1.0 / count);
				bias_estimated_ = true;
				std::cout << "[GYRO] Bias estimated from " << count << " samples: ("
					<< gyro_bias_.x << ", " << gyro_bias_.y << ", " << gyro_bias_.z
					<< ") rad/s (" << gyro_bias_.length() * 180 / M_PI << " deg/s magnitude)" << std::endl;
			}
		}


		cv::Mat applyAdaptiveZoom(const cv::Mat& frame, const Quaternion& compensation) {
			int w = frame.cols, h = frame.rows;

			Vec3 euler = compensation.toEulerAngles();
			double max_angle = std::max({ std::abs(euler.x), std::abs(euler.y), std::abs(euler.z) });

			zoom_history.push_back({ (double)frame_count, max_angle });
			if (zoom_history.size() > 60) zoom_history.pop_front();

			double max_hist_angle = 0;
			for (const auto& z : zoom_history) {
				max_hist_angle = std::max(max_hist_angle, z.second);
			}

			// 跑步场景：每 0.1 弧度需要约 3% 额外缩放
			double angle_scale = 1.0 + max_hist_angle * 0.3;
			double base_scale = lens.fov_scale * user_fov_scale;
			double safety_margin = 0.96;
			double final_scale = base_scale / angle_scale * safety_margin;

			final_scale = clamp_val(final_scale, 0.3, 1.0);
			if (final_scale >= 0.995) return frame.clone();

			int new_w = int(w * final_scale);
			int new_h = int(h * final_scale);
			new_w = (new_w / 2) * 2;
			new_h = (new_h / 2) * 2;

			int x = (w - new_w) / 2;
			int y = (h - new_h) / 2;
			if (x < 0 || y < 0 || new_w <= 0 || new_h <= 0) return frame.clone();

			cv::Rect roi(x, y, new_w, new_h);
			cv::Mat cropped = frame(roi);

			cv::Mat result;
			cv::resize(cropped, result, cv::Size(w, h), 0, 0, cv::INTER_LANCZOS4);
			return result;
		}
	};

	// ==================== 处理器 ====================
	class Processor {
	public:
		Stabilizer stab;
		std::string input, output;
		bool show_preview = false;

		Processor(const std::string& in, const std::string& out) : input(in), output(out) {}

		// 修正后的 Processor::run
		bool run(double smoothness, double fov, const std::string& lens,
			int max_frames = 0, double delay = 0.0, bool preview = false) {

			show_preview = preview;

			if (!stab.init(input, lens)) return false;

			// 获取陀螺仪时间原点偏移（使陀螺仪时间从0开始）
			double gyro_time_offset_ms = 0.0;
			if (!stab.gyro.samples.empty()) {
				gyro_time_offset_ms = stab.gyro.samples.front().timestamp_ms;
				std::cout << "[TIME] Gyro time offset: " << gyro_time_offset_ms << " ms" << std::endl;
			}

			stab.setParams(0.92, 0.85, 3.0);
			stab.reset();

			cv::VideoCapture cap(input);
			if (!cap.isOpened()) return false;

			double fps = cap.get(cv::CAP_PROP_FPS);
			int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
			int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
			int total = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);

			int process_frames = (max_frames > 0) ? std::min(max_frames, total) : total;

			cv::VideoWriter writer(output, cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
				fps, cv::Size(w, h));
			if (!writer.isOpened()) {
				std::cerr << "Failed to create output video!" << std::endl;
				return false;
			}

			std::cout << "Processing " << process_frames << " frames..............." <<"total"<< total<< std::endl;

			cv::Mat frame;
			int count = 0;
			double first_video_ts_ms = -1.0;  // 记录第一帧视频时间

			while (cap.read(frame) && count < process_frames) {
				// ========== 关键修正：获取实际视频时间戳 ==========
				double video_ts_ms = cap.get(cv::CAP_PROP_POS_MSEC);  // 实际PTS（毫秒）

																	  // 第一帧作为时间原点
				if (first_video_ts_ms < 0) {
					first_video_ts_ms = video_ts_ms;
					std::cout << "[TIME] First video frame at: " << first_video_ts_ms << " ms" << std::endl;
				}

				// 计算相对时间戳（从0开始）
				double relative_ts_ms = video_ts_ms - first_video_ts_ms;

				// 陀螺仪时间也要相对化：减去陀螺仪起点，使两者对齐
				double aligned_gyro_ts_ms = relative_ts_ms + gyro_time_offset_ms;

				// 调试输出（前5帧和每30帧）
				if ( count % 30 == 0) {
					std::cout << "[TIME] Frame#" << count
						<< " | Video PTS: " << video_ts_ms
						<< " | Relative: " << relative_ts_ms
						<< " | Aligned for gyro: " << aligned_gyro_ts_ms << std::endl;
				}

				cv::Mat out = stab.process(frame, aligned_gyro_ts_ms);
				writer.write(out);

				if (show_preview && count % 2 == 0) {
					cv::imshow("Stabilized", out);
					if (cv::waitKey(1) == 27) break;
				}

				if (++count <100) {
					std::cout << "\r" << (100 * count / process_frames) << "% ("
						<< count << "/" << process_frames << ")" << std::flush;
				}
			}

			std::cout << std::endl << "Done! Processed " << count << " frames." << std::endl;
			cap.release();
			writer.release();
			cv::destroyAllWindows();
			return true;
		}
	};

} // namespace

int main(int argc, char** argv) {
	using namespace Gyroflow;

	if (argc < 3) {
		std::cout << "Usage: " << argv[0] << " <input.MP4> <output.mp4> [options]" << std::endl;
		std::cout << "Options:" << std::endl;
		std::cout << "  --smoothness 0.0-1.0  (default: 0.8, higher = smoother)" << std::endl;
		std::cout << "  --fov 0.3-1.0         (default: 0.9, lower = more crop)" << std::endl;
		std::cout << "  --lens wide|linear    (default: wide)" << std::endl;
		std::cout << "  --delay N             (delay frames, default: 0, recommended: 2-4)" << std::endl;
		std::cout << "  --frames N            (test mode: process only N frames)" << std::endl;
		std::cout << "  --preview             (show preview window)" << std::endl;
		std::cout << std::endl;
		std::cout << "Recommended for walking:" << std::endl;
		std::cout << "  " << argv[0] << " input.MP4 out.mp4 --smoothness 0.85 --fov 0.88 --delay 2" << std::endl;
		return 1;
	}

	std::string input = argv[1];
	std::string output = argv[2];
	double smoothness = 0.8;
	double fov = 0.9;
	std::string lens = "wide";
	int max_frames = 0;
	double delay = 0.0;
	bool preview = false;

	for (int i = 3; i < argc; i++) {
		std::string arg = argv[i];
		if (arg == "--smoothness" && i + 1 < argc) smoothness = std::stod(argv[++i]);
		else if (arg == "--fov" && i + 1 < argc) fov = std::stod(argv[++i]);
		else if (arg == "--lens" && i + 1 < argc) lens = argv[++i];
		else if (arg == "--frames" && i + 1 < argc) max_frames = std::stoi(argv[++i]);
		else if (arg == "--delay" && i + 1 < argc) delay = std::stod(argv[++i]);
		else if (arg == "--preview") preview = true;
	}

	std::cout << "========================================" << std::endl;
	std::cout << "Gyroflow Stabilizer (Walking Optimized)" << std::endl;
	std::cout << "Input:  " << input << std::endl;
	std::cout << "Output: " << output << std::endl;
	std::cout << "Smooth: " << smoothness << std::endl;
	std::cout << "FOV:    " << fov << std::endl;
	std::cout << "Delay:  " << delay << " frames" << std::endl;
	std::cout << "========================================" << std::endl;

	Processor proc(input, output);
	if (!proc.run(smoothness, fov, lens, max_frames, delay, preview)) {
		std::cerr << "Failed!" << std::endl;
		return 1;
	}

	std::cout << "Saved: " << output << std::endl;
	return 0;
}