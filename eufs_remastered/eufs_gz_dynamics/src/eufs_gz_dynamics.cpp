#include <algorithm>
#include <cmath>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/contexts/default_context.hpp>

#include <gz/math/Pose3.hh>
#include <gz/math/Vector3.hh>
#include <gz/sim/Joint.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>

#include <ignition/plugin/Register.hh>

#include "eufs_models/eufs_models.hpp"

namespace gz_sim = ignition::gazebo;

namespace eufs_gz_dynamics {

struct Command {
  double stamp_sec = 0.0;
  eufs::models::Input input;
};

class EufsRaceCarModel final : public gz_sim::System,
                               public gz_sim::ISystemConfigure,
                               public gz_sim::ISystemPreUpdate {
 public:
  EufsRaceCarModel() = default;
  ~EufsRaceCarModel() override {
    if (spin_thread_.joinable()) {
      if (executor_) {
        executor_->cancel();
      }
      spin_thread_.join();
    }
  }

  void Configure(const gz_sim::Entity &_entity,
                 const std::shared_ptr<const sdf::Element> &_sdf,
                 gz_sim::EntityComponentManager &_ecm,
                 gz_sim::EventManager & /*_eventMgr*/) override {
    model_ = gz_sim::Model(_entity);
    if (!model_.Valid(_ecm)) {
      RCLCPP_ERROR(rclcpp::get_logger("eufs_gz_dynamics"),
                   "EufsRaceCarModel attached to invalid model entity");
      return;
    }

    InitRos();

    ParseSdf(_sdf);

    if (yaml_config_path_.empty()) {
      RCLCPP_ERROR(node_->get_logger(), "Missing <yaml_config> for EUFS dynamics plugin");
      return;
    }

    InitVehicleModel();

    auto canonical_link = model_.CanonicalLink(_ecm);
    if (canonical_link == gz_sim::kNullEntity) {
      RCLCPP_ERROR(node_->get_logger(), "Failed to resolve canonical link");
      return;
    }
    canonical_link_ = gz_sim::Link(canonical_link);

    left_steering_joint_ = gz_sim::Joint(model_.JointByName(_ecm, left_steering_joint_name_));
    right_steering_joint_ = gz_sim::Joint(model_.JointByName(_ecm, right_steering_joint_name_));
    front_left_wheel_joint_ = gz_sim::Joint(model_.JointByName(_ecm, front_left_wheel_joint_name_));
    front_right_wheel_joint_ = gz_sim::Joint(model_.JointByName(_ecm, front_right_wheel_joint_name_));
    rear_left_wheel_joint_ = gz_sim::Joint(model_.JointByName(_ecm, rear_left_wheel_joint_name_));
    rear_right_wheel_joint_ = gz_sim::Joint(model_.JointByName(_ecm, rear_right_wheel_joint_name_));

    if (use_initial_pose_override_) {
      offset_ = gz::math::Pose3d(initial_x_, initial_y_, initial_z_, 0.0, 0.0, initial_yaw_);
      offset_initialized_ = true;
      state_.z = initial_z_;
      RCLCPP_INFO(
          node_->get_logger(),
          "Using configured initial pose x=%.3f y=%.3f z=%.3f yaw=%.3f",
          initial_x_, initial_y_, initial_z_, initial_yaw_);
    }

    last_update_time_sec_ = 0.0;
    last_cmd_time_sec_ = node_->now().seconds();

    RCLCPP_INFO(node_->get_logger(), "EUFS dynamics plugin configured");
  }

  void PreUpdate(const gz_sim::UpdateInfo &_info,
                 gz_sim::EntityComponentManager &_ecm) override {
    if (_info.paused || !vehicle_) {
      return;
    }

    const double sim_time_sec = std::chrono::duration<double>(_info.simTime).count();
    const double dt = std::chrono::duration<double>(_info.dt).count();
    if (dt <= 0.0) {
      return;
    }

    if (update_rate_hz_ > 0.0 &&
        (sim_time_sec - last_update_time_sec_) < (1.0 / update_rate_hz_)) {
      return;
    }
    last_update_time_sec_ = sim_time_sec;

    const double now_sec = node_ ? node_->now().seconds() : sim_time_sec;
    ApplyPendingCommands(now_sec);

    if (command_mode_ == CommandMode::kVelocity) {
      const double current_speed = std::hypot(state_.v_x, state_.v_y);
      desired_input_.acc = (desired_input_.vel - current_speed) / dt;
    }

    actual_input_.acc = (now_sec - last_cmd_time_sec_ < 1.0) ? desired_input_.acc : -1.0;
    UpdateSteering(dt);

    auto pose = canonical_link_.WorldPose(_ecm);
    if (pose.has_value()) {
      if (!offset_initialized_) {
        offset_ = pose.value();
        offset_initialized_ = true;
        RCLCPP_INFO(
            node_->get_logger(),
            "Initial dynamics offset set to x=%.3f y=%.3f z=%.3f",
            offset_.Pos().X(), offset_.Pos().Y(), offset_.Pos().Z());
      }
      state_.z = pose->Pos().Z();
    }

    vehicle_->updateState(state_, actual_input_, dt);

    ApplyModelState(_ecm);
    UpdateWheelJoints(_ecm, dt);
    UpdateSteeringJoints(_ecm);
  }

 private:
  enum class CommandMode { kAcceleration, kVelocity };

  void InitRos() {
    auto context = rclcpp::contexts::get_global_default_context();
    if (!context->is_valid()) {
      rclcpp::init(0, nullptr);
    }

    rclcpp::NodeOptions node_options;
    node_options.allow_undeclared_parameters(true);
    node_options.automatically_declare_parameters_from_overrides(true);
    node_ = std::make_shared<rclcpp::Node>("eufs_gz_dynamics", node_options);
    node_->set_parameter(rclcpp::Parameter("use_sim_time", true));

    if (use_cmd_vel_) {
    sub_cmd_vel_ = node_->create_subscription<geometry_msgs::msg::Twist>(
          cmd_vel_topic_, 10,
          [this](const geometry_msgs::msg::Twist::SharedPtr msg) {
            Command cmd;
            cmd.stamp_sec = node_->now().seconds();
            cmd.input.vel = msg->linear.x;
            cmd.input.acc = 0.0;
            cmd.input.delta = ComputeSteeringFromTwist(msg->linear.x, msg->angular.z);
            EnqueueCommand(std::move(cmd));
          });
    }

    rclcpp::ExecutorOptions exec_options;
    exec_options.context = node_->get_node_base_interface()->get_context();
    executor_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>(exec_options);
    executor_->add_node(node_);
    spin_thread_ = std::thread([this]() { executor_->spin(); });
  }

  void ParseSdf(const std::shared_ptr<const sdf::Element> &_sdf) {
    if (_sdf->HasElement("vehicle_model")) {
      vehicle_model_ = _sdf->Get<std::string>("vehicle_model");
    }
    if (_sdf->HasElement("yaml_config")) {
      yaml_config_path_ = _sdf->Get<std::string>("yaml_config");
    }
    if (_sdf->HasElement("update_rate")) {
      update_rate_hz_ = _sdf->Get<double>("update_rate");
    }
    if (_sdf->HasElement("control_delay")) {
      control_delay_sec_ = _sdf->Get<double>("control_delay");
    }
    if (_sdf->HasElement("steering_lock_time")) {
      steering_lock_time_ = _sdf->Get<double>("steering_lock_time");
    }
    if (_sdf->HasElement("command_mode")) {
      const auto mode = _sdf->Get<std::string>("command_mode");
      command_mode_ = (mode == "velocity") ? CommandMode::kVelocity : CommandMode::kAcceleration;
    }
    if (_sdf->HasElement("use_cmd_vel")) {
      use_cmd_vel_ = _sdf->Get<bool>("use_cmd_vel");
    }
    if (_sdf->HasElement("cmd_vel_topic")) {
      cmd_vel_topic_ = _sdf->Get<std::string>("cmd_vel_topic");
    }
    if (_sdf->HasElement("initial_x")) {
      initial_x_ = _sdf->Get<double>("initial_x");
      use_initial_pose_override_ = true;
    }
    if (_sdf->HasElement("initial_y")) {
      initial_y_ = _sdf->Get<double>("initial_y");
      use_initial_pose_override_ = true;
    }
    if (_sdf->HasElement("initial_z")) {
      initial_z_ = _sdf->Get<double>("initial_z");
      use_initial_pose_override_ = true;
    }
    if (_sdf->HasElement("initial_yaw")) {
      initial_yaw_ = _sdf->Get<double>("initial_yaw");
      use_initial_pose_override_ = true;
    }

    if (_sdf->HasElement("front_left_wheel_steering")) {
      left_steering_joint_name_ = _sdf->Get<std::string>("front_left_wheel_steering");
    }
    if (_sdf->HasElement("front_right_wheel_steering")) {
      right_steering_joint_name_ = _sdf->Get<std::string>("front_right_wheel_steering");
    }
    if (_sdf->HasElement("front_left_wheel")) {
      front_left_wheel_joint_name_ = _sdf->Get<std::string>("front_left_wheel");
    }
    if (_sdf->HasElement("front_right_wheel")) {
      front_right_wheel_joint_name_ = _sdf->Get<std::string>("front_right_wheel");
    }
    if (_sdf->HasElement("rear_left_wheel")) {
      rear_left_wheel_joint_name_ = _sdf->Get<std::string>("rear_left_wheel");
    }
    if (_sdf->HasElement("rear_right_wheel")) {
      rear_right_wheel_joint_name_ = _sdf->Get<std::string>("rear_right_wheel");
    }
  }

  void InitVehicleModel() {
    if (vehicle_model_ == "PointMass") {
      vehicle_ = std::make_unique<eufs::models::PointMass>(yaml_config_path_);
    } else if (vehicle_model_ == "DynamicBicycle") {
      vehicle_ = std::make_unique<eufs::models::DynamicBicycle>(yaml_config_path_);
    } else {
      RCLCPP_ERROR(node_->get_logger(), "Unknown vehicle model: %s", vehicle_model_.c_str());
      return;
    }

    const auto &params = vehicle_->getParam();
    wheel_radius_ = params.tire.radius;
    wheelbase_ = params.kinematic.l;
    max_steering_rate_ =
        (params.input_ranges.delta.max - params.input_ranges.delta.min) / steering_lock_time_;
  }

  void EnqueueCommand(Command &&cmd) {
    std::lock_guard<std::mutex> lock(cmd_mutex_);
    cmd_queue_.push_back(std::move(cmd));
  }

  void ApplyPendingCommands(double now_sec) {
    std::lock_guard<std::mutex> lock(cmd_mutex_);
    if (cmd_queue_.empty()) {
      return;
    }

    if (control_delay_sec_ <= 0.0) {
      desired_input_ = cmd_queue_.back().input;
      cmd_queue_.clear();
      last_cmd_time_sec_ = now_sec;
      return;
    }

    if ((now_sec - cmd_queue_.front().stamp_sec) >= control_delay_sec_) {
      desired_input_ = cmd_queue_.front().input;
      cmd_queue_.pop_front();
      last_cmd_time_sec_ = now_sec;
    }
  }

  void UpdateSteering(double dt) {
    const double delta_diff = desired_input_.delta - actual_input_.delta;
    const double step = std::min(max_steering_rate_ * dt, std::abs(delta_diff));
    actual_input_.delta += (delta_diff >= 0.0 ? 1.0 : -1.0) * step;
    actual_input_.vel = desired_input_.vel;
  }

  double ComputeSteeringFromTwist(double linear, double angular) const {
    if (std::abs(linear) < 1e-3) {
      return 0.0;
    }
    return std::atan2(wheelbase_ * angular, linear);
  }

  void ApplyModelState(gz_sim::EntityComponentManager &_ecm) {
    const double yaw = state_.yaw + offset_.Rot().Yaw();
    const double cos_yaw = std::cos(offset_.Rot().Yaw());
    const double sin_yaw = std::sin(offset_.Rot().Yaw());

    const double x = offset_.Pos().X() + state_.x * cos_yaw - state_.y * sin_yaw;
    const double y = offset_.Pos().Y() + state_.x * sin_yaw + state_.y * cos_yaw;
    const double z = state_.z;

    const double vx = state_.v_x * std::cos(yaw) - state_.v_y * std::sin(yaw);
    const double vy = state_.v_x * std::sin(yaw) + state_.v_y * std::cos(yaw);

    model_.SetWorldPoseCmd(_ecm, gz::math::Pose3d(x, y, z, 0.0, 0.0, yaw));
    canonical_link_.SetLinearVelocity(_ecm, gz::math::Vector3d(vx, vy, 0.0));
    canonical_link_.SetAngularVelocity(_ecm, gz::math::Vector3d(0.0, 0.0, state_.r_z));
  }

  void UpdateSteeringJoints(gz_sim::EntityComponentManager &_ecm) {
    if (left_steering_joint_.Valid(_ecm)) {
      left_steering_joint_.ResetPosition(_ecm, {actual_input_.delta});
    }
    if (right_steering_joint_.Valid(_ecm)) {
      right_steering_joint_.ResetPosition(_ecm, {actual_input_.delta});
    }
  }

  void UpdateWheelJoints(gz_sim::EntityComponentManager &_ecm, double dt) {
    if (wheel_radius_ <= 0.0) {
      return;
    }
    const double omega = state_.v_x / wheel_radius_;

    wheel_angle_fl_ += omega * dt;
    wheel_angle_fr_ += omega * dt;
    wheel_angle_rl_ += omega * dt;
    wheel_angle_rr_ += omega * dt;

    if (front_left_wheel_joint_.Valid(_ecm)) {
      front_left_wheel_joint_.ResetPosition(_ecm, {wheel_angle_fl_});
      front_left_wheel_joint_.ResetVelocity(_ecm, {omega});
    }
    if (front_right_wheel_joint_.Valid(_ecm)) {
      front_right_wheel_joint_.ResetPosition(_ecm, {wheel_angle_fr_});
      front_right_wheel_joint_.ResetVelocity(_ecm, {omega});
    }
    if (rear_left_wheel_joint_.Valid(_ecm)) {
      rear_left_wheel_joint_.ResetPosition(_ecm, {wheel_angle_rl_});
      rear_left_wheel_joint_.ResetVelocity(_ecm, {omega});
    }
    if (rear_right_wheel_joint_.Valid(_ecm)) {
      rear_right_wheel_joint_.ResetPosition(_ecm, {wheel_angle_rr_});
      rear_right_wheel_joint_.ResetVelocity(_ecm, {omega});
    }
  }

  gz_sim::Model model_{};
  gz_sim::Link canonical_link_{};
  gz_sim::Joint left_steering_joint_{};
  gz_sim::Joint right_steering_joint_{};
  gz_sim::Joint front_left_wheel_joint_{};
  gz_sim::Joint front_right_wheel_joint_{};
  gz_sim::Joint rear_left_wheel_joint_{};
  gz_sim::Joint rear_right_wheel_joint_{};

  std::shared_ptr<rclcpp::Node> node_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_cmd_vel_;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
  std::thread spin_thread_;

  std::mutex cmd_mutex_;
  std::deque<Command> cmd_queue_;

  eufs::models::State state_{};
  eufs::models::Input desired_input_{};
  eufs::models::Input actual_input_{};
  std::unique_ptr<eufs::models::VehicleModel> vehicle_;

  gz::math::Pose3d offset_{};
  bool offset_initialized_ = false;

  double update_rate_hz_ = 1000.0;
  double steering_lock_time_ = 1.0;
  double max_steering_rate_ = 0.0;
  double control_delay_sec_ = 0.0;
  double last_cmd_time_sec_ = 0.0;
  double last_update_time_sec_ = 0.0;

  double wheel_radius_ = 0.25;
  double wheelbase_ = 1.58;
  double wheel_angle_fl_ = 0.0;
  double wheel_angle_fr_ = 0.0;
  double wheel_angle_rl_ = 0.0;
  double wheel_angle_rr_ = 0.0;

  bool use_initial_pose_override_ = false;
  double initial_x_ = 0.0;
  double initial_y_ = 0.0;
  double initial_z_ = 0.0;
  double initial_yaw_ = 0.0;

  std::string vehicle_model_ = "DynamicBicycle";
  std::string yaml_config_path_;

  std::string cmd_vel_topic_ = "/cmd_vel";
  bool use_cmd_vel_ = true;

  CommandMode command_mode_ = CommandMode::kVelocity;

  std::string left_steering_joint_name_ = "steering_fl_joint";
  std::string right_steering_joint_name_ = "steering_fr_joint";
  std::string front_left_wheel_joint_name_ = "front_left_wheel_joint";
  std::string front_right_wheel_joint_name_ = "front_right_wheel_joint";
  std::string rear_left_wheel_joint_name_ = "rear_left_wheel_joint";
  std::string rear_right_wheel_joint_name_ = "rear_right_wheel_joint";
};

}  // namespace eufs_gz_dynamics

IGNITION_ADD_PLUGIN(
    eufs_gz_dynamics::EufsRaceCarModel,
    gz_sim::System,
    gz_sim::ISystemConfigure,
    gz_sim::ISystemPreUpdate)

IGNITION_ADD_PLUGIN_ALIAS(eufs_gz_dynamics::EufsRaceCarModel,
                          "eufs_gz_dynamics::EufsRaceCarModel")
