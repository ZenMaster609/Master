#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <eufs_msgs/msg/cone_array_with_covariance.hpp>
#include <eufs_msgs/msg/cone_with_covariance.hpp>
#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>

#include <gz/math/Pose3.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Link.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/ParentEntity.hh>

#include <ignition/plugin/Register.hh>

namespace gz_sim = ignition::gazebo;

namespace gazebo_plugins {
namespace eufs_plugins {

namespace {
constexpr double kPi = 3.14159265358979323846;

enum class ConeType { kBlue, kYellow, kOrange, kBigOrange, kUnknown };

std::string ConeTypeToString(ConeType t) {
  switch (t) {
    case ConeType::kBlue:
      return "blue";
    case ConeType::kYellow:
      return "yellow";
    case ConeType::kOrange:
      return "orange";
    case ConeType::kBigOrange:
      return "big_orange";
    case ConeType::kUnknown:
    default:
      return "unknown_color";
  }
}

ConeType ConeTypeFromLinkName(const std::string &name) {
  if (name.rfind("blue_cone", 0) == 0) {
    return ConeType::kBlue;
  }
  if (name.rfind("yellow_cone", 0) == 0) {
    return ConeType::kYellow;
  }
  if (name.rfind("orange_cone", 0) == 0) {
    return ConeType::kOrange;
  }
  if (name.rfind("big_cone", 0) == 0) {
    return ConeType::kBigOrange;
  }
  return ConeType::kUnknown;
}

void AppendCone(eufs_msgs::msg::ConeArrayWithCovariance &msg,
                ConeType type,
                const eufs_msgs::msg::ConeWithCovariance &cone) {
  switch (type) {
    case ConeType::kBlue:
      msg.blue_cones.push_back(cone);
      break;
    case ConeType::kYellow:
      msg.yellow_cones.push_back(cone);
      break;
    case ConeType::kOrange:
      msg.orange_cones.push_back(cone);
      break;
    case ConeType::kBigOrange:
      msg.big_orange_cones.push_back(cone);
      break;
    case ConeType::kUnknown:
    default:
      msg.unknown_color_cones.push_back(cone);
      break;
  }
}

void SetStamp(eufs_msgs::msg::ConeArrayWithCovariance &msg,
              const std::chrono::steady_clock::duration &sim_time) {
  const auto ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(sim_time).count();
  msg.header.stamp.sec = static_cast<int32_t>(ns / 1000000000LL);
  msg.header.stamp.nanosec = static_cast<uint32_t>(ns % 1000000000LL);
}

}  // namespace

class GazeboGroundTruthConesRemastered final
    : public gz_sim::System,
      public gz_sim::ISystemConfigure,
      public gz_sim::ISystemPreUpdate {
 public:
  void Configure(const gz_sim::Entity &entity,
                 const std::shared_ptr<const sdf::Element> &sdf,
                 gz_sim::EntityComponentManager &ecm,
                 gz_sim::EventManager &) override {
    model_ = gz_sim::Model(entity);
    if (!model_.Valid(ecm)) {
      RCLCPP_ERROR(rclcpp::get_logger("gazebo_cone_plugins"),
                   "Plugin attached to invalid model entity");
      return;
    }

    InitRos();
    ParseSdf(sdf);

    if (car_frame_link_name_.empty()) {
      car_frame_link_name_ = "base_footprint";
    }

    car_link_entity_ = model_.LinkByName(ecm, car_frame_link_name_);
    if (car_link_entity_ == gz_sim::kNullEntity) {
      RCLCPP_ERROR(node_->get_logger(), "Car frame link '%s' not found",
                   car_frame_link_name_.c_str());
      return;
    }

    RefreshTrackLinks(ecm);

    last_publish_time_sec_ = 0.0;
    rng_.seed(std::random_device{}());

    RCLCPP_INFO(node_->get_logger(),
                "Fortress cone plugin configured. track_model='%s' car_link='%s'",
                track_model_name_.c_str(), car_frame_link_name_.c_str());
  }

  void PreUpdate(const gz_sim::UpdateInfo &info,
                 gz_sim::EntityComponentManager &ecm) override {
    if (info.paused || !node_) {
      return;
    }

    if (cone_entities_.empty() || track_model_entity_ == gz_sim::kNullEntity) {
      RefreshTrackLinks(ecm);
      if (cone_entities_.empty()) {
        return;
      }
    }

    const double sim_time_sec =
        std::chrono::duration<double>(info.simTime).count();
    if (update_rate_hz_ > 0.0 &&
        (sim_time_sec - last_publish_time_sec_) < (1.0 / update_rate_hz_)) {
      return;
    }
    last_publish_time_sec_ = sim_time_sec;

    auto track_msg_map = BuildTrackConesMapFrame(ecm, info);

    if (publish_track_ && track_pub_ && track_pub_->get_subscription_count() > 0) {
      auto track_msg = track_msg_map;
      if (track_frame_ == "base_footprint") {
        const auto car_pose = gz_sim::worldPose(car_link_entity_, ecm);
        track_msg = TranslateToFrame(track_msg_map, car_pose, "base_footprint");
        SetStamp(track_msg, info.simTime);
      }
      track_pub_->publish(track_msg);
    }

    if (!visible_pub_ || visible_pub_->get_subscription_count() == 0) {
      return;
    }

    const auto car_pose = gz_sim::worldPose(car_link_entity_, ecm);
    auto visible_msg = BuildVisibleCones(track_msg_map, car_pose, info);
    if (enable_miscolor_) {
      visible_msg = ApplyConfusionMatrix(visible_msg, info);
    }
    visible_pub_->publish(visible_msg);
  }

 private:
  struct ConeEntity {
    gz_sim::Entity entity{gz_sim::kNullEntity};
    ConeType type{ConeType::kUnknown};
  };

  void InitRos() {
    auto context = rclcpp::contexts::get_global_default_context();
    if (!context->is_valid()) {
      rclcpp::init(0, nullptr);
    }

    rclcpp::NodeOptions options;
    options.allow_undeclared_parameters(true);
    options.automatically_declare_parameters_from_overrides(true);
    node_ = std::make_shared<rclcpp::Node>("gazebo_ground_truth_cones");
    node_->set_parameter(rclcpp::Parameter("use_sim_time", true));
  }

  void ParseSdf(const std::shared_ptr<const sdf::Element> &sdf) {
    update_rate_hz_ = GetDouble(sdf, "updateRate", 25.0);

    track_model_name_ = GetString(sdf, "trackModelName", "track");
    car_frame_link_name_ = GetString(sdf, "carFrameLink", "base_footprint");

    track_frame_ = GetString(sdf, "trackFrame", "map");
    visible_frame_ = GetString(sdf, "visibleFrame", "base_footprint");

    camera_total_view_distance_ = GetDouble(sdf, "cameraViewDistance", 15.0);
    camera_min_view_distance_ = GetDouble(sdf, "cameraMinViewDistance", 0.5);
    camera_fov_ = GetDouble(sdf, "cameraFOV", 2.09);

    lidar_on_ = GetBool(sdf, "lidarOn", true);
    lidar_total_view_distance_ = GetDouble(sdf, "lidarViewDistance", 100.0);
    lidar_min_view_distance_ = GetDouble(sdf, "lidarMinViewDistance", 1.0);
    lidar_x_view_distance_ = GetDouble(sdf, "lidarXViewDistance", 20.0);
    lidar_y_view_distance_ = GetDouble(sdf, "lidarYViewDistance", 20.0);
    lidar_fov_ = GetDouble(sdf, "lidarFOV", kPi);

    publish_track_ = GetBool(sdf, "publishTrack", true);

    const std::string track_topic =
        GetString(sdf, "groundTruthTrackTopicName", "/ground_truth/track");
    std::string visible_topic =
        GetString(sdf, "visibleConesTopicName", "/ground_truth/cones");
    if (sdf->HasElement("groundTruthConesTopicName")) {
      visible_topic = GetString(sdf, "groundTruthConesTopicName", visible_topic);
    }

    track_pub_ =
        node_->create_publisher<eufs_msgs::msg::ConeArrayWithCovariance>(track_topic, 1);
    visible_pub_ =
        node_->create_publisher<eufs_msgs::msg::ConeArrayWithCovariance>(visible_topic, 1);

    enable_miscolor_ = GetBool(sdf, "enableConeMiscoloring", false);

    std::string confusion_matrix_yaml;
    if (sdf->HasElement("confusionMatrixYaml")) {
      confusion_matrix_yaml = GetString(sdf, "confusionMatrixYaml", "");
    } else if (sdf->HasElement("recolor_config")) {
      confusion_matrix_yaml = GetString(sdf, "recolor_config", "");
      enable_miscolor_ = true;
    }

    if (enable_miscolor_) {
      if (confusion_matrix_yaml.empty()) {
        RCLCPP_ERROR(node_->get_logger(),
                     "Miscoloring enabled but no confusion matrix YAML path provided");
        enable_miscolor_ = false;
      } else {
        try {
          confusion_matrix_ = YAML::LoadFile(confusion_matrix_yaml);
        } catch (const std::exception &e) {
          RCLCPP_ERROR(node_->get_logger(),
                       "Failed loading confusion matrix '%s': %s",
                       confusion_matrix_yaml.c_str(), e.what());
          enable_miscolor_ = false;
        }
      }
    }
  }

  void RefreshTrackLinks(gz_sim::EntityComponentManager &ecm) {
    cone_entities_.clear();
    track_model_entity_ = gz_sim::kNullEntity;

    ecm.Each<gz_sim::components::Model, gz_sim::components::Name>(
        [this](const gz_sim::Entity &entity,
               const gz_sim::components::Model *,
               const gz_sim::components::Name *name) {
          if (name && name->Data() == track_model_name_) {
            track_model_entity_ = entity;
            return false;
          }
          return true;
        });

    if (track_model_entity_ != gz_sim::kNullEntity) {
      ecm.Each<gz_sim::components::Model,
               gz_sim::components::Name,
               gz_sim::components::ParentEntity>(
          [this](const gz_sim::Entity &entity,
                 const gz_sim::components::Model *,
                 const gz_sim::components::Name *name,
                 const gz_sim::components::ParentEntity *parent) {
            if (!name || !parent || parent->Data() != track_model_entity_) {
              return true;
            }
            const auto type = ConeTypeFromLinkName(name->Data());
            if (type != ConeType::kUnknown) {
              cone_entities_.push_back({entity, type});
            }
            return true;
          });
      return;
    }

    // Fallback: collect cone models globally when track model name is unknown.
    ecm.Each<gz_sim::components::Model, gz_sim::components::Name>(
        [this](const gz_sim::Entity &entity,
               const gz_sim::components::Model *,
               const gz_sim::components::Name *name) {
          if (!name) {
            return true;
          }
          const auto type = ConeTypeFromLinkName(name->Data());
          if (type != ConeType::kUnknown) {
            cone_entities_.push_back({entity, type});
          }
          return true;
        });

    if (!cone_entities_.empty()) {
      RCLCPP_WARN_ONCE(node_->get_logger(),
                       "Track model '%s' not found; using global cone-model fallback",
                       track_model_name_.c_str());
    }
  }

  eufs_msgs::msg::ConeArrayWithCovariance BuildTrackConesMapFrame(
      gz_sim::EntityComponentManager &ecm,
      const gz_sim::UpdateInfo &info) {
    eufs_msgs::msg::ConeArrayWithCovariance msg;
    msg.header.frame_id = "map";
    SetStamp(msg, info.simTime);

    for (const auto &cone_ref : cone_entities_) {
      const auto pose = gz_sim::worldPose(cone_ref.entity, ecm);

      eufs_msgs::msg::ConeWithCovariance cone;
      cone.point.x = pose.Pos().X();
      cone.point.y = pose.Pos().Y();
      cone.point.z = 0.0;
      cone.covariance = {0.0, 0.0, 0.0, 0.0};

      AppendCone(msg, cone_ref.type, cone);
    }

    return msg;
  }

  eufs_msgs::msg::ConeArrayWithCovariance BuildVisibleCones(
      const eufs_msgs::msg::ConeArrayWithCovariance &map_cones,
      const gz::math::Pose3d &car_pose,
      const gz_sim::UpdateInfo &info) const {
    const auto cones_car = TranslateToFrame(map_cones, car_pose, visible_frame_);

    eufs_msgs::msg::ConeArrayWithCovariance out;
    out.header.frame_id = visible_frame_;
    SetStamp(out, info.simTime);

    ProcessVisibleForColor(cones_car.blue_cones, ConeType::kBlue, out);
    ProcessVisibleForColor(cones_car.yellow_cones, ConeType::kYellow, out);
    ProcessVisibleForColor(cones_car.orange_cones, ConeType::kOrange, out);
    ProcessVisibleForColor(cones_car.big_orange_cones, ConeType::kBigOrange, out);
    ProcessVisibleForColor(cones_car.unknown_color_cones, ConeType::kUnknown, out);

    return out;
  }

  void ProcessVisibleForColor(
      const std::vector<eufs_msgs::msg::ConeWithCovariance> &source,
      ConeType color,
      eufs_msgs::msg::ConeArrayWithCovariance &out) const {
    for (const auto &cone : source) {
      const bool lidar_sees = InRangeOfLidar(cone) && InFOVOfLidar(cone);
      const bool camera_sees = InRangeOfCamera(cone) && InFOVOfCamera(cone);

      if (camera_sees) {
        AppendCone(out, color, cone);
      } else if (lidar_sees) {
        AppendCone(out, ConeType::kUnknown, cone);
      }
    }
  }

  bool InRangeOfCamera(const eufs_msgs::msg::ConeWithCovariance &cone) const {
    const double dist_sq = (cone.point.x * cone.point.x) + (cone.point.y * cone.point.y);
    return camera_min_view_distance_ * camera_min_view_distance_ < dist_sq &&
           dist_sq < camera_total_view_distance_ * camera_total_view_distance_;
  }

  bool InFOVOfCamera(const eufs_msgs::msg::ConeWithCovariance &cone) const {
    const double angle = std::atan2(cone.point.y, cone.point.x);
    return std::abs(angle) < (camera_fov_ / 2.0);
  }

  bool InRangeOfLidar(const eufs_msgs::msg::ConeWithCovariance &cone) const {
    if (!lidar_on_) {
      return false;
    }
    const double dist_sq = (cone.point.x * cone.point.x) + (cone.point.y * cone.point.y);
    return lidar_min_view_distance_ * lidar_min_view_distance_ < dist_sq &&
           dist_sq < lidar_total_view_distance_ * lidar_total_view_distance_ &&
           std::abs(cone.point.x) < lidar_x_view_distance_ &&
           std::abs(cone.point.y) < lidar_y_view_distance_;
  }

  bool InFOVOfLidar(const eufs_msgs::msg::ConeWithCovariance &cone) const {
    if (!lidar_on_) {
      return false;
    }
    const double angle = std::atan2(cone.point.y, cone.point.x);
    return std::abs(angle) < (lidar_fov_ / 2.0);
  }

  eufs_msgs::msg::ConeArrayWithCovariance ApplyConfusionMatrix(
      const eufs_msgs::msg::ConeArrayWithCovariance &input,
      const gz_sim::UpdateInfo &info) {
    std::vector<std::pair<ConeType, std::vector<eufs_msgs::msg::ConeWithCovariance>>> groups = {
        {ConeType::kBlue, input.blue_cones},
        {ConeType::kYellow, input.yellow_cones},
        {ConeType::kOrange, input.orange_cones},
        {ConeType::kBigOrange, input.big_orange_cones},
        {ConeType::kUnknown, input.unknown_color_cones},
    };

    eufs_msgs::msg::ConeArrayWithCovariance out;
    out.header.frame_id = input.header.frame_id;
    SetStamp(out, info.simTime);

    for (const auto &[source_type, cones] : groups) {
      const std::string source = ConeTypeToString(source_type);
      for (const auto &cone : cones) {
        const std::string dest = SampleColor(source);
        if (dest == "undetected") {
          continue;
        }
        AppendCone(out, StringToConeType(dest), cone);
      }
    }

    return out;
  }

  std::string SampleColor(const std::string &source) {
    static const std::array<std::string, 6> kOrder = {
        "blue", "yellow", "orange", "big_orange", "unknown_color", "undetected"};

    if (!confusion_matrix_[source]) {
      return source;
    }

    std::array<double, 6> weights = {0, 0, 0, 0, 0, 0};
    double sum = 0.0;
    for (size_t i = 0; i < kOrder.size(); ++i) {
      if (confusion_matrix_[source][kOrder[i]]) {
        weights[i] = std::max(0.0, confusion_matrix_[source][kOrder[i]].as<double>());
        sum += weights[i];
      }
    }

    if (sum <= 0.0) {
      return source;
    }

    std::uniform_real_distribution<double> dist(0.0, sum);
    const double r = dist(rng_);
    double csum = 0.0;
    for (size_t i = 0; i < kOrder.size(); ++i) {
      csum += weights[i];
      if (r <= csum) {
        return kOrder[i];
      }
    }

    return source;
  }

  static ConeType StringToConeType(const std::string &color) {
    if (color == "blue") {
      return ConeType::kBlue;
    }
    if (color == "yellow") {
      return ConeType::kYellow;
    }
    if (color == "orange") {
      return ConeType::kOrange;
    }
    if (color == "big_orange") {
      return ConeType::kBigOrange;
    }
    return ConeType::kUnknown;
  }

  static std::vector<eufs_msgs::msg::ConeWithCovariance> TranslateCones(
      const std::vector<eufs_msgs::msg::ConeWithCovariance> &cones,
      const gz::math::Pose3d &frame) {
    std::vector<eufs_msgs::msg::ConeWithCovariance> out;
    out.reserve(cones.size());

    const double yaw = frame.Rot().Yaw();
    const double c = std::cos(yaw);
    const double s = std::sin(yaw);

    for (const auto &cone : cones) {
      const double x = cone.point.x - frame.Pos().X();
      const double y = cone.point.y - frame.Pos().Y();

      auto t = cone;
      t.point.y = (c * y) - (s * x);
      t.point.x = (s * y) + (c * x);
      out.push_back(t);
    }

    return out;
  }

  static eufs_msgs::msg::ConeArrayWithCovariance TranslateToFrame(
      const eufs_msgs::msg::ConeArrayWithCovariance &msg,
      const gz::math::Pose3d &frame,
      const std::string &frame_id) {
    auto out = msg;
    out.header.frame_id = frame_id;
    out.blue_cones = TranslateCones(msg.blue_cones, frame);
    out.yellow_cones = TranslateCones(msg.yellow_cones, frame);
    out.orange_cones = TranslateCones(msg.orange_cones, frame);
    out.big_orange_cones = TranslateCones(msg.big_orange_cones, frame);
    out.unknown_color_cones = TranslateCones(msg.unknown_color_cones, frame);
    return out;
  }

  static bool GetBool(const std::shared_ptr<const sdf::Element> &sdf,
                      const std::string &name,
                      bool default_value) {
    return sdf->HasElement(name) ? sdf->Get<bool>(name) : default_value;
  }

  static double GetDouble(const std::shared_ptr<const sdf::Element> &sdf,
                          const std::string &name,
                          double default_value) {
    return sdf->HasElement(name) ? sdf->Get<double>(name) : default_value;
  }

  static std::string GetString(const std::shared_ptr<const sdf::Element> &sdf,
                               const std::string &name,
                               const std::string &default_value) {
    return sdf->HasElement(name) ? sdf->Get<std::string>(name) : default_value;
  }

  gz_sim::Model model_;
  gz_sim::Entity track_model_entity_ = gz_sim::kNullEntity;
  gz_sim::Entity car_link_entity_ = gz_sim::kNullEntity;
  std::vector<ConeEntity> cone_entities_;

  std::shared_ptr<rclcpp::Node> node_;
  rclcpp::Publisher<eufs_msgs::msg::ConeArrayWithCovariance>::SharedPtr track_pub_;
  rclcpp::Publisher<eufs_msgs::msg::ConeArrayWithCovariance>::SharedPtr visible_pub_;

  std::mt19937 rng_;

  std::string track_model_name_ = "track";
  std::string car_frame_link_name_ = "base_footprint";

  double update_rate_hz_ = 25.0;
  double last_publish_time_sec_ = 0.0;

  bool publish_track_ = true;
  std::string track_frame_ = "map";
  std::string visible_frame_ = "base_footprint";

  double camera_total_view_distance_ = 15.0;
  double camera_min_view_distance_ = 0.5;
  double camera_fov_ = 2.09;

  bool lidar_on_ = true;
  double lidar_total_view_distance_ = 100.0;
  double lidar_min_view_distance_ = 1.0;
  double lidar_x_view_distance_ = 20.0;
  double lidar_y_view_distance_ = 20.0;
  double lidar_fov_ = kPi;

  bool enable_miscolor_ = false;
  YAML::Node confusion_matrix_;
};

IGNITION_ADD_PLUGIN(
    GazeboGroundTruthConesRemastered,
    gz_sim::System,
    gz_sim::ISystemConfigure,
    gz_sim::ISystemPreUpdate)

IGNITION_ADD_PLUGIN_ALIAS(
    GazeboGroundTruthConesRemastered,
    "gazebo_plugins::eufs_plugins::GazeboGroundTruthConesRemastered")

IGNITION_ADD_PLUGIN_ALIAS(
    GazeboGroundTruthConesRemastered,
    "gazebo_plugins::eufs_plugins::GazeboGroundTruthCones")

}  // namespace eufs_plugins
}  // namespace gazebo_plugins
