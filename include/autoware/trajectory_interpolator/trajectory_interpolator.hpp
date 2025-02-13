// Copyright 2025 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef AUTOWARE__TRAJECTORY_INTERPOLATOR_HPP_
#define AUTOWARE__TRAJECTORY_INTERPOLATOR_HPP_

#include "autoware/trajectory_interpolator/trajectory_interpolator_structs.hpp"
#include "autoware/universe_utils/ros/polling_subscriber.hpp"
#include "autoware/universe_utils/system/time_keeper.hpp"
#include "autoware/velocity_smoother/smoother/jerk_filtered_smoother.hpp"
#include "rclcpp/rclcpp.hpp"

#include <rclcpp/subscription.hpp>

#include "geometry_msgs/msg/accel_with_covariance_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include <autoware_new_planning_msgs/msg/trajectories.hpp>
#include <autoware_perception_msgs/msg/detail/predicted_objects__struct.hpp>
#include <autoware_perception_msgs/msg/predicted_objects.hpp>
#include <autoware_planning_msgs/msg/detail/trajectory__struct.hpp>
#include <autoware_planning_msgs/msg/trajectory.hpp>

#include <string>
#include <vector>

namespace autoware::trajectory_interpolator
{

using autoware::velocity_smoother::JerkFilteredSmoother;
using autoware_new_planning_msgs::msg::Trajectories;
using autoware_perception_msgs::msg::PredictedObjects;
using autoware_planning_msgs::msg::Trajectory;
using autoware_planning_msgs::msg::TrajectoryPoint;
using NewTrajectory = autoware_new_planning_msgs::msg::Trajectory;
using geometry_msgs::msg::AccelWithCovarianceStamped;
using nav_msgs::msg::Odometry;
using TrajectoryPoints = std::vector<TrajectoryPoint>;

class TrajectoryInterpolator : public rclcpp::Node
{
public:
  explicit TrajectoryInterpolator(const rclcpp::NodeOptions & options);

private:
  void on_traj(const Trajectories::ConstSharedPtr msg);
  void set_up_params();

  /**
   * @brief Callback for parameter updates
   * @param parameters Vector of updated parameters
   * @return Set parameters result
   */
  rcl_interfaces::msg::SetParametersResult on_parameter(
    const std::vector<rclcpp::Parameter> & parameters);

  // interface subscriber
  rclcpp::Subscription<Trajectories>::SharedPtr trajectories_sub_;
  // interface publisher
  rclcpp::Publisher<Trajectories>::SharedPtr trajectories_pub_;
  rclcpp::Publisher<autoware::universe_utils::ProcessingTimeDetail>::SharedPtr
    debug_processing_time_detail_;

  autoware::universe_utils::InterProcessPollingSubscriber<Odometry> sub_current_odometry_{
    this, "~/input/odometry"};
  autoware::universe_utils::InterProcessPollingSubscriber<AccelWithCovarianceStamped>
    sub_current_acceleration_{this, "~/input/acceleration"};
  autoware::universe_utils::InterProcessPollingSubscriber<Trajectory> sub_previous_trajectory_{
    this, "~/input/previous_trajectory"};

  Odometry::ConstSharedPtr current_odometry_ptr_;  // current odometry
  AccelWithCovarianceStamped::ConstSharedPtr current_acceleration_ptr_;
  Trajectory::ConstSharedPtr previous_trajectory_ptr_;
  Trajectory::ConstSharedPtr previous_output_ptr_;

  rclcpp::Publisher<autoware::universe_utils::ProcessingTimeDetail>::SharedPtr
    debug_processing_time_detail_pub_;
  mutable std::shared_ptr<autoware::universe_utils::TimeKeeper> time_keeper_{nullptr};
  std::shared_ptr<JerkFilteredSmoother> smoother_{nullptr};
  std::shared_ptr<rclcpp::Time> last_time_{nullptr};

  TrajectoryInterpolatorParams params_;
  OnSetParametersCallbackHandle::SharedPtr set_param_res_;
};

}  // namespace autoware::trajectory_interpolator

#endif  // AUTOWARE__TRAJECTORY_INTERPOLATOR_HPP_
