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

#ifndef AUTOWARE__TRAJECTORY_VELOCITY_OPTIMIZER_HPP_
#define AUTOWARE__TRAJECTORY_VELOCITY_OPTIMIZER_HPP_
#include "autoware/trajectory_optimizer/trajectory_optimizer_plugins/trajectory_optimizer_plugin_base.hpp"
#include "autoware/velocity_smoother/smoother/jerk_filtered_smoother.hpp"

#include <autoware_utils/system/time_keeper.hpp>
#include <rclcpp/rclcpp.hpp>

#include <autoware_planning_msgs/msg/trajectory.hpp>
#include <autoware_planning_msgs/msg/trajectory_point.hpp>

namespace autoware::trajectory_optimizer::plugin
{
using autoware_planning_msgs::msg::TrajectoryPoint;
using TrajectoryPoints = std::vector<TrajectoryPoint>;
using autoware::velocity_smoother::JerkFilteredSmoother;

class TrajectoryVelocityOptimizer : public TrajectoryOptimizerPluginBase
{
public:
  TrajectoryVelocityOptimizer(
    const std::string name, rclcpp::Node * node_ptr,
    const std::shared_ptr<autoware_utils_debug::TimeKeeper> time_keeper,
    const TrajectoryOptimizerParams & params);

  void set_up_velocity_smoother(
    rclcpp::Node * node_ptr, const std::shared_ptr<autoware_utils_debug::TimeKeeper> time_keeper);
  void optimize_trajectory(
    TrajectoryPoints & traj_points, const TrajectoryOptimizerParams & params) override;
  void set_up_params() override;
  rcl_interfaces::msg::SetParametersResult on_parameter(
    const std::vector<rclcpp::Parameter> & parameters) override;

private:
  std::shared_ptr<JerkFilteredSmoother> jerk_filtered_smoother_{nullptr};
};
}  // namespace autoware::trajectory_optimizer::plugin

#endif  // AUTOWARE__TRAJECTORY_VELOCITY_OPTIMIZER_HPP_
