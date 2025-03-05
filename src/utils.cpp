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

#include "autoware/trajectory_interpolator/utils.hpp"

#include "autoware/trajectory/interpolator/akima_spline.hpp"
#include "autoware/trajectory/interpolator/interpolator.hpp"
#include "autoware/trajectory/pose.hpp"
#include "autoware/trajectory_interpolator/trajectory_interpolator_structs.hpp"

#include <autoware/motion_utils/trajectory/trajectory.hpp>
#include <autoware_utils/geometry/geometry.hpp>
#include <autoware_utils/math/normalization.hpp>
#include <rclcpp/logging.hpp>

#include <autoware_planning_msgs/msg/detail/trajectory_point__struct.hpp>

#include <cmath>
#include <iostream>

namespace autoware::trajectory_interpolator::utils
{
using autoware::trajectory::interpolator::AkimaSpline;
using InterpolationTrajectory = autoware::trajectory::Trajectory<TrajectoryPoint>;

rclcpp::Logger get_logger()
{
  return rclcpp::get_logger("trajectory_interpolator");
}

void smooth_trajectory_with_elastic_band(
  TrajectoryPoints & traj_points, const Odometry & current_odometry,
  const std::shared_ptr<EBPathSmoother> & eb_path_smoother_ptr)
{
  if (!eb_path_smoother_ptr) {
    RCLCPP_ERROR(get_logger(), "Elastic band path smoother is not initialized");
    return;
  }
  traj_points = eb_path_smoother_ptr->smoothTrajectory(traj_points, current_odometry.pose.pose);
  eb_path_smoother_ptr->resetPreviousData();
}

void remove_invalid_points(TrajectoryPoints & input_trajectory)
{
  if (input_trajectory.size() < 2) {
    RCLCPP_ERROR(get_logger(), "No enough points in trajectory after overlap points removal");
    return;
  }
  utils::remove_close_proximity_points(input_trajectory, 1E-2);
  const bool is_driving_forward = true;
  autoware::motion_utils::insertOrientation(input_trajectory, is_driving_forward);

  autoware::motion_utils::removeFirstInvalidOrientationPoints(input_trajectory);
  size_t previous_size{input_trajectory.size()};
  do {
    previous_size = input_trajectory.size();
    // Set the azimuth orientation to the next point at each point
    autoware::motion_utils::insertOrientation(input_trajectory, is_driving_forward);
    // Use azimuth orientation to remove points in reverse order
    autoware::motion_utils::removeFirstInvalidOrientationPoints(input_trajectory);
  } while (previous_size != input_trajectory.size());
}

void remove_close_proximity_points(TrajectoryPoints & input_trajectory_array, const double min_dist)
{
  if (std::size(input_trajectory_array) < 2) {
    return;
  }

  input_trajectory_array.erase(
    std::remove_if(
      std::next(input_trajectory_array.begin()),  // Start from second element
      input_trajectory_array.end(),
      [&](const TrajectoryPoint & point) {
        const auto prev_it = std::prev(&point);
        const auto dist = autoware_utils::calc_distance2d(point, *prev_it);
        return dist < min_dist;
      }),
    input_trajectory_array.end());
}

void clamp_velocities(
  TrajectoryPoints & input_trajectory_array, float min_velocity, float min_acceleration)
{
  std::for_each(
    input_trajectory_array.begin(), input_trajectory_array.end(),
    [min_velocity, min_acceleration](TrajectoryPoint & point) {
      point.longitudinal_velocity_mps = std::max(point.longitudinal_velocity_mps, min_velocity);
      point.acceleration_mps2 = std::max(point.acceleration_mps2, min_acceleration);
    });
}

void set_max_velocity(TrajectoryPoints & input_trajectory_array, const float max_velocity)
{
  std::for_each(
    input_trajectory_array.begin(), input_trajectory_array.end(),
    [max_velocity](TrajectoryPoint & point) {
      point.longitudinal_velocity_mps = std::min(point.longitudinal_velocity_mps, max_velocity);
    });
}

void filter_velocity(
  TrajectoryPoints & input_trajectory, const InitialMotion & initial_motion,
  const TrajectoryInterpolatorParams & params,
  const std::shared_ptr<JerkFilteredSmoother> & smoother, const Odometry & current_odometry)
{
  if (!smoother) {
    RCLCPP_ERROR(get_logger(), "JerkFilteredSmoother is not initialized");
    return;
  }
  // Lateral acceleration limit
  const auto & nearest_dist_threshold = params.nearest_dist_threshold_m;
  const auto & nearest_yaw_threshold = params.nearest_yaw_threshold_rad;
  const auto & initial_motion_speed = initial_motion.speed_mps;
  const auto & initial_motion_acc = initial_motion.acc_mps2;

  constexpr bool enable_smooth_limit = true;
  constexpr bool use_resampling = true;

  input_trajectory = smoother->applyLateralAccelerationFilter(
    input_trajectory, initial_motion_speed, initial_motion_acc, enable_smooth_limit,
    use_resampling);

  // Steering angle rate limit (Note: set use_resample = false since it is resampled above)
  input_trajectory = smoother->applySteeringRateLimit(input_trajectory, false);
  // Resample trajectory with ego-velocity based interval distance

  input_trajectory = smoother->resampleTrajectory(
    input_trajectory, initial_motion_speed, current_odometry.pose.pose, nearest_dist_threshold,
    nearest_yaw_threshold);

  if (input_trajectory.size() < 2) {
    return;
  }

  const size_t traj_closest = autoware::motion_utils::findFirstNearestIndexWithSoftConstraints(
    input_trajectory, current_odometry.pose.pose, nearest_dist_threshold, nearest_yaw_threshold);

  // // Clip trajectory from closest point
  TrajectoryPoints clipped;
  clipped.insert(
    clipped.end(),
    input_trajectory.begin() + static_cast<TrajectoryPoints::difference_type>(traj_closest),
    input_trajectory.end());
  input_trajectory = clipped;

  std::vector<TrajectoryPoints> debug_trajectories;
  if (!smoother->apply(
        initial_motion_speed, initial_motion_acc, input_trajectory, input_trajectory,
        debug_trajectories, false)) {
    RCLCPP_WARN(get_logger(), "Fail to solve optimization.");
  }
}

bool validate_pose(const geometry_msgs::msg::Pose & pose)
{
  return std::isfinite(pose.position.x) && std::isfinite(pose.position.y) &&
         std::isfinite(pose.position.z) && std::isfinite(pose.orientation.x) &&
         std::isfinite(pose.orientation.y) && std::isfinite(pose.orientation.z) &&
         std::isfinite(pose.orientation.w) && !std::isnan(pose.position.x) &&
         !std::isnan(pose.position.y) && !std::isnan(pose.position.z) &&
         !std::isnan(pose.orientation.x) && !std::isnan(pose.orientation.y) &&
         !std::isnan(pose.orientation.z) && !std::isnan(pose.orientation.w);
}

void apply_spline(TrajectoryPoints & traj_points, const TrajectoryInterpolatorParams & params)
{
  std::optional<InterpolationTrajectory> interpolation_trajectory_util =
    InterpolationTrajectory::Builder{}
      .set_xy_interpolator<AkimaSpline>()  // Set interpolator for x-y plane
      .build(traj_points);
  if (!interpolation_trajectory_util) {
    RCLCPP_WARN(get_logger(), "Failed to build interpolation trajectory");
    return;
  }
  interpolation_trajectory_util->align_orientation_with_trajectory_direction();
  TrajectoryPoints output_points;
  output_points.reserve(traj_points.size());

  const auto ds = params.spline_interpolation_resolution_m;

  for (auto s = 0.0; s <= interpolation_trajectory_util->length(); s += ds) {
    auto p = interpolation_trajectory_util->compute(s);
    if (!validate_pose(p.pose)) {
      continue;
    }
    output_points.push_back(p);
  }

  if (output_points.size() < 2) {
    RCLCPP_WARN(get_logger(), "Not enough points in trajectory after akima spline interpolation");
    return;
  }
  constexpr double epsilon{1e-2};
  auto last_interpolated_point = output_points.back();
  auto & original_trajectory_last_point = traj_points.back();

  if (!validate_pose(original_trajectory_last_point.pose)) {
    RCLCPP_WARN(get_logger(), "Last point in original trajectory is invalid. Removing last point");
    traj_points = output_points;
    return;
  }

  auto d = autoware_utils::calc_distance2d(
    last_interpolated_point.pose.position, original_trajectory_last_point.pose.position);
  if (d > epsilon) {
    output_points.push_back(original_trajectory_last_point);
  };
  traj_points = output_points;
}

void interpolate_trajectory(
  TrajectoryPoints & traj_points, const Odometry & current_odometry,
  const AccelWithCovarianceStamped & current_acceleration,
  const TrajectoryInterpolatorParams & params,
  const std::shared_ptr<JerkFilteredSmoother> & jerk_filtered_smoother,
  const std::shared_ptr<EBPathSmoother> & eb_path_smoother_ptr)
{
  // Remove overlap points and wrong orientation points
  if (params.fix_invalid_points) {
    remove_invalid_points(traj_points);
  }

  if (traj_points.size() < 2) {
    RCLCPP_ERROR(get_logger(), "No enough points in trajectory after overlap points removal");
    return;
  }

  const double & target_pull_out_speed_mps = params.target_pull_out_speed_mps;
  const double & target_pull_out_acc_mps2 = params.target_pull_out_acc_mps2;
  const double & max_speed_mps = params.max_speed_mps;

  const auto current_speed = current_odometry.twist.twist.linear.x;
  const auto current_linear_acceleration = current_acceleration.accel.accel.linear.x;
  auto initial_motion_speed =
    (current_speed > target_pull_out_speed_mps) ? current_speed : target_pull_out_speed_mps;
  auto initial_motion_acc = (current_speed > target_pull_out_speed_mps)
                              ? current_linear_acceleration
                              : target_pull_out_acc_mps2;
  InitialMotion initial_motion{initial_motion_speed, initial_motion_acc};

  // Set engage speed and acceleration
  if (current_speed < target_pull_out_speed_mps) {
    clamp_velocities(
      traj_points, static_cast<float>(initial_motion_speed),
      static_cast<float>(initial_motion_acc));
  }
  // Limit ego speed
  if (params.limit_velocity) {
    set_max_velocity(traj_points, static_cast<float>(max_speed_mps));
  }

  // Smooth velocity profile
  if (params.smooth_velocities) {
    filter_velocity(traj_points, initial_motion, params, jerk_filtered_smoother, current_odometry);
  }
  // Apply spline to smooth the trajectory
  if (params.use_akima_spline_interpolation) {
    apply_spline(traj_points, params);
  }
  // Use elastic band to smooth the trajectory
  if (params.smooth_trajectories) {
    smooth_trajectory_with_elastic_band(traj_points, current_odometry, eb_path_smoother_ptr);
  }
  // Recalculate timestamps
  motion_utils::calculate_time_from_start(traj_points, current_odometry.pose.pose.position);

  if (traj_points.size() < 2) {
    RCLCPP_ERROR(get_logger(), "Not enough points in trajectory after overlap points removal");
    return;
  }
}

void add_ego_state_to_trajectory(
  TrajectoryPoints & traj_points, const Odometry & current_odometry,
  const TrajectoryInterpolatorParams & params)
{
  TrajectoryPoint ego_state;
  ego_state.pose = current_odometry.pose.pose;
  ego_state.longitudinal_velocity_mps = current_odometry.twist.twist.linear.x;

  if (traj_points.empty()) {
    traj_points.push_back(ego_state);
    return;
  }
  const auto & last_point = traj_points.back();
  const auto yaw_diff = std::abs(
    autoware_utils::normalize_degree(ego_state.pose.orientation.z - last_point.pose.orientation.z));
  const auto distance = autoware_utils::calc_distance2d(last_point, ego_state);
  constexpr double epsilon{1e-2};
  const bool is_change_small = distance < epsilon && yaw_diff < epsilon;
  if (is_change_small) {
    return;
  }

  const bool is_change_large =
    distance > params.nearest_dist_threshold_m || yaw_diff > params.nearest_yaw_threshold_rad;
  if (is_change_large) {
    traj_points = {ego_state};
    return;
  }

  traj_points.push_back(ego_state);

  size_t clip_idx = 0;
  double accumulated_length = 0.0;
  for (size_t i = traj_points.size() - 1; i > 0; i--) {
    accumulated_length += autoware_utils::calc_distance2d(traj_points.at(i - 1), traj_points.at(i));
    if (accumulated_length > params.backward_path_extension_m) {
      clip_idx = i;
      break;
    }
  }
  traj_points.erase(traj_points.begin(), traj_points.begin() + static_cast<int>(clip_idx));
}

void expand_trajectory_with_ego_history(
  TrajectoryPoints & traj_points, const TrajectoryPoints & ego_history_points)
{
  if (ego_history_points.empty() || traj_points.empty()) {
    return;
  }

  traj_points.insert(traj_points.begin(), ego_history_points.begin(), ego_history_points.end());
}
}  // namespace autoware::trajectory_interpolator::utils
