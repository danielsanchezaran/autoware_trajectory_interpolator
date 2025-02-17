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

#include "autoware/trajectory_interpolator/trajectory_interpolator_structs.hpp"

#include <autoware/universe_utils/geometry/geometry.hpp>
#include <rclcpp/duration.hpp>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>

namespace autoware::trajectory_interpolator::utils
{

rclcpp::Logger get_logger()
{
  return rclcpp::get_logger("trajectory_interpolator");
}

void resample_with_time(TrajectoryPoints & input_trajectory, const double dt)
{
  if (dt < 1e-2) {
    RCLCPP_ERROR(get_logger(), "dt is too low for resampling with time");
    return;
  }

  if (input_trajectory.size() < 2) {
    RCLCPP_ERROR(get_logger(), "No enough points in for time resampling");
    return;
  }

  auto lerp = [](const double val1, const double val2, const double ratio) -> double {
    return ratio * (val2 - val1) + val1;
  };

  auto n_samples = (input_trajectory.back().time_from_start.sec +
                    input_trajectory.back().time_from_start.nanosec * 1e-9) /
                   dt;

  TrajectoryPoints resampled_points{input_trajectory.front()};
  resampled_points.reserve(static_cast<size_t>(n_samples));

  auto curr_time = 0.0;
  std::cerr << "IM here\n";
  for (auto curr_itr = input_trajectory.begin(); curr_itr < input_trajectory.end(); curr_itr++) {
    const auto & last_point = resampled_points.back();
    auto distance_between_points_m =
      autoware::universe_utils::calcDistance2d(last_point.pose.position, curr_itr->pose.position);
    const auto velocity_mps = curr_itr->longitudinal_velocity_mps;

    if (velocity_mps * dt > distance_between_points_m) continue;

    auto d = 0.0;
    do {
      d += velocity_mps * dt;
      auto ratio = d / distance_between_points_m;
      auto pose = lerp_by_pose(last_point.pose, curr_itr->pose, ratio);
      auto p = *curr_itr;

      p.pose = pose;
      p.time_from_start.sec = static_cast<int>(curr_time);
      p.time_from_start.nanosec = static_cast<int>((curr_time - static_cast<int>(curr_time)) * 1e9);
      p.heading_rate_rps = lerp(last_point.heading_rate_rps, curr_itr->heading_rate_rps, ratio);
      resampled_points.push_back(p);
      curr_time += dt;
    } while (d < distance_between_points_m + std::numeric_limits<double>::epsilon());
    if (d > distance_between_points_m) curr_itr++;
  }
  input_trajectory = resampled_points;
}

// apply linear interpolation to position
geometry_msgs::msg::Pose lerp_by_pose(
  const geometry_msgs::msg::Pose & p1, const geometry_msgs::msg::Pose & p2, const float t)
{
  tf2::Transform tf_transform1, tf_transform2;
  tf2::fromMsg(p1, tf_transform1);
  tf2::fromMsg(p2, tf_transform2);
  const auto & tf_point = tf2::lerp(tf_transform1.getOrigin(), tf_transform2.getOrigin(), t);

  geometry_msgs::msg::Pose pose;
  pose.position.x = tf_point.getX();
  pose.position.y = tf_point.getY();
  pose.position.z = tf_point.getZ();
  pose.orientation = p1.orientation;
  return pose;
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
        const auto dist = autoware::universe_utils::calcDistance2d(point, *prev_it);
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

void interpolate_trajectory(
  TrajectoryPoints & traj_points, const Odometry & current_odometry,
  const AccelWithCovarianceStamped & current_acceleration,
  const TrajectoryInterpolatorParams & params,
  const std::shared_ptr<JerkFilteredSmoother> & smoother)
{
  // Remove overlap points and wrong orientation points
  utils::remove_invalid_points(traj_points);

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
  // limit ego speed
  set_max_velocity(traj_points, static_cast<float>(max_speed_mps));

  // Smooth velocity profile
  filter_velocity(traj_points, initial_motion, params, smoother, current_odometry);
  // Recalculate timestamps
  motion_utils::calculate_time_from_start(traj_points, current_odometry.pose.pose.position);
  std::for_each(traj_points.begin(), traj_points.end(), [](const auto & a) {
    auto t = static_cast<double>(a.time_from_start.sec) +
             static_cast<double>(a.time_from_start.nanosec) * 1e-9;
    std::cerr << "time from start " << t << "\n";
  });
  // resample_with_time(traj_points, 0.1);

  if (traj_points.size() < 2) {
    RCLCPP_ERROR(get_logger(), "No enough points in trajectory after overlap points removal");
    return;
  }
}

}  // namespace autoware::trajectory_interpolator::utils
