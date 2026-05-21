#!/usr/bin/env python3

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from math import pi
import cv2
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from scipy.spatial.transform import Rotation
import yaml
from ament_index_python.packages import get_package_share_directory

from dual_amm.dual_amm import DUAL_AMM_Node  # pyright: ignore[reportMissingImports]
from drinks_robot_interface.action import ExecuteDrinkTask  # pyright: ignore[reportMissingImports]
from .perception import ArucoContainerDetector, ContainerDetection, CupDetection, CupDetector
from controller_api import MotionType


class DrinkTaskState(str, Enum):
    IDLE = 'idle'
    GET_ORDER = 'get_order'
    DETECT_CUP = 'detect_cup'
    PICK_CUP = 'pick_cup'
    DETECT_CONTAINER = 'detect_container'
    PICK_CONTAINER = 'pick_container'
    POUR = 'pour'
    FINISH = 'finish'
    FAILED = 'failed'


@dataclass
class DrinkRobotConfig:
    supported_drinks: List[str]
    container_id_to_drink: Dict[int, str]
    aruco_marker_length: float
    hand_shake_pin_left: int
    hand_shake_pin_right: int


class DrinkRobotNode(DUAL_AMM_Node):
    """Dual-arm drink robot with an eye-in-hand perception and motion flow."""


    LEFT_HOME_POSE  = [1.8,  0.785398163, -2.35619449, 1.570796327, -1.570796327, 0.0]
    RIGHT_HOME_POSE = [-1.8, 0.785398163, -2.35619449, 1.570796327, -1.570796327, 0.0]

    # Initial camera observation poses (Cartesian: x, y, z, rx, ry, rz)
    LEFT_CUP_VIEW_POSE        = [0.17220297, -0.70942737,  0.43800946, pi, 0.0, 0.25]
    RIGHT_CONTAINER_VIEW_POSE = [0.28012134,  0.47946701,  0.65280273, pi, 0.0, 2.9]

    # 覆寫父類變數，統一設定所有路徑的 home pose
    _left_home_joint_pose  = LEFT_HOME_POSE
    _right_home_joint_pose = RIGHT_HOME_POSE

    # Cup detection calibration offsets and fixed orientation
    CUP_DETECT_OFFSET = (-0.03, 0.15, -0.01)   # xyz adjustment applied after detection
    CUP_GRASP_ORIENT  = (0.7 * pi, 0.0, 0.23)  # roll, pitch, yaw for cup grasp
    CUP_PLACE_XY      = (-0.1, -0.7)            # xy position to set cup down after task

    # Container handle offset from ArUco marker origin (marker frame, metres)
    # LEE: Maybe the problem is from here
    # 容器把手相對於 ArUco marker 的位移 (單位：公尺)
    HANDLE_OFFSETS = {
        "water": (-0.17, -0.09, -0.21),  # water
        "coffee": (-0.16, -0.1, -0.225),  # coffee
        "tea": (-0.157, -0.108, -0.22)   # tea
    }

    # Container intermediate pose in joint space before approaching handle
    CONTAINER_READY_POSE_J = [-1.1 * pi, pi / 4, -3 / 4 * pi, pi / 2, -pi / 2, 0.0]

    # Left arm holding pose while right arm executes the pour
    LEFT_ARM_POUR_HOLD_POSE = [0.02, -0.501, 0.4, 0.7 * pi, 0.0, -1.0]

    # Pour trajectory waypoints (base XY + 0.02, base Z + 0.01 calibration applied)
    POUR_WAYPOINTS = [
        [ 0.06,  0.395, 0.383, pi / 2, -pi / 2,       3.0],
        [ 0.03,  0.398, 0.423, pi / 2, -3 / 8 * pi,   3.0],
        [-0.02,  0.405, 0.488, pi / 2, -1 / 4 * pi,   3.0],
        [-0.05,  0.412, 0.535, pi / 2, -1 / 8 * pi,   3.0],
        [-0.08,  0.420, 0.525, pi / 2, -1 / 9 * pi,   3.0], 
        [-0.09,  0.420, 0.525, pi / 2, -1 / 14 * pi,  3.0], # 6, 0.5
        [-0.10,  0.420, 0.525, pi / 2, -1 / 17 * pi,  3.0], # 5, 0.4
        [-0.11,  0.420, 0.525, pi / 2, -1 / 21 * pi,  3.0], # 4, 0.5
        [-0.12,  0.420, 0.525, pi / 2, -1 / 40 * pi,  3.0], # 3, 0.5
        [-0.13,  0.423, 0.545, pi / 2, -1 / 155 * pi, 3.0], # 2, 0.4
        [-0.16,  0.425, 0.555, pi / 2, 1 / 100 * pi,   3.0], # 1, 3.0
        
    ]
    # Return path mirrors POUR_WAYPOINTS in reverse; first X is slightly offset to clear the cup
    POUR_RETURN_WAYPOINTS = [
        [-0.08,  0.420, 0.525, pi / 2, -1 / 9 * pi,   3.0],
        [-0.05,  0.412, 0.535, pi / 2, -1 / 8 * pi,   3.0],
        [-0.02,  0.405, 0.488, pi / 2, -1 / 4 * pi,   3.0],
        [ 0.03,  0.398, 0.423, pi / 2, -3 / 8 * pi,   3.0],
        [ 0.06,  0.395, 0.383, pi / 2, -pi / 2,       3.0],
    ]

    def __init__(self):
        super().__init__(name='drink_robot_node')
        self.state = DrinkTaskState.IDLE

        self.config = self._load_config()

        self.container_detector: Optional[ArucoContainerDetector] = None
        self.cup_detector: Optional[CupDetector] = None
        self.current_order: Optional[Dict[str, object]] = None

        self.drink_counters = {"water": 0, "coffee": 0, "tea": 0}
        self.pour_depth_mapping = {1: 5, 2: 5, 3: 6, 4: 6, 5: 7, 6: 8}
        self.pour_wait_mapping = {1: 0.2, 2: 0.6, 3: 0.2, 4: 0.6, 5: 0.2, 6: 0.0}
        

        # Initialize detection output directory
        self.detection_output_dir = Path('/workspaces/AI_Robot_ws/data/drink_robot_detections')
        self.detection_output_dir.mkdir(parents=True, exist_ok=True)
        self.get_logger().info(f'Detection results will be saved to: {self.detection_output_dir}')

        self._action_server = ActionServer(
            self,
            ExecuteDrinkTask,
            '/drinks_robot/execute_task',
            execute_callback=self._execute_action,
            goal_callback=self._goal_callback,
            handle_accepted_callback=self._handle_accepted_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self.cb_me,
        )

        self.get_logger().info(
            f'drink_robot_node ready, supported drinks: {self.config.supported_drinks}'
        )

    def _load_config(self) -> DrinkRobotConfig:
        pkg_path = get_package_share_directory('drinks_robot')
        config_path = os.path.join(pkg_path, 'config', 'drink_robot.yaml')

        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f) or {}

        params = config_data.get('drink_robot', {}).get('ros__parameters', {})
        mapping = {int(k): str(v) for k, v in params.get('container_id_to_drink', {}).items()}

        return DrinkRobotConfig(
            supported_drinks=list(params.get('supported_drinks', ['water', 'coffee', 'tea'])),
            container_id_to_drink=mapping,
            aruco_marker_length=float(params.get('aruco_marker_length', 0.02)),
            hand_shake_pin_left=int(params.get('hand_shake_pin_left', 0)),
            hand_shake_pin_right=int(params.get('hand_shake_pin_right', 0)),
        )

    def _goal_callback(self, goal_request: ExecuteDrinkTask.Goal) -> GoalResponse:
        """Accept or reject incoming goals based on current state."""
        if self.state != DrinkTaskState.IDLE:
            self.get_logger().warning(
                f'Rejecting goal: robot is not IDLE (current state: {self.state})'
            )
            return GoalResponse.REJECT
        if goal_request.drink_type not in self.config.supported_drinks:
            self.get_logger().warning(
                f'Rejecting goal: unsupported drink "{goal_request.drink_type}"'
            )
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle: ServerGoalHandle) -> CancelResponse:
        """Always accept cancel requests; the execute loop checks the flag."""
        self.get_logger().info('Cancel request received, will abort after current motion.')
        return CancelResponse.ACCEPT

    def _handle_accepted_callback(self, goal_handle: ServerGoalHandle) -> None:
        """Start goal execution in the background thread (non-blocking)."""
        import threading
        from rclpy.action.server import GoalEvent
        # Transition ACCEPTED → EXECUTING directly, bypassing notify_execute so
        # the ActionServer's execute_callback is not scheduled a second time.
        goal_handle._update_state(GoalEvent.EXECUTE)
        threading.Thread(
            target=lambda: asyncio.run(self._execute_action(goal_handle)), daemon=True
        ).start()

    def _publish_feedback(
        self,
        goal_handle: ServerGoalHandle,
        state: DrinkTaskState,
        progress: float,
        description: str,
    ) -> None:
        """Publish feedback to the action client."""
        fb = ExecuteDrinkTask.Feedback()
        fb.state = state.value
        fb.progress = float(progress)
        fb.description = description
        goal_handle.publish_feedback(fb)
        self.get_logger().info(f'[feedback] {state.value} ({progress:.0%}) - {description}')

    async def _abort_with_home(
        self,
        goal_handle: ServerGoalHandle,
        result: ExecuteDrinkTask.Result,
    ) -> ExecuteDrinkTask.Result:
        """Helper to abort goal after attempting home reset."""
        self.get_logger().info('Goal cancelled — returning arms to home.')
        await self._handle_cancel_or_failure_async()
        result.success = False
        result.message = 'Order cancelled by client.'
        goal_handle.canceled()
        self.state = DrinkTaskState.IDLE
        return result

    async def _execute_action(self, goal_handle: ServerGoalHandle) -> ExecuteDrinkTask.Result:
        """Execute the drink task as a ROS2 action."""
        drink_type = goal_handle.request.drink_type
        result = ExecuteDrinkTask.Result()

        self.get_logger().info('\n' + '='*60 + f'\nNew action goal: {drink_type}\n' + '='*60)
        self.state = DrinkTaskState.GET_ORDER
        self.current_order = {'drink_type': drink_type}

        try:
            # ==========================================
            # Phase A: Detect + pick cup (left) and container (right) in parallel
            # ==========================================
            left_task = asyncio.create_task(
                self._phase_cup_detect_pick_and_stage(goal_handle)
            )
            right_task = asyncio.create_task(
                self._phase_container_detect_pick_and_stage(goal_handle, drink_type)
            )
            try:
                await asyncio.gather(left_task, right_task)
            except Exception:
                for t in (left_task, right_task):
                    if not t.done():
                        t.cancel()
                await asyncio.gather(left_task, right_task, return_exceptions=True)
                raise

            if goal_handle.is_cancel_requested:
                return await self._abort_with_home(goal_handle, result)

            # ==========================================
            # Phase B: Pour (both arms already staged)
            # ==========================================
            self.state = DrinkTaskState.POUR
            self._publish_feedback(goal_handle, self.state, 0.67, '執行倒水動作...')

            if goal_handle.is_cancel_requested:
                return await self._abort_with_home(goal_handle, result)
            
            self.drink_counters[drink_type] += 1
            current_cup_num = self.drink_counters[drink_type]
            self.get_logger().info(f'目前是第 {current_cup_num} 杯 {drink_type}')

            target_wp_index = self.pour_depth_mapping.get(current_cup_num, 10)
            executed_waypoints = self.POUR_WAYPOINTS[0 : target_wp_index + 1]
            

            # Right arm is already at POUR_WAYPOINTS[0]; move through the remaining waypoints
            for wp in executed_waypoints:
                await self.robot_controller_right.move_to_pose_async(
                    wp, motion_type=MotionType.LINE_T, velocity=0.3
                )


            await self.robot_controller_right.wait_for_arrival_async(self.POUR_WAYPOINTS[target_wp_index])
            # await self.robot_controller_right.wait_for_arrival_async(self.POUR_WAYPOINTS[-1])
            await asyncio.sleep(self.pour_wait_mapping.get(current_cup_num, 1.5))  # Hold pour position for a moment
            # ==========================================
            # Phase 6: Reset to home
            # ==========================================

            # Return container: reverse pour path → lower to holder → release
            # for wp in self.POUR_RETURN_WAYPOINTS:
                # await self.robot_controller_right.move_to_pose_async(wp, motion_type=MotionType.LINE_T, velocity=0.3)

            self._publish_feedback(goal_handle, self.state, 0.84, '任務完成，手臂復位...')
            return_waypoints = list(reversed(executed_waypoints[:-1])) + [self.POUR_WAYPOINTS[0]]

            for wp in return_waypoints:
                await self.robot_controller_right.move_to_pose_async(
                    wp, motion_type=MotionType.LINE_T, velocity=0.3
                )

            self._container_place_base[0] = self._container_place_base[0] - 0.01
            self._container_place_approach_base[0] = self._container_place_approach_base[0] - 0.01
            container_lift     = [*self._container_place_base[:2], self._container_place_base[2] + 0.5, *self._container_place_orient[:3]]
            container_place    = [*self._container_place_base[:2], self._container_place_base[2] +0.005, *self._container_place_orient[:3]]
            container_approach = [*self._container_place_approach_base[:2], self._container_place_approach_base[2], *self._container_place_orient[:3]]
            await self.robot_controller_right.move_to_pose_async(container_lift, velocity=0.8)
            await self.robot_controller_right.wait_for_arrival_async(container_lift)
            

            # self._publish_feedback(goal_handle, self.state, 0.84, '任務完成，手臂復位...')

            # Right arm reached container_lift (above holder) → left cup return and
            # right container place+home can now run in parallel without collision.
            cup_return_lift  = [*self._cup_place_base[:2], self._cup_place_base[2] + 0.20, *self._cup_place_orient[:3]]
            cup_return_place = [*self._cup_place_base[:2], self._cup_place_base[2] + 0.0,  *self._cup_place_orient[:3]]

            async def _right_place_container_and_home():
                await self.robot_controller_right.move_to_pose_async(container_place, motion_type=MotionType.LINE_T, velocity=0.5)
                await self.robot_controller_right.wait_for_arrival_async(container_place, timeout=30.0)
                await asyncio.sleep(1.0)  # Wait for stability before releasing
                await self.gripper_controller_right.open_gripper_async()
                await self.robot_controller_right.move_to_pose_async(container_approach, motion_type=MotionType.LINE_T, velocity=0.8)
                await self.robot_controller_right.wait_for_arrival_async(container_approach)
                await self.robot_controller_right.move_to_pose_async(container_lift, velocity=0.8)
                await self.robot_controller_right.wait_for_arrival_async(container_lift)
                await self.robot_controller_right.move_to_home_async()

            async def _left_place_cup_and_home():
                await self.robot_controller_left.move_to_pose_async(cup_return_lift)
                await self.robot_controller_left.wait_for_arrival_async(cup_return_lift)
                await self.robot_controller_left.move_to_pose_async(cup_return_place, motion_type=MotionType.LINE_T, velocity=0.5)
                await self.robot_controller_left.wait_for_arrival_async(cup_return_place)
                await asyncio.sleep(1.0)  # Wait for stability before releasing
                await self.gripper_controller_left.open_gripper_async()
                await self.robot_controller_left.move_to_pose_async(cup_return_lift, motion_type=MotionType.LINE_T, velocity=0.8)
                await self.robot_controller_left.wait_for_arrival_async(cup_return_lift)
                await self.robot_controller_left.move_to_home_async()

            left_home_task = asyncio.create_task(_left_place_cup_and_home())
            right_home_task = asyncio.create_task(_right_place_container_and_home())
            try:
                await asyncio.gather(left_home_task, right_home_task)
            except Exception:
                for t in (left_home_task, right_home_task):
                    if not t.done():
                        t.cancel()
                await asyncio.gather(left_home_task, right_home_task, return_exceptions=True)
                raise

            self.state = DrinkTaskState.FINISH
            self._publish_feedback(goal_handle, self.state, 1.0, '飲料任務成功完成！')
            result.success = True
            result.message = f'Order completed: {drink_type}'
            goal_handle.succeed()
            self.state = DrinkTaskState.IDLE
            return result

        except Exception as e:
            self.state = DrinkTaskState.FAILED
            self.get_logger().error(f'任務失敗: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            self.get_logger().info('嘗試安全復位...')
            await self._handle_cancel_or_failure_async()
            result.success = False
            result.message = f'Order failed: {e}'
            goal_handle.abort()
            self.state = DrinkTaskState.IDLE
            return result

    def _detect_right_container_for_drink(
        self, drink_type: str
    ) -> Tuple[Optional[ContainerDetection], Optional[np.ndarray], Optional[np.ndarray], Optional[dict]]:
        target_ids = [mid for mid, drink in self.config.container_id_to_drink.items() if drink == drink_type]
        if not target_ids:
            self.get_logger().error(f'no ArUco mapping for drink type: {drink_type}')
            return None, None, None, None

        image = self.realsense_controller_right.get_rgb_image()
        depth_img = self.realsense_controller_right.get_depth_image()
        intrinsics = self.realsense_controller_right.get_camera_intrinsics()
        if image is None or depth_img is None or intrinsics is None:
            self.get_logger().error('right camera data is not ready')
            return None, None, None, None

        if self.container_detector is None:
            self.container_detector = ArucoContainerDetector(
                camera_matrix=self._as_matrix_3x3(intrinsics.get('matrix', [])),
                distortion=self._as_distortion(intrinsics.get('distortion', [])),
                marker_length=self.config.aruco_marker_length,
            )

        detections = self.container_detector.detect(image, allowed_ids=target_ids)
        detection = detections[0] if detections else None

        if detection is None:
            return None, None, depth_img, intrinsics

        try:
            self._draw_and_save_container_detection(image, detection)
        except Exception as e:
            self.get_logger().error(f'failed drawing container detection: {e}')

        # Depth-correct tvec using the depth map
        matrix = self._as_matrix_3x3(intrinsics.get('matrix', []))
        fx, fy = matrix[0, 0], matrix[1, 1]
        cx, cy = matrix[0, 2], matrix[1, 2]

        t_marker2cam = np.array(detection.tvec, dtype=float).reshape(3, 1)
        center_u = int(t_marker2cam[0, 0] * fx / t_marker2cam[2, 0] + cx)
        center_v = int(t_marker2cam[1, 0] * fy / t_marker2cam[2, 0] + cy)

        sampled_depth = self._sample_depth(depth_img, center_u, center_v)
        if sampled_depth is not None and sampled_depth > 0:
            t_marker2cam[2, 0] = sampled_depth
            t_marker2cam[0, 0] = (center_u - cx) * sampled_depth / fx
            t_marker2cam[1, 0] = (center_v - cy) * sampled_depth / fy
            detection = ContainerDetection(
                marker_id=detection.marker_id,
                rvec=detection.rvec,
                tvec=t_marker2cam.flatten().tolist(),
            )

        # Build T_marker2base so callers can reuse it without redoing the math
        R_marker2cam, _ = cv2.Rodrigues(np.array(detection.rvec, dtype=float))
        T_marker2cam = np.eye(4)
        T_marker2cam[:3, :3] = R_marker2cam
        T_marker2cam[:3, 3] = t_marker2cam.flatten()

        T_cam2base = self._camera_to_base_transform('right')
        T_marker2base = T_cam2base @ T_marker2cam

        position = T_marker2base[:3, 3]
        orientation = Rotation.from_matrix(T_marker2base[:3, :3]).as_euler('xyz')
        pos_str = f"x={position[0]:.3f}, y={position[1]:.3f}, z={position[2]:.3f}"
        rot_str = f"rx={orientation[0]:.3f}, ry={orientation[1]:.3f}, rz={orientation[2]:.3f}"
        self.get_logger().info(
            f'Container {detection.marker_id} detected (base frame):\n'
            f'  Pos: {pos_str}\n  Rot: {rot_str}'
        )

        return detection, T_marker2base, depth_img, intrinsics

    def _detect_left_cup(self) -> Tuple[Optional[CupDetection], Optional[np.ndarray], Optional[dict]]:
        if self.cup_detector is None:
            pkg_path = get_package_share_directory('drinks_robot')
            candidate = os.path.join(pkg_path, 'resource', 'yolov11sObb_cup.pt')
            model_path = candidate if os.path.exists(candidate) else None
            if model_path is None:
                self.get_logger().error('cup model not found under resource/yolov11sObb_cup.pt')
                return None, None, None
            self.cup_detector = CupDetector(model_path=model_path)

        image = self.realsense_controller_left.get_rgb_image()
        depth_image = self.realsense_controller_left.get_depth_image()
        intrinsics = self.realsense_controller_left.get_camera_intrinsics()

        if image is None or depth_image is None or intrinsics is None:
            self.get_logger().error('left camera data is not ready')
            return None, None, None

        # Detect cups and save both image and results
        detections = self.cup_detector.detect(image)
        if not detections:
            self.get_logger().error('cup not detected in left camera image (empty detection result)')
            return None, None, None
        
        # Save detection results and visualization (include depth + intrinsics)
        self._save_detection_results(image, detections, depth_image, intrinsics)
        
        return detections[0], depth_image, intrinsics

    def _draw_and_save_container_detection(
        self, image: np.ndarray, detection: ContainerDetection
    ) -> None:
        """Delegate drawing to detector, then save the annotated image."""
        img_out = self.container_detector.draw_detections(image, [detection])
        fname = f"container_{detection.marker_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        out_path = self.detection_output_dir / fname
        cv2.imwrite(str(out_path), img_out)
        self.get_logger().info(f'saved container detection image: {out_path}')

    def _cup_detection_to_base_point(self, detection: CupDetection, depth_image: np.ndarray, intrinsics: dict) -> List[float]:
        center_u, center_v = self._cup_center_pixel(detection)
        depth = self._sample_depth(depth_image, center_u, center_v)
        if depth is None:
            raise RuntimeError('no valid depth near detected cup center')

        matrix = self._as_matrix_3x3(intrinsics.get('matrix', []))
        fx = matrix[0, 0]
        fy = matrix[1, 1]
        cx = matrix[0, 2]
        cy = matrix[1, 2]

        x = (center_u - cx) * depth / fx
        y = (center_v - cy) * depth / fy
        point_camera = np.array([x, y, depth, 1.0], dtype=float)
        point_base = self._camera_to_base_transform('left') @ point_camera
        return point_base[:3].tolist()

    def _camera_to_base_transform(self, side: str) -> np.ndarray:
        T_cam2gripper, T_cam2gripper_offset, _ = self._load_hand_eye_transforms(side)
        robot_controller = self.robot_controller_left if side == 'left' else self.robot_controller_right
        feedback = robot_controller.current_feedback
        if feedback is None or feedback.tool_pose is None:
            raise RuntimeError(f'{side} robot feedback is not ready')
        pose = np.array(feedback.tool_pose, dtype=float)
        T_gripper2base = np.eye(4)
        T_gripper2base[:3, :3] = Rotation.from_euler('xyz', pose[3:6]).as_matrix()
        T_gripper2base[:3, 3] = pose[:3]
        self.get_logger().info(f'{side} hand eye offset: {T_cam2gripper_offset}')

        return T_gripper2base @ T_cam2gripper_offset @ T_cam2gripper

    def _cup_center_pixel(self, detection: CupDetection) -> Tuple[int, int]:
        points = np.array(detection.obb, dtype=float).reshape(-1, 2)
        center = points.mean(axis=0)
        return int(round(center[0])), int(round(center[1]))

    @staticmethod
    def _calculate_obb_direction(
        obb: Optional[np.ndarray]
    ) -> Optional[Dict[str, float]]:
        """
        Calculate the OBB direction.
        Standardizes it to point towards the top of the image (negative y).
        """
        if obb is None:
            return None

        points = np.array(obb, dtype=float).reshape(-1, 2)
        if len(points) < 4:
            return None
            
        # Extract the two edge vectors from the rectangle
        v1 = points[1] - points[0]
        v2 = points[2] - points[1]
        
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        
        # Direction is parallel to the longer edge
        if len1 > len2:
            direction = v1 / len1
        else:
            direction = v2 / len2
            
        # Standardize to point "up" in the image (negative y direction)
        if direction[1] > 0:
            direction = -direction
            
        # Calculate angle (-90 degrees is straight up)
        angle_rad = np.arctan2(direction[1], direction[0])
        angle_deg = np.degrees(angle_rad)
        
        return {
            'dx': float(direction[0]),
            'dy': float(direction[1]),
            'angle_rad': float(angle_rad),
            'angle_deg': float(angle_deg)
        }

    @staticmethod
    def _sample_depth(depth_image: np.ndarray, u: int, v: int, window: int = 5) -> Optional[float]:
        h, w = depth_image.shape[:2]
        half = window // 2
        u0 = max(0, u - half)
        u1 = min(w, u + half + 1)
        v0 = max(0, v - half)
        v1 = min(h, v + half + 1)
        region = depth_image[v0:v1, u0:u1]
        valid = region[region > 0]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    @staticmethod
    async def _set_and_wait_io(robot_controller, pin: int, value: int, timeout: Optional[float] = None):
        await robot_controller.set_io_async(0, 1, pin, float(value))
        if timeout is None:
            await robot_controller.wait_for_io_async(pin, value)
        else:
            await robot_controller.wait_for_io_async(pin, value, timeout)

    async def _phase_cup_detect_pick_and_stage(self, goal_handle: ServerGoalHandle) -> None:
        """Left arm: detect cup → grasp → stage at pour hold pose. Raises on failure."""
        if goal_handle.is_cancel_requested:
            return

        self.state = DrinkTaskState.DETECT_CUP
        self._publish_feedback(goal_handle, self.state, 0.0, '移動左臂至杯子觀測點...')

        left_view_pose = self.LEFT_CUP_VIEW_POSE
        await self.robot_controller_left.move_to_pose_async(left_view_pose)
        await self.robot_controller_left.wait_for_arrival_async(left_view_pose, timeout=30.0)
        await asyncio.sleep(3.0)

        if goal_handle.is_cancel_requested:
            return

        self._publish_feedback(goal_handle, self.state, 0.05, '執行杯子影像偵測 (初步定位)...')
        cup_detection, cv_depth, intrinsics = self._detect_left_cup()
        if cup_detection is None:
            raise RuntimeError('無法在畫面中偵測到杯子')

        cup_base = self._cup_detection_to_base_point(cup_detection, cv_depth, intrinsics)

        # YENLEE: Here debug need
        # --- Move camera to align with cup OBB center ---
        self._publish_feedback(goal_handle, self.state, 0.08, '移動相機對齊杯子中心...')
        T_cam2base = self._camera_to_base_transform('left')
        cam_pos_base = T_cam2base[:3, 3]
        move_x = cup_base[0] - cam_pos_base[0]
        move_y = cup_base[1] - cam_pos_base[1]
        self.get_logger().info(f'Camera alignment shift: dx={move_x:.3f}, dy={move_y:.3f}')

        aligned_view_pose = list(left_view_pose)
        aligned_view_pose[0] += move_x
        aligned_view_pose[1] += move_y
        await self.robot_controller_left.move_to_pose_async(aligned_view_pose)
        await self.robot_controller_left.wait_for_arrival_async(aligned_view_pose, timeout=15.0)
        await asyncio.sleep(1.5)

        self._publish_feedback(goal_handle, self.state, 0.12, '執行杯子影像偵測 (精確定位)...')
        cup_detection, cv_depth, intrinsics = self._detect_left_cup()
        if cup_detection is None:
            raise RuntimeError('無法在畫面中二次偵測到杯子')

        cup_base = self._cup_detection_to_base_point(cup_detection, cv_depth, intrinsics)

        obb_dir = self._calculate_obb_direction(cup_detection.obb)
        if obb_dir is not None:
            matrix = self._as_matrix_3x3(intrinsics.get('matrix', []))
            fx = matrix[0, 0]
            fy = matrix[1, 1]
            center_u, center_v = self._cup_center_pixel(cup_detection)
            depth = self._sample_depth(cv_depth, center_u, center_v)
            if depth is not None:
                cam_vec = np.array([
                    obb_dir['dx'] * depth / fx,
                    obb_dir['dy'] * depth / fy,
                    0.0, 0.0
                ])
                T_cam2base = self._camera_to_base_transform('left')
                base_vec = (T_cam2base @ cam_vec)[:3]
                norm_val = np.linalg.norm(base_vec)
                if norm_val > 0.0:
                    base_vec /= norm_val
                    cup_base[0] += self.CUP_DETECT_OFFSET[0]
                    cup_base[1] += self.CUP_DETECT_OFFSET[1]
                    cup_base[2] += self.CUP_DETECT_OFFSET[2]

        cup_target_pose = [float(cup_base[0]), float(cup_base[1]), float(cup_base[2]), *self.CUP_GRASP_ORIENT]
        self._cup_place_base = [*self.CUP_PLACE_XY, float(cup_base[2])]
        self._cup_place_orient = cup_target_pose[3:6]

        self.state = DrinkTaskState.PICK_CUP
        self._publish_feedback(goal_handle, self.state, 0.17, '執行夾取杯子...')

        if goal_handle.is_cancel_requested:
            return

        await self.gripper_controller_left.open_gripper_async()
        cup_approach = [*cup_target_pose[:2], cup_target_pose[2] + 0.18, *cup_target_pose[3:6]]
        self.get_logger().info(f'moving left arm to cup approach pose: {cup_approach}')
        await self.robot_controller_left.move_to_pose_async(cup_approach)
        await self.robot_controller_left.wait_for_arrival_async(cup_approach)

        if goal_handle.is_cancel_requested:
            return

        cup_grasp = [*cup_target_pose[:2], cup_target_pose[2] + 0.0, *cup_target_pose[3:6]]
        self.get_logger().info(f'moving left arm to cup grasp pose: {cup_grasp}')
        await self._pick_at_pose('left', cup_grasp, 0.3)

        # Stage at pour hold pose so Phase B (pour) can start immediately
        self.get_logger().info(f'staging left arm at pour hold pose: {self.LEFT_ARM_POUR_HOLD_POSE}')
        await asyncio.sleep(2.0)
        await self.robot_controller_left.move_to_pose_async(self.LEFT_ARM_POUR_HOLD_POSE)
        await self.robot_controller_left.wait_for_arrival_async(self.LEFT_ARM_POUR_HOLD_POSE)


    async def _phase_container_detect_pick_and_stage(
        self, goal_handle: ServerGoalHandle, drink_type: str
    ) -> None:
        """Right arm: detect container → grasp handle → stage at first pour waypoint."""
        if goal_handle.is_cancel_requested:
            return

        self.state = DrinkTaskState.DETECT_CONTAINER
        self._publish_feedback(goal_handle, self.state, 0.34, '移動右臂至容器觀測點...')

        right_view_pose = self.RIGHT_CONTAINER_VIEW_POSE
        await self.robot_controller_right.move_to_pose_async(right_view_pose)
        await self.robot_controller_right.wait_for_arrival_async(right_view_pose, timeout=30.0)
        await asyncio.sleep(3.0)

        if goal_handle.is_cancel_requested:
            return

        self._publish_feedback(goal_handle, self.state, 0.36, '執行容器影像偵測 (初步定位)...')
        container_detection, T_marker2base, _cv_depth_right, _intrinsics_right = (
            self._detect_right_container_for_drink(drink_type)
        )
        if container_detection is None or T_marker2base is None:
            raise RuntimeError(f'無法找到對應的飲料容器: {drink_type}')

        T_handle2marker = np.eye(4)
        T_handle2marker[:3, 3] = self.HANDLE_OFFSETS[drink_type]
        R_handle2marker = np.array([
            [np.cos(pi/2), 0, np.sin(pi/2)],
            [0, 1, 0],
            [-np.sin(pi/2), 0, np.cos(pi/2)]
        ], dtype=float) @ np.array([
            [np.cos(pi), -np.sin(pi), 0],
            [np.sin(pi), np.cos(pi), 0],
            [0, 0, 1]
        ], dtype=float)
        T_handle2marker[:3, :3] = R_handle2marker

        T_handle2marker_approach = T_handle2marker.copy()
        T_handle2marker_approach[0, 3] -= 0.05

        T_handle2base = T_marker2base @ T_handle2marker
        T_handle2base_approach = T_marker2base @ T_handle2marker_approach
        R_handle2base = T_handle2base[:3, :3]
        handle_base = T_handle2base[:3, 3].tolist()
        handle_approach_base = T_handle2base_approach[:3, 3].tolist()
        self.get_logger().info(f'Handle base after matrix offset: {handle_base}')
        self.get_logger().info(f'Handle approach base after matrix offset: {handle_approach_base}')

        handle_grasp_orient = Rotation.from_matrix(R_handle2base).as_euler('xyz').tolist()
        self._container_place_base = handle_base
        self._container_place_approach_base = handle_approach_base
        self._container_place_orient = handle_grasp_orient
        self.get_logger().info(f'Handle grasp orientation (ArUco Y-axis +90°): {handle_grasp_orient}')

        self.state = DrinkTaskState.PICK_CONTAINER
        self._publish_feedback(goal_handle, self.state, 0.50, '執行夾取容器...')

        if goal_handle.is_cancel_requested:
            return

        await self.robot_controller_right.move_to_pose_async(
            self.CONTAINER_READY_POSE_J, motion_type=MotionType.PTP_J
        )

        await self.gripper_controller_right.open_gripper_async()
        handle_approach = [*handle_approach_base[:2], handle_approach_base[2], *handle_grasp_orient[:3]]
        self.get_logger().info(f'moving right arm to handle approach pose: {handle_approach}')
        await self.robot_controller_right.move_to_pose_async(handle_approach, velocity=1.0)
        await self.robot_controller_right.wait_for_arrival_async(handle_approach)

        if goal_handle.is_cancel_requested:
            return

        handle_grasp = [*handle_base[:2], handle_base[2], *handle_grasp_orient[:3]]
        self.get_logger().info(f'moving right arm to handle grasp pose: {handle_grasp}')
        await self.robot_controller_right.move_to_pose_async(
            handle_grasp, motion_type=MotionType.LINE_T, velocity=0.5
        )
        await self.robot_controller_right.wait_for_arrival_async(handle_grasp)
        await self.gripper_controller_right.set_gripper_state_async(
            position=int(1.0 * 255 + 0.5),
            wait_time=2,
        )
        handle_grasp[2] += 0.5  # Lift up slightly after grasping
        self.get_logger().info(f"Moving to lifted pose: {handle_grasp}")
        await self.robot_controller_right.move_to_pose_async(handle_grasp, motion_type=MotionType.LINE_T, velocity=0.5)


        # Stage at first pour waypoint so Phase B (pour) can start immediately
        self.get_logger().info(
            f'staging right arm at first pour waypoint: {self.POUR_WAYPOINTS[0]}'
        )



        await self.robot_controller_right.move_to_pose_async(
            self.POUR_WAYPOINTS[0], motion_type=MotionType.LINE_T, velocity=1.0
        )
        await self.robot_controller_right.wait_for_arrival_async(self.POUR_WAYPOINTS[0])

    # NOTE: 目前只有左手使用。右手夾 container 握把需要 LINE_T 軌跡
    # （PTP_T 曲線會撞到容器），所以走 Phase 3 / 4 內的 inline 實作。
    async def _pick_at_pose(
        self,
        arm: str,
        object_pose: List[float],
        opening: float,
    ) -> Tuple[bool, int]:
        robot_controller = self.robot_controller_left if arm == 'left' else self.robot_controller_right
        gripper_controller = self.gripper_controller_left if arm == 'left' else self.gripper_controller_right
        hand_shake_pin = self.config.hand_shake_pin_left if arm == 'left' else self.config.hand_shake_pin_right
        try:
            await self._set_and_wait_io(robot_controller, hand_shake_pin, 0)

            self.get_logger().info(f'moving to target pose: {object_pose}')
            await robot_controller.move_to_pose_async(object_pose)

            # 等待到達目標位置
            try:
                await robot_controller.wait_for_arrival_async(object_pose, timeout=15.0)
            except Exception as e:
                self.get_logger().warn(f'等待夾取位置異常: {e}')

            await self._set_and_wait_io(robot_controller, hand_shake_pin, 1, timeout=15)
            self.get_logger().info('target pose arrival confirmed')

            self.get_logger().info(f'closing gripper (opening={opening})')
            await gripper_controller.set_gripper_state_async(
                position=int(opening * 255 + 0.5),
                wait_time=2,
            )

            await self._set_and_wait_io(robot_controller, hand_shake_pin, 0)
            await self._set_and_wait_io(robot_controller, hand_shake_pin, 1)
            await self._set_and_wait_io(robot_controller, hand_shake_pin, 0)

            status = await gripper_controller.get_gripper_status_async()
            self.get_logger().info('grasp sequence complete')
            return status.ok, status.status_code
        except Exception as e:
            self.get_logger().error(f'grasp sequence failed: {e}')
            await robot_controller.move_to_home_async()
            raise

    @staticmethod
    def _as_matrix_3x3(values: List[float]):
        if values is None:
            raise ValueError('camera matrix must have 9 values')

        # Accept several common formats:
        # - a flat list/iterable of 9 numbers
        # - a nested 3x3 list/array
        # - a numpy array with shape (3,3)
        arr = np.array(values, dtype=float)

        # flat of length 9
        if arr.size == 9 and arr.ndim == 1:
            return arr.reshape(3, 3)

        # already shaped 3x3
        if arr.shape == (3, 3):
            return arr.astype(float)

        # nested lists like [[..],[..],[..]] will be caught above by shape
        raise ValueError('camera matrix must have 9 values')

    @staticmethod
    def _as_distortion(values: List[float]):
        if values is None or len(values) == 0:
            return np.zeros(5, dtype=float)
        return np.array(values, dtype=float)

    def _save_detection_results(
        self,
        image: np.ndarray,
        detections: List[CupDetection],
        depth_image: Optional[np.ndarray] = None,
        intrinsics: Optional[dict] = None,
    ):
        """Save detection image and results to disk.

        If `depth_image` and `intrinsics` are provided, compute and save
        the pixel center (u,v) and the corresponding 3D point in base frame
        for each detection.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # millisecond precision

        # Save raw detection image
        image_filename = f'cup_detection_raw_{timestamp}.png'
        image_path = self.detection_output_dir / image_filename
        cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        # Create visualization image with detection boxes
        vis_image = self.cup_detector._draw_detections(image, detections)
        vis_filename = f'cup_detection_vis_{timestamp}.png'
        vis_path = self.detection_output_dir / vis_filename
        cv2.imwrite(str(vis_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

        # Build detection entries with optional uv and 3D base points
        detection_entries = []
        for det in detections:
            entry = {
                'cup_id': det.cup_id,
                'obb': list(det.obb),
            }

            try:
                obb_dir = self._calculate_obb_direction(det.obb)
                entry['obb_direction'] = obb_dir
            except Exception as e:
                entry['obb_direction'] = None
                self.get_logger().warn(f'failed to compute OBB direction for cup {det.cup_id}: {e}')

            try:
                u, v = self._cup_center_pixel(det)
                entry['center_uv'] = [int(u), int(v)]
            except Exception:
                entry['center_uv'] = None

            if depth_image is not None and intrinsics is not None:
                try:
                    point_base = self._cup_detection_to_base_point(det, depth_image, intrinsics)
                    entry['center_3d_base'] = [float(x) for x in point_base]
                except Exception as e:
                    entry['center_3d_base'] = None
                    self.get_logger().warn(f'failed to compute 3D center for cup {det.cup_id}: {e}')
            else:
                entry['center_3d_base'] = None

            detection_entries.append(entry)

        # Save detection results as JSON
        detection_data = {
            'timestamp': timestamp,
            'image_width': image.shape[1],
            'image_height': image.shape[0],
            'num_cups_detected': len(detections),
            'detections': detection_entries,
            'raw_image_path': image_filename,
            'visualization_image_path': vis_filename,
        }

        json_filename = f'cup_detection_results_{timestamp}.json'
        json_path = self.detection_output_dir / json_filename
        with open(json_path, 'w') as f:
            json.dump(detection_data, f, indent=2)

        # Log summary for quick inspection
        for entry in detection_entries:
            dir_str = 'None'
            if entry.get('obb_direction') is not None:
                dir_str = f"{entry['obb_direction']['angle_deg']:.1f}deg"
            
            self.get_logger().info(
                f"cup {entry.get('cup_id')} uv={entry.get('center_uv')} 3d_base={entry.get('center_3d_base')} dir={dir_str}"
            )

        self.get_logger().info(
            f'Detection results saved: {json_filename} ({len(detections)} cups detected)'
        )



def main(args=None):
    rclpy.init(args=args)
    node = DrinkRobotNode()

    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)

    try:
        node.get_logger().info('drink_robot_node spinning')
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
