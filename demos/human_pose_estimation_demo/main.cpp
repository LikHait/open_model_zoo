// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the Inference Engine Human Pose Estimation demo application
* \file human_pose_estimation_demo/main.cpp
* \example human_pose_estimation_demo/main.cpp
*/

#include <vector>
#include <ctime>

#include <inference_engine.hpp>

#include <samples/ocv_common.hpp>

#include "human_pose_estimation_demo.hpp"
#include "human_pose_estimator.hpp"
#include "render_human_pose.hpp"

using namespace InferenceEngine;
using namespace human_pose_estimation;

bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }

    std::cout << "Parsing input parameters" << std::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    return true;
}

int main(int argc, char* argv[]) {
    try {
        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return EXIT_SUCCESS;
        }

        HumanPoseEstimator estimator(FLAGS_m, FLAGS_d, FLAGS_pc);
        cv::VideoCapture cap;
        if (!(FLAGS_i == "cam" ? cap.open(0) : cap.open(FLAGS_i))) {
            throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
        }

        int delay = 33;
        double inferenceTime = 0.0;
        cv::Mat image;
        if (!cap.read(image)) {
            throw std::logic_error("Failed to get frame from cv::VideoCapture");
        }
        estimator.estimate(image);  // Do not measure network reshape, if it happened

        std::cout << "To close the application, press 'CTRL+C' here";
        if (!FLAGS_no_show) {
            std::cout << " or switch to the output window and press ESC key" << std::endl;
            std::cout << "To pause execution, switch to the output window and press 'p' key" << std::endl;
        }
        std::cout << std::endl;

        do {
            double t1 = static_cast<double>(cv::getTickCount());
            std::vector<HumanPose> poses = estimator.estimate(image);
            double t2 = static_cast<double>(cv::getTickCount());
            if (inferenceTime == 0) {
                inferenceTime = (t2 - t1) / cv::getTickFrequency() * 1000;
            } else {
                inferenceTime = inferenceTime * 0.95 + 0.05 * (t2 - t1) / cv::getTickFrequency() * 1000;
            }
            if (FLAGS_r) {
                for (HumanPose const& pose : poses) {
                    std::stringstream rawPose;
                    rawPose << std::fixed << std::setprecision(0);
                    for (auto const& keypoint : pose.keypoints) {
                        rawPose << keypoint.x << "," << keypoint.y << " ";
                    }
                    rawPose << pose.score;
                    std::cout << rawPose.str() << std::endl;
                }
            }

            if (FLAGS_no_show) {
                continue;
            }

            std::time_t cur_time = std::time(nullptr);
            const std::string filename_r = "./hand/" + std::to_string(FLAGS_num) + std::to_string(cur_time) + "_r.png";
            const std::string filename_l = "./hand/" + std::to_string(FLAGS_num) + std::to_string(cur_time) + "_l.png";

            auto height = image.rows;
            auto width = image.cols;

            for (const auto &pose : poses) {
                if (pose.keypoints[4].x >= 0 and pose.keypoints[7].x >= 0 and pose.keypoints[3].x >= 0 and
                    pose.keypoints[6].x >= 0 and pose.keypoints[4].y >= 0 and pose.keypoints[7].y >= 0 and
                    pose.keypoints[3].y >= 0 and pose.keypoints[6].y >= 0) {
                    auto right_hand = pose.keypoints[4];
                    auto left_hand = pose.keypoints[7];
                    auto right_cubit = pose.keypoints[3];
                    auto left_cubit = pose.keypoints[6];

                    float right_hand_bbox_size = cv::norm(right_hand - right_cubit);
                    float left_hand_bbox_size = cv::norm(left_hand - left_cubit);

                    float right_hand_bbox_x_min = right_hand.x - right_hand_bbox_size;
                    float right_hand_bbox_y_min = right_hand.y - right_hand_bbox_size;

                    float left_hand_bbox_x_min = left_hand.x - left_hand_bbox_size;
                    float left_hand_bbox_y_min = left_hand.y - left_hand_bbox_size;

                    left_hand_bbox_x_min = (left_hand_bbox_x_min >= 0) ? left_hand_bbox_x_min : 0;
                    left_hand_bbox_y_min = (left_hand_bbox_y_min >= 0) ? left_hand_bbox_y_min : 0;

                    float left_hand_bbox_x_max = (left_hand_bbox_x_min + left_hand_bbox_size * 2 <= width)
                        ? left_hand_bbox_x_min + left_hand_bbox_size * 2
                        : width;
                    float left_hand_bbox_y_max = (left_hand_bbox_y_min + left_hand_bbox_size * 2 <= height)
                        ? left_hand_bbox_y_min + left_hand_bbox_size * 2
                        : height;

                    cv::Mat ROI_for_crop(image, cv::Rect(static_cast<uint32_t>(left_hand_bbox_x_min),
                        static_cast<uint32_t>(left_hand_bbox_y_min),
                        static_cast<uint32_t>(left_hand_bbox_x_max - left_hand_bbox_x_min),
                        static_cast<uint32_t>(left_hand_bbox_y_max - left_hand_bbox_y_min)));

                    cv::Mat cropped_image;
                    ROI_for_crop.copyTo(cropped_image);
                    cv::imwrite(filename_l, cropped_image);

                    right_hand_bbox_x_min = (right_hand_bbox_x_min >= 0) ? right_hand_bbox_x_min : 0;
                    right_hand_bbox_x_min = (right_hand_bbox_x_min <= width) ? right_hand_bbox_x_min : width;
                    right_hand_bbox_y_min = (right_hand_bbox_y_min >= 0) ? right_hand_bbox_y_min : 0;
                    right_hand_bbox_y_min = (right_hand_bbox_y_min <= height) ? right_hand_bbox_y_min : height;

                    float right_hand_bbox_x_max = (right_hand_bbox_x_min + right_hand_bbox_size * 2 >= 0)
                        ? right_hand_bbox_x_min + right_hand_bbox_size * 2
                        : 0;
                    right_hand_bbox_x_max = (right_hand_bbox_x_min + right_hand_bbox_size * 2 <= width)
                        ? right_hand_bbox_x_min + right_hand_bbox_size * 2
                        : width;
                    float right_hand_bbox_y_max = (right_hand_bbox_y_min + right_hand_bbox_size * 2 >= 0)
                        ? right_hand_bbox_y_min + right_hand_bbox_size * 2
                        : 0;
                    right_hand_bbox_y_max = (right_hand_bbox_y_min + right_hand_bbox_size * 2 <= height)
                        ? right_hand_bbox_y_min + right_hand_bbox_size * 2
                        : height;

                    cv::Mat ROI_for_crop_r(image, cv::Rect(static_cast<uint32_t>(right_hand_bbox_x_min),
                        static_cast<uint32_t>(right_hand_bbox_y_min),
                        static_cast<uint32_t>(right_hand_bbox_x_max - right_hand_bbox_x_min),
                        static_cast<uint32_t>(right_hand_bbox_y_max - right_hand_bbox_y_min)));

                    cv::Mat cropped_image_r;
                    ROI_for_crop_r.copyTo(cropped_image_r);
                    cv::imwrite(filename_r, cropped_image_r);
                }
            }
            cv::Mat fpsPane(35, 155, CV_8UC3);
            fpsPane.setTo(cv::Scalar(153, 119, 76));
            cv::Mat srcRegion = image(cv::Rect(8, 8, fpsPane.cols, fpsPane.rows));
            cv::addWeighted(srcRegion, 0.4, fpsPane, 0.6, 0, srcRegion);
            std::stringstream fpsSs;
            fpsSs << "FPS: " << int(1000.0f / inferenceTime * 100) / 100.0f;
            cv::putText(image, fpsSs.str(), cv::Point(16, 32),
                        cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 0, 100));
            cv::imshow("ICV Human Pose Estimation", image);

            int key = cv::waitKey(delay) & 255;
            if (key == 'p') {
                delay = (delay == 0) ? 33 : 0;
            } else if (key == 27) {
                break;
            }
            FLAGS_num++;
        } while (cap.read(image));
    }
    catch (const std::exception& error) {
        std::cerr << "[ ERROR ] " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch (...) {
        std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Execution successful" << std::endl;
    return EXIT_SUCCESS;
}
