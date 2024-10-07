#pragma once
#include <arpa/inet.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

#include <thread>
#include <chrono>
#include <opencv2/opencv.hpp>  // opencv 绘图
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <JetsonGPIO.h>

// Global variables
std::vector<float> sock_angle_data;
std::vector<float> angle_1(3,0.0);
std::vector<float> angle_2(3,0.0);
// std::atomic<int> count(0);
// std::mutex lock;
    int count=0;

// Function to draw a vertical arrow on the image
void draw_vertical_arrow(cv::Mat image, double direction) {
    // cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return;
    }
    int height = image.rows;
    int width = image.cols;
    double length = 15 + std::abs(direction) * 0.75; // Scaling factor for arrow length
    cv::Point start_point(width / 2, height / 2);
    cv::Point end_point(start_point.x, static_cast<int>(start_point.y - length * std::sin(direction * CV_PI / 180.0)));
    cv::arrowedLine(image, start_point, end_point, cv::Scalar(255, 0, 0), 5); // Red arrow
    // cv::imwrite("/workspace/chang/deepstream_docker/sources/apps/sample_apps/deepstream_parallel_rgbt/imu/1.jpg", image);
}


// Function to draw a horizontal arrow on the image
void draw_horizontal_arrow(cv::Mat image, double direction) {
    // cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return;
    }
    int absangle = static_cast<int>(direction + 360.0);
    if( absangle % 360 < 5){
        return;
    }
    int height = image.rows;
    int width = image.cols;
    double length = 15 + std::abs(direction)*2; // Scaling factor for arrow length
    cv::Point start_point(width / 2, height / 2);
    cv::Point end_point;
    if (direction >= 0) {
        end_point = cv::Point(static_cast<int>(start_point.x + length), start_point.y);
    } else {
        end_point = cv::Point(static_cast<int>(start_point.x - length), start_point.y);
    }
    cv::arrowedLine(image, start_point, end_point, cv::Scalar(0, 255, 0), 10, 8, 0, 0.3); // Green arrow
    // cv::imwrite("/workspace/chang/deepstream_docker/sources/apps/sample_apps/deepstream_parallel_rgbt/imu&key_Test/1.jpg", image);
}


// Function to find the maximum absolute value and its index
std::pair<double, int> max_absolute_value_with_index(const std::vector<double>& arr) {
    double max_value = std::abs(arr[0]);
    int max_index = 0;
    for (size_t i = 1; i < arr.size(); ++i) {
        if (std::abs(arr[i]) > max_value) {
            max_value = arr[i];
            max_index = i;
        }
    }
    return {max_value, max_index};
}

void sock_recesive()
{
 int sock_server = socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in server_addr;
  memset(&server_addr, 0, sizeof(server_addr));
  server_addr.sin_family = AF_INET;
  server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
  server_addr.sin_port = htons(18889);
  bind(sock_server, (struct sockaddr *)&server_addr, sizeof(server_addr));
  listen(sock_server, 5);

// 2. 服务端接受客户端的请求
  int socklen = sizeof(struct sockaddr_in);
  sockaddr_in client_addr;
  int sock_client = accept(sock_server, (struct sockaddr *)&client_addr,
                           (socklen_t *)&socklen);
  printf("client %s has connnected\n", inet_ntoa(client_addr.sin_addr));

// 3. 接收来自Python程序的数据
    long long count=0;
    char buffer[1024]={0};
    while(1){

        memset(buffer, 0, sizeof(buffer));
        int bytesRead = recv(sock_client, buffer, sizeof(buffer)-1, 0);

        // 判断是否接收到了数据
        if (bytesRead > 0) {
            // 确保字符串结束符
            buffer[bytesRead] = '\0'; 
            //打印输出接收到的字符串数据 
            // printf("receive: %s\n", buffer);
            // 解析字符串数据
            std::string data_str(buffer);
            std::vector<float> imu_data;
            std::string token;
            std::stringstream ss(data_str);
            while (std::getline(ss, token, ',')) {
                try {
                    imu_data.push_back(std::stof(token));  // 转换为浮点数并存储
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Invalid number: " << token << std::endl;
                }
            }
            
            // 打印解析后的数据
            sock_angle_data=imu_data;
            // std::cout << "Parsed IMU data:" << std::endl;
            // for (float value : sock_angle_data) {
            //     std::cout << value << std::endl;
            // }

            // 发送响应给Python程序
            const char* response = "Data received successfully!";
            send(sock_client, response, strlen(response), 0);
        } 
        // else {
        //     if(count % 1000 == 0) printf("Failed to receive data from Python program.");
        // }
        count++;
    }

    // 关闭连接
    close(sock_server);
    close(sock_client);
}


double normalizeAngle(double angle) {
    // 使用fmod确保角度在-360到360之间
    angle = std::fmod(angle, 360.0);
    // 调整角度使其在-180到180之间
    if (angle > 180.0) {
        angle -= 360.0;
    } else if (angle < -180.0) {
        angle += 360.0;
    }
    return angle;
}

// 函数：计算两个角度之间的最小夹角
double calculateSmallestAngle(double initialAngle, double currentAngle) {
    // 首先标准化两个角度
    initialAngle = normalizeAngle(initialAngle);
    currentAngle = normalizeAngle(currentAngle);
    
    // 计算差值并标准化
    double angleDifference = normalizeAngle(currentAngle - initialAngle);
    
    // 返回最小夹角
    return angleDifference;
}

// Function to handle space key press
void handle_button_press(double& maxAngleVal, int& angleIndex) {
    // BOARD pin 16
    int count=0;
    int but_pin = 32; 
    // Pin Setup.
    GPIO::setmode(GPIO::BOARD);
    // set pin as an output pin with optional initial state of HIGH
    GPIO::setup(but_pin, GPIO::IN);
    while (1) {
        int stableState = GPIO::input(but_pin);
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 延时100ms
        int currentState = GPIO::input(but_pin);
        if(stableState && currentState == 0){
            count++;
            if (count == 1) {
                printf("First button record");
                // Record the first angle (Placeholder values)
                angle_1 = sock_angle_data;
                std::cout << "angle_1: " << angle_1[0] << ", " << angle_1[1] << ", " << angle_1[2] << std::endl;

            } 
            else if (count == 2) {
                printf("Second button record");
                maxAngleVal = 0.0;
                angleIndex  = 0;
                // std::string endpoint_image = "/workspace/chang/deepstream_docker/sources/apps/sample_apps/deepstream_parallel_rgbt/imu&key_Test/0.jpg";
                
                // if (angle_index == 1) {
                //     std::cout << "Drawing arrow on the image..." << std::endl;
                //     draw_vertical_arrow(endpoint_image, Max_value);
                //     std::cout << "Drawing vertical finished" << std::endl;
                // }
                // if (angle_index == 2) {
                //     std::cout << "Drawing arrow on the image..." << std::endl;
                //     draw_horizontal_arrow(endpoint_image, Max_value);
                //     std::cout << "Drawing horizontal finished" << std::endl;
                // }
                count = 0;
            }
        }
        // 循环延时100ms
        // std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if(count == 1){
                            // Record the second angle (Placeholder values)
            angle_2 = sock_angle_data;
            std::cout << "angle_2: " << angle_2[0] << ", " << angle_2[1] << ", " << angle_2[2] << std::endl;

            std::vector<double> angle_change(angle_2.size());
            for (size_t i = 0; i < angle_2.size(); ++i) {
                angle_change[i] = calculateSmallestAngle(angle_1[i], angle_2[i]);
                // angle_change[i] = angle_2[i] - angle_1[i];
            }
            std::cout << "angle_change: " << angle_change[0] << ", " << angle_change[1] << ", " << angle_change[2] << std::endl;

            auto [Max_value, angle_index] = max_absolute_value_with_index(angle_change);
            std::cout << "Max_value: " << Max_value << std::endl;
            std::cout << "angle_index: " << angle_index << std::endl;
            maxAngleVal = Max_value;
            angleIndex  = angle_index;
        }
    }
    GPIO::cleanup(but_pin); // cleanup only chan1

}

// int main() {

// if(1)
// {
//     std::thread sock_recesive_thread(sock_recesive);
//     std::thread handle_press_thread(handle_button_press);
//     sock_recesive_thread.join();
//     handle_press_thread.join();
// }
// else
// {
//  /**
//    * 1. 创建服务端socket，并绑定相应ip和端口
//    *    SOCK_STREAM对应的是TCP协议，安全可靠；SOCK_DGRAM是UDP协议，不可靠
//    *    listen使得该进程可以接收socket的请求，成为一个服务端。对应的是客户端的connect。
// */
//   int sock_server = socket(AF_INET, SOCK_STREAM, 0);
//   sockaddr_in server_addr;
//   memset(&server_addr, 0, sizeof(server_addr));
//   server_addr.sin_family = AF_INET;
//   server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
//   server_addr.sin_port = htons(8889);
//   bind(sock_server, (struct sockaddr *)&server_addr, sizeof(server_addr));
//   listen(sock_server, 5);

// // 2. 服务端接受客户端的请求
//   int socklen = sizeof(struct sockaddr_in);
//   sockaddr_in client_addr;
//   int sock_client = accept(sock_server, (struct sockaddr *)&client_addr,
//                            (socklen_t *)&socklen);
//   printf("client %s has connnected\n", inet_ntoa(client_addr.sin_addr));

// // 3. 接收来自Python程序的数据
//     char buffer[1024]={0};
//     while(1){

//         memset(buffer, 0, sizeof(buffer));
//         int bytesRead = recv(sock_client, buffer, sizeof(buffer)-1, 0);

//         // 判断是否接收到了数据
//         if (bytesRead > 0) {
//             // 确保字符串结束符
//             buffer[bytesRead] = '\0'; 
//             //打印输出接收到的字符串数据 
//             printf("receive: %s\n", buffer);
//             // 解析字符串数据
//             std::string data_str(buffer);
//             std::vector<float> imu_data;
//             std::string token;
//             std::stringstream ss(data_str);
//             while (std::getline(ss, token, ',')) {
//                 try {
//                     imu_data.push_back(std::stof(token));  // 转换为浮点数并存储
//                 } catch (const std::invalid_argument& e) {
//                     std::cerr << "Invalid number: " << token << std::endl;
//                 }
//             }
            
//             // 打印解析后的数据
//             std::cout << "Parsed IMU data:" << std::endl;
//             for (float value : imu_data) {
//                 std::cout << value << std::endl;
//             }

//             // 发送响应给Python程序
//             const char* response = "Data received successfully!";
//             send(sock_client, response, strlen(response), 0);
//         } else {
//             printf("Failed to receive data from Python program.");
//         }
//     }

//     // 关闭连接
//     close(sock_server);
//     close(sock_client);
// }
//     return 0;
// }
