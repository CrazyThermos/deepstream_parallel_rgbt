

import numpy as np
import cv2 as cv


def main():
    img = cv.imread('./1_1.png', 0) # 增强后图像
    img0 = cv.imread('./1.png', 0) # 增强前图像
    
    # 定义卷积核
    kernel = np.array([[-1/8, -1/8, -1/8], [-1/8, 1, -1/8], [-1/8, -1/8, -1/8]])
    
    # 执行滤波
    avg_filtered = cv.filter2D(img, -1, kernel)
    avg_filtered0 = cv.filter2D(img0, -1, kernel)

    result = np.mean(avg_filtered)
    result0 = np.mean(avg_filtered0)

    print(result, result0) # 韦伯对比度
    print(f"{int((result-result0)/result0*100)}%") # 提升程度


if __name__ == "__main__":
    main()
