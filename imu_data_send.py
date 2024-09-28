# coding:UTF-8
# Version: V1.0.1
import serial
import numpy as np
import socket
import struct
import time
# 创建了三个列表（List）来存储数据，每个列表都被初始化为包含8个0.0的元素
ACCData = [0.0]*8
GYROData = [0.0]*8
AngleData = [0.0]*8
MAGData = [0.0]*8
    

FrameState = 0  # What is the state of the judgment
Bytenum = 0  # Read the number of digits in this paragraph #读取本段中的数字位数
CheckSum = 0  # Sum check bit 总和校验位

acc = [0.0]*3
gyro = [0.0]*3
Angle = [0.0]*3
mag= [0.0]*3
data_imu_sub=[]


# 这些数据点通常是连续采集的，因此在处理时需要考虑数据的采集频率（即每秒采集多少个数据点）
def datafrom_serial(inputdata):  # New core procedures, read the data partition, each read to the corresponding array  新的核心程序，读取数据分区，每个读取到相应的数组
    global FrameState    # Declare global variables
    global Bytenum
    global CheckSum
    global acc
    global gyro
    global Angle               
    global mag
    global data_imu_sub
    # result=[]
    for data in inputdata:  # Traversal the input data  遍历输入数据遍历输入数据
        if FrameState == 0:  # When the state is not determined, enter the following judgment #当状态未确定时，输入以下判断
            if data == 0x55 and Bytenum == 0:  # When 0x55 is the first digit, start reading data and increment bytenum 当0x55是第一位数字时，开始读取数据并递增字节数
                CheckSum = data
                Bytenum = 1
                continue
            elif data == 0x51 and Bytenum == 1:  # Change the frame if byte is not 0 and 0x51 is identified #如果字节不为0并且标识了0x51，则更改帧
                CheckSum += data
                FrameState = 1
                Bytenum = 2
            elif data == 0x52 and Bytenum == 1:
                CheckSum += data
                FrameState = 2
                Bytenum = 2
            elif data == 0x53 and Bytenum == 1:
                CheckSum += data
                FrameState = 3
                Bytenum = 2
            elif data == 0x54 and Bytenum == 1:
                CheckSum += data
                FrameState = 4
                Bytenum = 2
        elif FrameState == 1:  # acc

            if Bytenum < 10:            # Read 8 data
                ACCData[Bytenum-2] = data  # Starting from 0
                CheckSum += data
                Bytenum += 1
            else:
                if data == (CheckSum & 0xff):  # verify check bit
                    acc = get_acc(ACCData)
                CheckSum = 0  # Each data is zeroed and a new circular judgment is made
                Bytenum = 0
                FrameState = 0
        elif FrameState == 2:  # gyro

            if Bytenum < 10:
                GYROData[Bytenum-2] = data
                CheckSum += data
                Bytenum += 1
            else:
                if data == (CheckSum & 0xff):
                    gyro = get_gyro(GYROData)
                CheckSum = 0
                Bytenum = 0
                FrameState = 0
        elif FrameState == 3:  # angle

            if Bytenum < 10:
                AngleData[Bytenum-2] = data
                CheckSum += data
                Bytenum += 1
            else:
                if data == (CheckSum & 0xff):
                    Angle = get_angle(AngleData)
                   
                CheckSum = 0
                Bytenum = 0
                FrameState = 0
        elif FrameState == 4:  # mag
            if Bytenum < 10:
                MAGData[Bytenum-2] = data
                CheckSum += data
                Bytenum += 1
            else:
                if data == (CheckSum & 0xff):
                    mag = get_mag(MAGData)
                    data_imu_sub = list(gyro)+list(acc)+list(mag)

                CheckSum = 0
                Bytenum = 0
                FrameState = 0

    # print(
    #     "acc:%10.3f %10.3f %10.3f \ngyro:%10.3f %10.3f %10.3f \nmag:%10.3f %10.3f %10.3f" % result)

def get_acc(datahex):
    axl = datahex[0]
    axh = datahex[1]
    ayl = datahex[2]
    ayh = datahex[3]
    azl = datahex[4]
    azh = datahex[5]
    k_acc = 16.0
    acc_x = (axh << 8 | axl) / 32768.0 * k_acc
    acc_y = (ayh << 8 | ayl) / 32768.0 * k_acc
    acc_z = (azh << 8 | azl) / 32768.0 * k_acc
    if acc_x >= k_acc:
        acc_x -= 2 * k_acc
    if acc_y >= k_acc:
        acc_y -= 2 * k_acc
    if acc_z >= k_acc:
        acc_z -= 2 * k_acc
    return acc_x, acc_y, acc_z

def get_gyro(datahex):
    wxl = datahex[0]
    wxh = datahex[1]
    wyl = datahex[2]
    wyh = datahex[3]
    wzl = datahex[4]
    wzh = datahex[5]
    k_gyro = 2000.0
    gyro_x = (wxh << 8 | wxl) / 32768.0 * k_gyro
    gyro_y = (wyh << 8 | wyl) / 32768.0 * k_gyro
    gyro_z = (wzh << 8 | wzl) / 32768.0 * k_gyro
    if gyro_x >= k_gyro:
        gyro_x -= 2 * k_gyro
    if gyro_y >= k_gyro:
        gyro_y -= 2 * k_gyro
    if gyro_z >= k_gyro:
        gyro_z -= 2 * k_gyro
    return gyro_x, gyro_y, gyro_z

def get_angle(datahex):
    rxl = datahex[0]
    rxh = datahex[1]
    ryl = datahex[2]
    ryh = datahex[3]
    rzl = datahex[4]
    rzh = datahex[5]
    k_angle = 180.0
    angle_x = (rxh << 8 | rxl) / 32768.0 * k_angle
    angle_y = (ryh << 8 | ryl) / 32768.0 * k_angle
    angle_z = (rzh << 8 | rzl) / 32768.0 * k_angle
    if angle_x >= k_angle:
        angle_x -= 2 * k_angle
    if angle_y >= k_angle:
        angle_y -= 2 * k_angle
    if angle_z >= k_angle:
        angle_z -= 2 * k_angle
    return angle_x, angle_y, angle_z

def get_mag(datahex):
    # 分辨率从mGauss转换为uT  1guass =100uT 0.0667mguass * 10^-3 = 0.0000667guass*100=0.00667uT
    RESOLUTION_UT_PER_LSB = 0.00667  # 0.0667 mGauss * 10 = 0.667 uT  
    # 量程（以uT为单位）  
    RANGE_UT = 200  # 2 gauss=0.0002T * 10^6= 200 uT  

    mxl = int(datahex[0])
    mxh = int(datahex[1])
    myl = datahex[2]
    myh = datahex[3]
    mzl = datahex[4]
    mzh = datahex[5]
    k_mag=200
    mag_x  = mxh << 8 | mxl 
    mag_y  = myh << 8 | myl
    mag_z  = mzh << 8 | mzl
    
        # 将有符号的LSB值转换为uT  
    # 注意：Python的int类型可以自动处理负数，所以我们不需要额外的符号处理  
    # 转换为有符号的uT值，并保留两位小数  

    mag_x = round(mag_x * RESOLUTION_UT_PER_LSB, 2)  
    mag_y = round(mag_y * RESOLUTION_UT_PER_LSB, 2)  
    mag_z = round(mag_z * RESOLUTION_UT_PER_LSB, 2)  

    return mag_x, mag_y, mag_z

def socket_senddata(host, port, data_message):
    # 创建一个socket对象  
    server_client  =socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    # 连接到服务器  
    server_client.connect((host, port))
    # 将列表打包成字节串发送，这里简单使用struct.pack，可以根据需要调整格式  
    packed_data = struct.pack('9f', *data_message)  
    # 发送数据  
    server_client.sendall(packed_data)  
    print(f"Sent: {packed_data}")
    # 接收C++程序的响应
    response = server_client.recv(1024).decode()
    if len(response) > 0:
        print("Received response from C++ program:", response)
    else:
        print("Failed to receive response from C++ program.")


if __name__ == '__main__':
    #变量初始化
    HOST = "localhost"  # 服务器的IP地址  
    PORT = 18889        # 服务器监听的端口  
    DEBUG = True
    #IMU串口配置
    port = '/dev/ttyUSB0' # USB serial port 
    baud = 9600   # Same baud rate as the INERTIAL navigation module
    ser = serial.Serial(port, baud, timeout=0.5)
    print("Serial is Opened:", ser.is_open)

    print('===================Angle Data Send is beginning=====================')
    server_client  =socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    # 连接到服务器  
    # server_client.connect((HOST, PORT))
    while True:
            try:
                # 创建 socket 对象
                server_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # 尝试连接
                server_client.connect((HOST, PORT))
                print(f"连接成功: {HOST}:{PORT}")
                # 连接成功，退出循环
                break
            except socket.error as e:
                print(f"连接失败: {HOST}:{PORT}，原因: {e}")
                # attempt += 1
                # print(f"等待 {delay} 秒后重试...")
                time.sleep(1)
                # 如果达到重试次数，抛出异常
                # if attempt == retries:
                #     raise Exception(f"无法连接到 {HOST}:{PORT}，已达到最大重试次数 {retries}")

    print("TCP Data Transmission connected")
    index = 0
    while True:
        datahex = ser.read(33)
        #获取Angle数据
        datafrom_serial(datahex)
        if DEBUG and index % 25 == 0:
            print(f"Angle_Data(the order of Now_Angle is X(Roll) Y(Pitch) Z(Yal)):{Angle}\n")
        # 将数据转换为逗号分隔的字符串
        data_str = ','.join(map(str, Angle))
        # 转换为字节
        data_bytes = data_str.encode('utf-8')        
        # 发送数据
        server_client.sendall(data_bytes)  
        # 接收C++程序的响应
        response = server_client.recv(1024).decode()
        index +=1
        if len(response) > 0:
            if index % 100 == 0:
                print("Received response from C++ program:", response)
        else:
            print("Failed to receive response from C++ program.")
            break
    server_client.close()
    print("server end, exit!")