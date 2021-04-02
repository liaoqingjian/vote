#   测试有没有使用gpu

```
tensorflow-gpu
```

```python
import tensorflow as tf
print(tf.test.is_gpu_available())
```

# 进入虚拟环境

```python
activate
conda activate keras-yolo3-1
```

# 有用的链接

https://blog.csdn.net/qq_38163755/article/details/88583016

https://zhuanlan.zhihu.com/p/38720146

# 自己博客的链接

https://gitee.com/liaoqingjian/lqj/tree/master/pythonProject

https://github.com/liaoqingjian    

用户名：

```
liaoqingjian
```

 密码：

```
$Jian19980501
```

https://blog.csdn.net/liaoqingjianpip

# python安装包

```python
pip install opencv-python==3.4.8.29
Keras== 2.1.5
```

```python
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
```

```
cuda路径：
C:\ProgramData\NVIDIA Corporation\CUDA Samples\v9.0
```

```python
pip install torch===1.2.0 torchvision===0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
```

```python
pip install http://download.pytorch.org/whl/cu80/torch-0.4.0-cp36-cp36m-win_amd64.whl
    
pip install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-win_amd64.whl
    
pip install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp36-cp36m-win_amd64.whl
```

```
import torch
torch.cuda.is_available() *# True 就完事儿了
```

*

# 任意收放窗口代码

```
cv2.namedWindow("video", cv2.WINDOW_NORMAL)
cv2.imshow("video",frame)
```

# 创建python虚拟环境

```python
conda env list
conda create -n keras python=3.6
conda activate keras-yolo-test
```

https://github.com/ming71/yolov3-pytorch

```python
recognizer = cv2.face.LBPHFaceRecognizer_create()
Traceback (most recent call last):
  File "C:/Users/m'y/PycharmProjects/pythonProject/faceStorage/faceTraining.py", line 37, in <module>
    recognizer = cv2.face.LBPHFaceRecognizer_create()
AttributeError: module 'cv2.cv2' has no attribute 'face'
    
pip install opencv-contrib-python
```

# os模块用法



# 安装MySQL安装包设置命令

```
mysqld -remove mysql
mysqld -install
管理员cmd命令下：net start mysql
mysqld --initialize-insecure
mysql -h localhost -u root -p
ALTER USER root@localhost IDENTIFIED BY '123456'
```

# 两个可以连接外网的VPN

setup vpn 
坚果 vpn  https://nutsvpn.work/ 用户名：13524853024 密码：7yrqasqi

# Ubuntu 常用命令大全

1. 系统相关
uname -a 显示当前系统相关信息
sudo 临时获取超级用户权限
su root 切换 root 用户
sudo shutdown 关机
sudo reboot 重启
sudo nautilus 进入有 root 权限的文件管理器
ps -A 查看当前有哪些进程
kill 5 位进程号 结束进程
sudo fdisk -l 查看磁盘信息
sudo mount /dev/sdb1 /mnt 挂载磁盘到某一路径
sudo mount -r /dev/sdb1 /mnt/ 以只读方式挂载
sudo umount /dev/sdb1 卸载磁盘
sudo blkid 查看磁盘分区的 UUID
sudo vi /etc/fstab 开机自动挂载磁盘
UUID=11263962-9715-473f-9421-0b604e895aaa /data ext4 defaults 0 1
sudo mount -a 验证一下配置是否正确
efibootmgr 查看系统启动顺序
ifconfig 网络配置，IP 地址查看
man command-name 查找命令详细手册
command-name --help 查找某一命令的帮助
设置静态 IP 地址
sudo vi /etc/network/interfaces
添加以下内容
auto enp129s0f1
iface enp129s0f1 inet static
address 192.168.1.254 # IP 地址
gateway 192.168.1.1 #
netmask 255.255.255.0 # 子网掩码
dns-nameservers 8.8.8.8 8.8.4.4 # DNS 解析
2. 用户及权限管理
sudo adduser username 新添加用户
sudo passwd root 设置 root 用户密码
sudo vim /etc/sudoers 赋予新用户 root 权限

root ALL=(ALL:ALL) ALL
username ALL=(ALL:ALL) ALL   新添加此行即可
chown user-name filename 改变文件的所属用户
chmod u+rwx g+r o+r filename 用户添加读写运行权限，组成员添加读权限，其他用户添加读权限
chmod a+w filename 所有用户添加写权限
chmod 777 filename 所有用户添加读写运行权限
3. 软件安装
sudo apt-get update 更新软件列表，在文件 /etc/apt/sources.list 中列出
sudo apt-get upgrade 更新软件
sudo apt-get install software-name 安装在软件库中的软件
sudo apt-get remove 卸载软件
sudo apt-get purge 卸载软件并删除配置文件
sudo apt-get clean 清除软件包缓存
sudo apt-get autoclean 清除缓存
sudo apt-get autoremove 清除不必要的依赖
sudo apt-get install -f 修复安装依赖问题
sudo dpkg -i *.deb 安装 deb 软件
dpkg -l 查看所有安装的软件
dpkg -l | grep software-name 配合 grep 命令可查看具体的某一个软件是否安装
sudo echo "google-chrome-stable hold" | sudo dpkg --set-selections 不更新某个软件
sudo echo "google-chrome-stable install" | sudo dpkg --set-selections 恢复更新某个软件
4. 目录文件操作
cd 切换目录，～为家目录，/为根目录，./为当前目录
cd .. 切换到上级目录
cd - 切换到上一次所在的目录
pwd 查看当前所在目录
ls 查看当前目录下的文件夹和文件名，-a显示隐藏文件，-l显示文件详细信息
mkdir directory-name 新建文件夹
rmdir directory-name 删除文件夹(必须为空)
rm -rf directory-name 强制并递归删除文件夹
cp src-file dst-file 复制文件
mv src-file dst-file 移动文件
ln -s src-file dst-file 建立软链接
find path -name string 查找路经所在范围内满足字符串匹配的文件和目录
cat filename 显示文件内容
head -n 2 filename 显示文件前两行的内容
tail -n 2 filename 显示文件末尾两行的内容
5. 终端快捷键
ctrl + l 清屏
ctrl + c 终止命令
ctrl + d 退出 shell
ctrl + z 将当前进程置于后台，fg 还原
ctrl + r 从命令历史中找
ctrl + u 清除光标到行首的字符（还有剪切功能）
ctrl + w 清除光标之前一个单词 （还有剪切功能）
ctrl + k 清除光标到行尾的字符（还有剪切功能）
ctrl + y 粘贴 Ctrl+u 或 Ctrl+k 剪切的内容
ctrl + t 交换光标前两个字符
Alt + d 由光标位置开始，往行尾删删除单词
Alt + . 使用上一条命令的最后一个参数
Alt – b || ctrl + 左方向键 往回(左)移动一个单词
Alt – f || ctrl + 右方向键 - 往后(右)移动一个单词
!! 执行上一条命令。

# Vmware虚拟机安装

Vmware官网：https://www.virtualbox.org/wiki/Downloads

Ubuntu18.04的镜像：http://releases.ubuntu.com/18.04.2/ubuntu-18.04.2-desktop-amd64.iso
其他版本都可以在这里找：http://releases.ubuntu.com/

# 创建数据库数据表代码

```
创建数据库test1
create database images；
数据库查询
show databases;
选择要操作的数据库
use images；
查看 test1 数据库中创建的所有数据表
show tables;
创建表
create table images
(
	id int auto_increment,
	data mediumblob null,
	name varchar(50) null,
	constraint images_pk
		primary key (id)
);
 create table information
 (
    id int auto_increment,
    yname varchar(50) not null,
    name varchar(50) not null,
    email varchar(50) not null,
    telephone varchar(50) not null,
    constraint images_pk
    primary key (id)
 
 );
查看表：
desc images;
插入记录：
insert into emp(ename,hiredate,sal) values('zzx1','2000-01-01','2000');
查看表里面的内容：
select * from information;
删除表：
drop table information;
修改数据表某一行里的内容：
update information set email="981479538@qq.com" where id=1;
```

# python相关

```python
安装scrapy库：conda install scrapy
```

```
# coding=gbk
#encoding=utf-8
#encoding=gb2312
#unicoding=gb2312
```

代理ip软件，土豆ip

用户名：13524853024

密码：jian19980501

cuda地址：https://developer.nvidia.com/cuda-toolkit-archive

conda env list

conda create -n test python=3.6

conda install cudatookit

conda install cudnn

nvcc --version

VMware25位密钥：YG5H2-ANZOH-M8ERY-TXZZZ-YKRV8

pip install h5py==2.10 -i https://pypi.tuna.tsinghua.edu.cn/simple/

pip install http://download.pytorch.org/whl/cu80/torch-0.4.0-cp36-cp36m-win_amd64.whl

```python
需要给gpu分配内存，注意代码放入运行的文件：
import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
#选择哪一块gpu,如果是-1，就是调用cpu

config = tf.ConfigProto()#对session进行参数配置
config.allow_soft_placement=True
# 如果你指定的设备不存在，允许TF自动分配设备
config.gpu_options.per_process_gpu_memory_fraction=0.7
#分配百分之七十的显存给程序使用，避免内存溢出，可以自己调整
config.gpu_options.allow_growth = True
#按需分配显存，这个比较重要
session = tf.Session(config=config)
```

# 在虚拟机里面的操作

tf-slim

u-net slim

sh ./pycharm.sh

sudo gedit ~/.bashrc

export PATH=/home/lqj/anaconda3/bin:$PATH

source ~/.bashrc

su root

/usr/local/bin/charm

bash Anaconda3-5.0.1-Linux-x86_64.sh

/home/lqj/Anaconda3

永久修改
linux:
修改 ~/.pip/pip.conf (没有就创建一个)， 内容如下：

```
[global]
timeout=6000
index-url=http://pypi.douban.com/simple/
[install]
trusted-host=pypi.douban.com
```

pip install scrapy -i [https://pypi.douban.com/simple/](http://pypi.douban.com/simple/) 

sh NVIDIA-Linux-x86_64-390.42.run

source ~/.bashrc

nvidia-smi

sudo passwd root

su root

su lqj

sudo chmod +x NVIDIA-Linux-x86_64-455.38.run

sudo bash  NVIDIA-Linux-x86_64-455.38.run --no-opengl-files   -no-nouveau-check

sudo apt install nvidia-utils-390 

lshw -numeric -C display

sudo apt install net-tools

sudo apt-get remove –purge nvidia*

sudo apt install nvidia-utils-440

```python
if hasattr(torch.cuda, 'empty_cache'): 
	torch.cuda.empty_cache()
```

pip install tqdm

```python
import torch
use_gpu = torch.cuda.is_available()
print(use_gpu)
```

sudo nano /etc/apt/sources.list

```
deb http://debian.ustc.edu.cn/ubuntu/ trusty main multiverse restricted universe

deb http://debian.ustc.edu.cn/ubuntu/ trusty-backports main multiverse restricted universe

deb http://debian.ustc.edu.cn/ubuntu/ trusty-proposed main multiverse restricted universe

deb http://debian.ustc.edu.cn/ubuntu/ trusty-security main multiverse restricted universe

deb http://debian.ustc.edu.cn/ubuntu/ trusty-updates main multiverse restricted universe

deb-src http://debian.ustc.edu.cn/ubuntu/ trusty main multiverse restricted universe

deb-src http://debian.ustc.edu.cn/ubuntu/ trusty-backports main multiverse restricted universe

deb-src http://debian.ustc.edu.cn/ubuntu/ trusty-proposed main multiverse restricted universe

deb-src http://debian.ustc.edu.cn/ubuntu/ trusty-security main multiverse restricted universe

deb-src http://debian.ustc.edu.cn/ubuntu/ trusty-updates main multiverse restricted universe
```

sudo apt-get update

sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

sudo apt-get install ros-desktop-full

```
sudo rosdep init
rosdep update
```

```
echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

```
sudo apt-get install python-rosinstall python-rosinstall-generator python-wstool build-essential
```

sudo apt install python3-rosdep2

```python
# 加载启动项，这里设置headless，表示不启动浏览器，只开一个监听接口获取返回值
option = webdriver.ChromeOptions()
option.add_argument('headless')

```

# 人脸入库识别代码包括UI界面设计需要的命令

pip install sklearn

pip install numpy==1.16.2

pip install PyQt5-tools

background-color:rgb(0,0,0)

python -m PyQt5.uic.pyuic face2.ui -o face2.py

```
show = cv2.resize(image, (280, 320))  # 把读到的帧的大小重新设置为 640x480
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                 QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.label_3.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 往显示视频的Label里 显示QImage
        self.label_2.setScaledContents(True)
```

self.lineEdit.setPlaceholderText("请输入姓名拼音")

self.lineEdit_2.setPlaceholderText("请输入中文姓名")

```
self.label_4.setText("{}是安防系统的安防人员，请进入！".format(x))
```

# python相关

a=self.lineEdit.text()

牛巴巴视频解析网站：http://mv.688ing.com/

```python
torch.set_num_threads(1)
```

pyinstaller -F -w    a.py

pyinstaller -F -w  -i 1.ico  a.py 

pip install -U wxPython

猪八戒网

解放号、云沃客、码市、程序员客栈等平台

```python
pip install -r requirements.txt
```

# Django网页制作前期准备代码命令

django-admin startproject shop

cd shop

django-admin startapp shopping

python manage.py createsuperuser（创建超级用户的用户名和密码）

python manage.py runserver

用户名：liaoqingjian

密码：jian19980501

pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

python demo/demo.py video --config config/nanodet-m.yml --model nanodet_m.pth --path 1.mp4

# 学python以后的工作方向

人工智能,大数据,网络爬虫工程师,Python web全栈工程师,Python自动化运维,Python自动化测试

# 导入库，导出库

```python
pip freeze > requirements.txt
pip install -r requirements.txt
```

#  selenium+Chrome获取数据

driver = webdriver.Chrome()

driver.maximize_window()  *# 窗口最大化*

driver.quit()#关闭窗口

# pytorch yolov5命令

python detect.py --source 3.mp4 --view-img

# python多线程网络摄像头堵塞问题

```python
import cv2
import time
import multiprocessing as mp

"""
Source: Yonv1943 2018-06-17
https://github.com/Yonv1943/Python/tree/master/Demo
"""

def image_put(q, name, pwd, ip, channel=1):
    cap = cv2.VideoCapture("rtsp://%s:%s@%s//Streaming/Channels/%d" % (name, pwd, ip, channel))
    if cap.isOpened():
        print('HIKVISION')
    else:
        cap = cv2.VideoCapture("rtsp://%s:%s@%s/cam/realmonitor?channel=%d&subtype=0" % (name, pwd, ip, channel))
        print('DaHua')

    while True:
        q.put(cap.read()[1])
        q.get() if q.qsize() > 1 else time.sleep(0.01)

def image_get(q, window_name):
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    while True:
        frame = q.get()
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)

def run_multi_camera():
    # user_name, user_pwd = "admin", "password"
    user_name, user_pwd = "admin", "admin123456"
    camera_ip_l = [
        "172.20.114.26",  # ipv4
        "[fe80::3aaf:29ff:fed3:d260]",  # ipv6
        # 把你的摄像头的地址放到这里，如果是ipv6，那么需要加一个中括号。
    ]

    mp.set_start_method(method='spawn')  # init
    queues = [mp.Queue(maxsize=4) for _ in camera_ip_l]

    processes = []
    for queue, camera_ip in zip(queues, camera_ip_l):
        processes.append(mp.Process(target=image_put, args=(queue, user_name, user_pwd, camera_ip)))
        processes.append(mp.Process(target=image_get, args=(queue, camera_ip)))

    for process in processes:
        process.daemon = True
        process.start()
    for process in processes:
        process.join()

if __name__ == '__main__':
    run_multi_camera()
```

# yolo3-pytorch需要的包

pip install opencv-contrib-python

https://blog.csdn.net/gentlemanman/article/details/84965144

pip install pillow

pip install opencv-python

pip install tqdm

pip install numpy==1.16.2

pip install torch===1.2.0 torchvision===0.4.0 -f https://download.pytorch.org/whl/torch_stable.html



# 人脸打卡系统需要的库

pip install tensorflow-Gpu==1.13.1

pip install numpy==1.16.2

pip install scipy==1.2.1

pip install sklearn

pip install xpinyin

pip install Pillow

pip install opencv-python==3.4.8.29

conda install cudatoolkit=10.0

conda install cudnn=7

# 创建vue运行环境代码

npm config set prefix "c:\Program Files \nodejs\node_global"

npm config set cache "c:\Program Files\nodejs\node_cache"

npm config set strict-ssl false

npm config set strict-ssl false

npm create project

npm install video.js

npm run  serve

