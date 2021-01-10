# USTC Cluster Usage
See [官方使用教程](http://mccipc.ustc.edu.cn/mediawiki/index.php/Gpu-cluster-manual) for more details.


## 常用指令：

```python
# 登录集群gwork节点(用于任务提交)
ssh -p 39099 YOUR_USER_NAME@202.38.69.241 #初始密码a123456, passwd 修改密码 
# 注意修改此节点密码并不影响gproc节点密码

# 登录集群gproc节点(用于数据传输与解压缩)
ssh -p 37240 username@202.38.69.241 # 初始密码a123456, passwd修改密码
ssh gproc # 或从gwork节点进入

# 数据传输
zip -r file.zip file #压缩数据
scp -P 37240 wangshuai@202.38.69.241:remote_file local_folder #1. 从集群下载数据
scp -P 37240 local_file wangshuai@202.38.69.241:remote_folder #2. 往集群传数据
# -r 下载文件夹
md5sum local_file #3. 数据完整性校验
md5sum remote_file
unzip -d folder_name file.zip #4. 解压缩

# 进入调试节点开启容器进行调试
ssh g101 # 从gwork 节点进入
sudo docker images # 查看现有镜像
startdocker -u "-it -v /gdata/wangshuai:/gdata/wangshuai -w /ghome/wangshuai/xx" -c /bin/bash bit:5000/deepo_9 # 或使用 bit:5000/ws-py3-tf-keras; 可使用多个-v命令挂载多个


sudo docker ps # 查看正在运行的容器
sudo docker ps -a # 查看所有容器
exit # 正常退出容器并销毁
close terminal directly # 非正常退出容器，方便下次使用 （但这样是有风险的，容器无法自动删除，如果你挂载了你的gdata，然后你在gwork节点就对gdata没有写权限了，容器里才有，就要请管理员手动删除--貌似不手动删除过一会就又好了？下次申请容器时不挂载数据试试）

# 提交任务并查看
ssh gwork
startdocker -u "-v /gdata/wangshuai:/gdata/wangshuai -w /ghome/wangshuai/UltraNet/pointnet" -c "python train.py" bit:5000/deepo # 编写pbs; 通过python解释器运行

startdocker -u "-v /gdata/用户名:/gdata/用户名 -v /gpub/ImageNet-Caffe:/gpub/ImageNet-Caffe" -s /ghome/用户名/run.sh bit:5000/deepo # 直接运行脚本文件

startdocker  -D <my-data-path> -P <my-proc-config-path> -s <my-script-file> bit:5000/deepo # 更通用的指令
startdocker -u "--ipc=host -v /gpub/leftImg8bit_sequence:/gpub/leftImg8bit_sequence --shm-size 8G" -D "/gdata/jinlk" -s "/ghome/jinlk/VSS/DVSNet_pytorch/scripts/train_feat2_df_with_scale_4.sh" bit:5000/ws-py3-tf-keras # 示例
chk_gpu <结点名> # 查看可用资源
qsub xx.pbs # 提交任务
qstat # 查看任务状态
qdel # 中止任务
sudo chk_res <结点名> <用户名> # 查看job资源是否正确释放
chk_gpuused # 查看gpu使用情况


## 多卡调试 (G101默认仅分配一块GPU卡，由 login 时随机决定)
ssh g101
echo $CUDA_VISIBLE_DEVICES # 查看所分配卡的物理编号
nvidia-smi # 查看空闲的卡编号
export CUDA_VISIBLE_DEVICES="0,1,3,5" # 设置使用多卡
# 注意： 在容器内的代码中无需在意CUDA_VISIBLE_DEVICES的值，也禁止改变它，容器内使用GPUid时，总是从0编号，和实际的物理卡ID无关。例如，某用户登录后echo $CUDA_VISIBLE_DEVICES 发现CUDA_VISIBLE_DEVICES=4，但通过nvidia-smi查看目前3，6卡空闲，则可以通过命令export CUDA_VISIBLE_DEVICES="3,6"来指定可用的物理卡，然后 在containter内部，如pytorch可以通过device 指定cuda:0，cuda:1使用这两块卡。
startdocker -u "-it -v /gdata/wangshuai:/gdata/wangshuai -w /ghome/wangshuai/xx" -c /bin/bash bit:5000/ws-py3-tf-keras

sudo docker stats <containerid> # 监控开启容器内 内存使用情况 (需新开终端)
sudo docker exec -it wangshuai /bin/bash # 再次新开终端后登录集群进入容器
watch nvidia-smi # 监控显存使用情况: 主要监控两个GPU是否同时被使用！

sudo rpt_detail ## 根据walltime计算本月集群任务使用时间


# 使用tensorborad
id # 获取自己的用户id, port即为自己的id (1364)
tensorboard --logdir <log-path> --port 31364 # gwork运行
202.38.69.241:31364 #本地浏览器访问


# 自定义镜像
@管理员_朱宏民 您好，能否帮忙开一个root容器，挂载 /gdata/wangshuai，基于镜像 bit:5000/ws-py3-tf-keras, 谢谢！#1.请管理员开容器
sudo docker attach container_name #2.进入容器调试
sudo docker start CONTAINER_ID/CONTAINER_NAME #3.重启已经关闭的容器
在自己的根目录下面建立一个dockertmp子目录并编辑 Dockerfile #4.Dockerfile
@管理员_朱宏民 您好，能否帮忙build一个镜像，路径在 /ghome/wangshuai/dockertmp/Dockerfile, 基础镜像为bit:5000/ws-py3-tf-keras,新增加包为: laspy等点云处理包，命名为 bit:5000/ws-py3-tf-keras, 谢谢！ #5. 请管理员帮忙编译

# 
# 1. 模型文件无删除权限可以通过root容器删除

## 图形界面登录
# 1. 本地Linux系统文件管理器 -> 连接到服务器 -> 输入对应文件夹网址 -> 输入用户名和密码
  - sftp://202.38.69.241:39099/ghome/username
  - sftp://202.38.69.241:39099/gdata/wangshuai
  - sftp://202.38.69.241:37240/gdata/wangshuai
```

### 使用Putty

```python
## 安装
sudo apt-get install putty

## 设置
- Session/Host Name: 202.38.69.241
- Session/port: 39099
- Session/saved_sessions: ustc_cluster-> save (下次可直接双击连接或load->open；修改设置后记得save)
- window: columns: 90; rows: 90
- window/fonts: Font used for ordinary text: client:Ubuntu Mono 14
- Connection: keepalives: 60
- connection/data: username: wangshuai
- Session/saved_sessions: ustc_cluster-> save 
    
## 参考资料
<https://blog.csdn.net/skypeGNU/article/details/11655713>
<https://www.cnblogs.com/yuwentao/archive/2013/01/06/2846953.html>
```




## pbs 任务脚本
```bash
#PBS    -N  testjob
#PBS    -o  /ghome/<username>/$PBS_JOBID.out
#PBS    -e  /ghome/<username>/$PBS_JOBID.err
#PBS    -l nodes=1:gpus=1:S
#PBS    -r y
cd $PBS_O_WORKDIR
echo Time is `date`
echo Directory is $PWD
echo This job runs on following nodes:
echo -n "Node:"
cat $PBS_NODEFILE
echo -n "Gpus:"
cat $PBS_GPUFILE
echo "CUDA_VISIBLE_DEVICES:"$CUDA_VISIBLE_DEVICES
startdocker -c "python /ghome/<username>/mytest.py"  bit:5000/deepo
```


## 5.Dockerfile 示例
完整镜像列表：http://mccipc.ustc.edu.cn/mediawiki/index.php/Docker-images

```dockerfile
  FROM bit:5000/nvidia-tensorflow-17.05
  ADD cuda8-cudnn5.tar /tmp/
  RUN apt update -y && \
      apt install -y cmake libboost-dev libboost-thread-dev libboost-filesystem-dev python-tk && \
      git clone https://github.com/opencv/opencv  && \
      cd opencv && git checkout 3.2.0 && mkdir release && cd release && \
      cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local .. && \
      make -j40 && \
      make install -j40 && \
      cd ../.. && rm -rf opencv && \
      pip install tqdm easydict pyyaml matplotlib scipy ipython h5py numpy tensorflow-gpu==1.0.0 &&\
      tar zxvf /tmp/cuda8-cudnn5.tar -C /usr/local/ &&  rm /tmp/cuda8-cudnn5.tar && \
      rm -rf /tmp/* && \
      apt autoremove && apt autoclean
  ENV  LD_LIBRARY_PATH=/usr/local/cuda8-cudnn5/lib64:$LD_LIBRARY_PATH  CUDNN_VERSION=5.0
```

我的Dockerfile：bit:5000/ws-py3-tf-keras

```dockerfile
FROM bit:5000/deepo_9
RUN pip install keras_applications==1.0.7 --no-deps && \
    pip install keras_preprocessing==1.0.9 --no-deps && \
    pip install h5py==2.9.0
```

```dockerfile
FROM bit:5000/ws-py3-tf-keras
RUN pip install laspy && \
    pip install Pillow==2.2.1 && \ # 下次编译镜像恢复原始版本！！
    pip install seaborn && \ 
    pip install ffmpeg-python && \
    pip install imageio && \
    pip install Flask-SocketIO && \
    pip install eventlet && \
    pip install Shapely && \
    pip install numba && \
    pip install easydict 
```