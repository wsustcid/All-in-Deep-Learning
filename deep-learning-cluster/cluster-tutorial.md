# 集群学习

如有侵权请联系 zzpzkd@mail.ustc.edu.cn

[TOC]

## 0. 预备知识

### 集群与分布式的概念

**单机结构：**

我们自己的一台台式机或者笔记本，或者一个服务器，就是单机结构。但是单机的处理能力有限，比如一台服务器能同时响应1万个用户输出请求，并返回hello world，  如果同时有10万个用户发出请求，则一台服务器就无法处理，因此就需要集群的概念。（注意这种请求都是相同的，即都是**同一个任务**）

**集群：**

集群是一组协同工作的服务器集合，用来提供比单一服务更稳定、更高效、更具扩展性的服务平台。也就是说，多个服务器放一块，就可以称作一个集群。集群中的每台服务器叫做一个”节点“，节点之间可以相互通信。如果每个节点都处理相同的服务，那么这样系统的处理能力就提升了几倍。如对于刚才的10万个输出请求，使用具有10个节点的集群就可以完成。

问题：

如果现在单个请求是从1万个大电影里面里面找出有王宝强得画面，单个服务器耗尽cpu性能，用尽内存也要1个小时才能完成一次查找，你此时增加多少集群节点也不会缩短单个请求得时间。对于这个任务而言，并不是多少个相同任务的重复，你只有一个任务，从1万个电影中找，因此，再多的节点也无能为力。这时就需要分布式处理。

*Remark:*

- *但问题是用户的请求究竟由哪个节点来处理呢？最好能够让此时此刻负载较小的节点来处理，这样使得每个节点的压力都比较平均。要实现这个功能，就需要在所有节点之前增加一个“调度者”的角色，用户的所有请求都先交给它，然后它根据当前所有节点的负载情况，决定将这个请求交给哪个节点处理。这个“调度者”称作负载均衡服务器。*
- *在集群中，同样的服务可以由多个服务实体提供。因此，当一个节点出现故障时，集群的另一个节点，可以自动接管故障节点的资源，从而保证服务持久、不间断地运行。因而集群具有故障自动转移功能。*

这时把这些大电影分给1万个服务器，每个服务器只负责从一个电影里面找王宝强，找完再汇总给一个服务器，就可以极大的缩短单次查找的时间，这就是分布式计算解决的问题之一

**分布式结构：**

分布式结构就是将一个完整的系统，按照业务功能，拆分成一个个独立的子系统，在分布式结构中，每个子系统就被称为“服务”。这些子系统能够独立运行在web容器中，它们之间通过RPC方式通信。

举个例子，假设需要开发一个在线商城。按照微服务的思想，我们需要按照功能模块拆分成多个独立的服务，如：用户服务、产品服务、订单服务、后台管理服务、数据分析服务等等。这一个个服务都是一个个独立的项目，可以独立运行。如果服务之间有依赖关系，那么通过RPC方式调用。

这样的好处有很多：

1. 系统之间的耦合度大大降低，可以独立开发、独立部署、独立测试，系统与系统之间的边界非常明确，排错也变得相当容易，开发效率大大提升。
2. 系统之间的耦合度降低，从而系统更易于扩展。我们可以针对性地扩展某些服务。假设这个商城要搞一次大促，下单量可能会大大提升，因此我们可以针对性地提升订单系统、产品系统的节点数量，而对于后台管理系统、数据分析系统而言，节点数量维持原有水平即可。
3. 服务的复用性更高。比如，当我们将用户系统作为单独的服务后，该公司所有的产品都可以使用该系统作为用户系统，无需重复开发。

回到刚才的任务，我们需要把从1万个电影中查找的任务，分成1万个子任务：子任务为从1个电影中找王宝强，那么，把这个子任务分配给1万个服务器，这就依靠分布式计算实现了1万倍的加速。

**总结：**

- 从单机结构到集群结构，你的代码基本无需要作任何修改，你要做的仅仅是多部署几台服务器，每台服务器上运行相同的代码就行了。但是，当你要从集群结构演进到分布式结构的时候，之前的那套代码就需要发生较大的改动了（将处理1万部电影的程序改成处理1部）
- 集群是个物理形态，只是把同一个业务，部署在多个服务器上
- 分布式是个工作方式，把一个业务分拆多个子业务，部署在不同的服务器上
- 分布式与集群的比喻： 小饭店原来只有一个厨师，切菜洗菜备料炒菜全干。后来客人多了，厨房一个厨师忙不过来，又请了个厨师，两个厨师都能炒一样的菜，这两个厨师的关系是集群。为了让厨师专心炒菜，把菜做到极致，又请了个配菜师负责切菜，备菜，备料，厨师和配菜师的关系是分布式，一个配菜师也忙不过来了，又请了个配菜师，两个配菜师关系是集群

### Linux 文件结构

在Linux中所有的文件都是基于目录的方式存储的。一切都是目录，一切都是文件。

**以Ubuntu系统为例：**我们打开终端后，可以看到如下界面：

<img src=imgs/0_1.png />

每行命令提示符$之前的内容为：`用户名@主机名：当前所在目录`

- 主机名只有一个，因为我们只有一个主机，但用户名可以有多个，因为我们一台主机我们可以建立多个账户；
- 所有linux系统的根目录文件都和上述截图类似，但后期可能存在多个用户，因此各根目录下的文件夹会根据其自身属性为各个用户根据用户名创建不同的文件夹（有的文件是所有用户共享的，则不会创建）
- 比如终端打开后默认`home`目录~，其实是当前用户的home目录，其绝对路径为`/home/ubuntu16/`

```python
/是一切目录的起点，如大树的主干。其它的所有目录都是基于树干的枝条或者枝叶。ubuntu中硬件设备如光驱、软驱、usb设备都将挂载到这颗繁茂的枝干之下，作为文件来管理。

/home: 用户的主目录，在Linux中，每个用户都有一个自己的目录，一般该目录名是以用户的账号命名的。
/media: ubuntu系统挂载的硬盘、usb设备，存放临时读入的文件。
/tmp: 这个目录是用来存放一些临时文件的，所有用户对此目录都有读写权限。

/bin: bin是Binary的缩写。存放系统中最常用的可执行文件（二进制）。
/boot: 这里存放的是Linux内核和系统启动文件，包括Grub、lilo启动器程序。
/dev: dev是Device(设备)的缩写。该目录存放的是linux的外部设备，如硬盘、分区、键盘、鼠标、usb等。
/etc: 这个目录用来存放所有的系统管理所需要的配置文件和子目录，如passwd、hostname等。
/lib: 存放共享的库文件，包含许多被/bin和/sbin中程序使用的库文件。
/lost+found: 这个目录一般情况下是空的，当系统非法关机后，这里就存放了一些零散文件。
/mnt: 作为被挂载的文件系统得挂载点。
/opt: 作为可选文件和程序的存放目录，主要被第三方开发者用来简易安装和卸载他们的软件。
/proc: 这个目录是一个虚拟的目录，它是系统内存的映射，我们可以通过直接访问这个目录来获取系统信息。这里存放所有标志为文件的进程，比较cpuinfo存放cpu当前工作状态的数据。
/root: 该目录为系统管理员，也称作超级权限者的用户主目录。
/sbin: s就是Super User的意思，这里存放的是系统管理员使用的系统管理程序，如系统管理、目录查询等关键命令文件。
/ srv: 存放系统所提供的服务数据。
/sys: 系统设备和文件层次结构，并向用户程序提供详细的内核数据信息。
    
/usr: 存放与系统用户有关的文件和目录。
/usr 目录具体来说：
/usr/X11R6: 存放X-Windows的目录；
/usr/games: 存放着XteamLinux自带的小游戏；
/usr/bin: 用户和管理员的标准命令；
/usr/sbin: 存放root超级用户使用的管理程序；
/usr/doc: Linux技术文档；
/usr/include: 用来存放Linux下开发和编译应用程序所需要的头文件，for c 或者c++；
/usr/lib: 应用程序和程序包的连接库；
/usr/local: 系统管理员安装的应用程序目录；
/usr/man: 帮助文档所在的目录；
/usr/src: Linux开放的源代码；


/var: 长度可变的文件，尤其是些记录数据，如日志文件和打印机文件。
/var/cache: 应用程序缓存目录；
/var/crash: 系统错误信息；
/var/games: 游戏数据；
/var/log: 日志文件；
/var/mail: 电子邮件；
/var/tmp: 临时文件目录；
    
注: ubuntu严格区分大小写和空格，所以Sun和sun是两个不同的文件。
```

### 脚本

**编程语言的分类：**

由于计算机不能理解任何除机器语言以外的语言，所以要把程序员通过高级语言写的程序 翻译 成机器语言，计算机才能执行，这种 “翻译” 的方式有两种，一种是**编译**， 一种是**解释**。由此诞生了两种类型的语言：编译性语言与解释性语言。

编译性语言的代表：C/C++、Pascal/Object Pascal（Delphi）等

解释性语言的代表：JavaScript、VBScript、Perl、Python、Ruby、MATLAB 等

两种类型的语言的区别主要在于翻译的时间点不同：

- 编译性语言在程序执行之前进行翻译，称之为“编译”，把程序编译成机器语言的可执行文件，如.exe文件，以后运行时就不用重新翻译了，直接使用翻译后的结果。即 一次编译，多次使用，所以编译性语言程序的执行效率高。（编译编译型语言的工具我们通常称之为**编译器**）
- 而解释性语言写完后不需要进行编译，在程序运行时才进行翻译，在程序运行时，翻译一句执行一句，并且每次执行都是这样，因此执行效率比较低。（程序运行时翻译解释性语言的工具我们称之为**解释器**）

**脚本语言：**

> 许多脚本语言用来执行一次性任务，尤其是系统管理方面。它可以把服务组件粘合起来，因此被广泛用于GUI创建或者命令行，[操作系统](https://zh.wikipedia.org/wiki/操作系统)通常提供一些默认的脚本语言，即通常所谓shell脚本语言。脚本通常以文本（如[ASCII](https://zh.wikipedia.org/wiki/ASCII)）保存，只在被调用时进行解释或编译。 *- 维基百科定义*

```
shell指的是外壳程序，处于操作系统内核和应用程序之间,是用户和系统交互的界面。也理解成命令解释器。把多个在shell中执行的指令写到一个文本文件中，并指定其解释器，就成为了一个shell脚本。
```

其他解释：

> 其实“脚本语言”与“非脚本语言”并没有语义上，或者执行方式上的区别。它们的区别只在于它们设计的初衷：脚本语言的设计，往往是作为一种临时的“补丁”。它的设计者并没有考虑把它作为一种“通用程序语言”，没有考虑用它构建大型的软件。这些设计者往往没有经过系统的训练，有些甚至连最基本的程序语言概念都没搞清楚。相反，“非脚本”的通用程序语言，往往由经过严格训练的专家甚至一个小组的专家设计，它们从一开头就考虑到了“通用性”，以及在大型工程中的可靠性和可扩展性。
>
> 首先我们来看看“脚本”这个概念是如何产生的。使用 Unix 系统的人都会敲入一些命令，而命令貌似都是“一次性”或者“可抛弃”的。然而不久，人们就发现这些命令其实并不是那么的“一次性”，自己其实一直在重复的敲入类似的命令，所以有人就发明了“脚本”这东西。它的设计初衷是“批量式”的执行命令，你在一个文件里把命令都写进去，然后执行这个文件。可是不久人们就发现，这些命令行其实可以用更加聪明的方法构造，比如定义一些变量，或者根据系统类型的不同执行不同的命令。于是，人们为这脚本语言加入了变量，条件语句，数组，等等构造。“脚本语言”就这样产生了。

例子：

> 假设你经常从网上下东西，全都放在 D 盘那个叫做 downloads 的文件夹里。而你有分类的癖好，每周都要把下载下来的图片放到 pic 文件夹里，pdf 放到 book 文件夹里，mp3 和 wma 文件放到 music 文件夹里。手动分了一年之后你终于厌倦了，于是你打开记事本，写了以下的三行字：      
>
>  copy /Y D:\download\*.jpg D:\pic\        
>  copy /Y D:\download\*.pdf D:\book\        
>  copy /Y D:\download\*.mp3 D:\music\        
>
>  然后把它存成一个叫做 cleanupdownload.bat 的文件。想起来的时候你就双击一下这个文件，然后就发现 download 里的三类文件都被拷到它们该去的地方了。这就是个非常简单的脚本。

从以上叙述我们可以看出，脚本语言应该是解释性语言的一种，但其从起源上应该不算是我们理解的狭义的编程语言（实现某种算法），只是后来大家都相互融合，很多解释性的语言也可以写成脚本的形式，比如python是一种解释性语言，也可以称之为一种脚本语言。

**脚本：**

可直接运行的文本文件，便是脚本，（注意编译性语言编译后的可直接运行的文件是二进制文件，并不是文本文件）。因此脚本本质上是一个程序，只不过严格来讲，脚本通常是由一种称之为脚本语言的很烂的编程语言写出来的。

**python脚本：**

用python语言写的可以直接运行的文本文件，当然就是一个python脚本。

- 如果文件首行没有指定解释器，它就只是一个包含该了python编程语言的文本文件，他不能直接运行，只能通过 `python test1.py` 来运行。
- 如果首行指定了解释器，他就成为了一个python脚本，便可以直接运行该文件。

示例：

```python
# 文件 test1.py
print("hello world")
# 运行： python test.py

# 文件 test2.py
#!/usr/bin/python
print("hello world")
# 添加运行权限：chomd +x test2.py
# 运行： ./test2.py
# 注意：./ 是代表文件的路径：当前目录下，并不是执行指令
```

当Linux执行一个文件的时候，如果发现首行是这样的格式，就会把!后面的内容提取出来拼在你的脚本文件或路径之前，当作实际执行的命令，对这个脚本来说就是`/usr/bin/python ./test.py` 。

- 它代表调用`/usr/bin/`路径下的python的解释器，然后用这个解释器来解释运行这个文件。

- 现在我们手动来看看，`/usr/bin/` 下都有哪些python解释器：

  ```python
  ubuntu16@ubuntu16:/usr/bin$ ls |grep python
  python
  python2
  python2.7
  python3
  python3.5
  ```

  - 因此在`/usr/bin`下我们其实有以上5个python解释器版本可供选择，我们可以任意指定一种解释器来解释运行我们的程序文件。如`#！/usr/bin/env python3.5`

- 但这种方式的缺点是有些操作系统用户并没有将python或其他版本的python装在默认的`/usr/bin`路径下，比如可能在`/usr/local/bin/`下，这样在程序跨平台使用时，解释器找不到，使得脚本不可用。

- 当然我们使用`which python`手动查找我们需要的python按照到了哪个位置，然后对脚本做对应的修改，但我们有更通用的方法。

**更常见的指定python解释器的方式：**

```python
#!/usr/bin/env python
```

这是它代表调用`/usr/bin/`，将后面两个值作为参数。env是个Linux命令，它会根据当前环境变量设置（主要是PATH变量），查找python的安装路径，再调用对应路径下的名为python的解释器，然后用这个解释器来解释运行这个文件。这就避免了将python写死到固定位置，导致python安装到其他可选位置（如/usr/local/bin/）用户安装程序后默认的可执行文件位置 下时脚本不可用的问题。

## 1.如何登录集群

首先要有一个帐号，申请帐号:

- GPU/CPU集群帐号申请方法

- 信息学院的学生将姓名、学号、导师和集群类型信息发邮件给ypb@ustc.edu.cn,并抄送导师，导师邮件确认后即可创建帐号。申请帐号成功后，请加入GPU使用群，修改昵称为姓名_帐号。
- 集群类型指的是要CPU集群还是GPU集群，这里我们申请的是GPU集群

拿到帐号后即可以**登录集群:**

我们要登录的集群的地址：内网地址192.168.9.99，外网SSH连接：202.38.69.241:39099

```
ssh -p 39099 YOUR_USER_NAME@202.38.69.241
```

登录成功后用户名是你的用户名，主机名为gwork，说明当前我们处于gwork节点。

Remark:

- 首次登录会提示修改密码；*如未提示，使用passwd 指令修改密码*

## 2.节点与文件系统管理

根据我的猜测，学院应该是有两种类型的集群设备，一种是由GTX1080 构成的**GPU集群**，另一种是两台配置有V100卡的DGX1服务器构成的**高性能计算集群**。我们主要关注GPU集群的使用。

**GPU集群：**

- 该集群共有34个计算节点（34台服务器），其中 PBS 计算节点 G101-G125 每个结点8块GTX 1080Ti GPU卡，共200块卡；G131-G144有76块（152核心）tesla K80卡。
- 该集群计算节点本身没有安装任何计算软件，他们把常用的软件环境封装在[Docker 容器](http://mccipc.ustc.edu.cn/mediawiki/index.php/Dcoker-images)中。
- 该集群使用torque PBS管理计算job，job以docker方式进行运行，禁止以docker方式以外的方式运行job
- 集群提供了一些经典的用于深度学习的镜像，如果仍不能满足要求，可以根据后续章节要求生产自己的镜像；
- 集群提供了一些数据集，如果用户有自己的数据需要提交到数据集，可以向管理员申请，获得批准后将数据集上传到 /gpub/temp，上传完毕后通知管理员处理，将不会占用自己的空间配额。/gpub/temp只是临时空间，会定期清理，重要数据不可以放在这里长存。



**节点类型：**

- 登录/操作控制台 节点 *gwork* (192.168.9.99/202.38.69.241:39099)，用于外部访问和提交 Torque PBS 任务，目前我们就处于这个节点
- 测试节点 `g101`: 专门用于任务调试，正常运行后将限制调试job运行不超过15分钟；可以从gwork节点 通过`ssh g101` 命令登录进去，任务调试成功后，可以切换到Gwork节点下真正执行任务；
- `gproc`节点：专门用于数据传输与解压缩; 可以在gwork节点直接登录进入，也可以在外部通过网址访问：`ssh -p 37240 username@202.38.69.241`

Remark:

- 该集群**所有节点**共享数据存储，有三个共享mount点，分别是 `/ghome; /gdata; /gpub`

  

**文件系统的组成：**

首先我们先查看该集群的根目录：

```python
[username@gwork /]$ ls
bin             chk_nvidia_output  etc     gdata1  home   lifedat  nfstest  root  srv  usr
boot            data0              fengxy  ghome   lib    media    opt      run   sys  var
chk_gpu_output  dev                gdata   gpub    lib64  mnt      proc     sbin  tmp
```

我们可以发现，该集群和之前提到的Linux文件系统组成类似，Linux中我们经常使用`/home/username/`来存放我们的代码和项目，在`/media/ubuntu16/`下挂载我们的硬盘存放数据集，集群中也是如此。

集群所有结点共享数据存储，有三个共享mount点，分别是：

-  /ghome  所有用户的根文件系统，用于存放代码等重要数据，限额50G
-  /gdata  所有用户的数据区，用于存放job运行过程生成的数据以及结果，用户可读写，暂限额500G。
-  /gpub   公共数据集集区，用户只读，对于一些下载的公共数据集，可以提交管理员转移到这里，将不会占用个人的磁盘限额。

同样的，不同用户的文件被放在了不同的文件夹下，该文件夹名称与用户名相同，每个用户只能访问自己的文件夹，别人的文件夹无权访问，即

- 登录后默认的～路径 完整路径为 `/ghome/username/`
- 存放自己的数据集，完整路径为`/gdata/username/`
- `/gpub`下不区分用户



**文件系统登录与文件传输：**

如何通过客户端连接进入ghome下自己的文件夹和gdata下自己的文件夹：

- 打开默认的文件管理器，选最下面的“连接到服务器”，

- 输入对应文件夹网址

  - sftp://202.38.69.241:39099/ghome/username
  - sftp://202.38.69.241:39099/gdata/wangshuai

- 输入用户名和密码

- 此时对应的文件夹便挂载到了自己的文件系统中：如果是运行代码文件，直接复制粘贴到ghome下的自己的文件夹里就行；如果是数据文件，也是复制粘贴到相应文件夹里

  具体操作结果如下图：

<img src=imgs/sftp.png width=400>



##  3.测试节点测试docker

**1. Docker**

- 为什么要用到Docker？

  计算集群中各计算节点必须保证软件环境同步。系统软件由管理员人工保持同步（手动安装相同的软件）。系统软件只能安装一套，因此经常出现有的人需要的软件版本与系统中软件版本不一致的情况。集群通过 Docker 容器为用户提供多种不同的软件环境。

- Docker是干什么的？

  Docker 容器可以看作一个与主系统隔离的虚拟机环境。每个用户启动的容器都是从基础镜像生成的一个全新副本，互不干扰。根据使用的基础镜像的不同，有的容器启动后里面安装了 CentOS 系统 + CUDA8，另一些容器安装了 Ubuntu + CUDA9 + TensorFlow，你可以根据自己的需求选择对应的基础镜像然后启动一个容器作为自己的基础环境。

- 区分镜像和容器的概念：镜像是已经配置好的一些虚拟环境，容器是用户申请的工作环境，是系统分配的一个基于某一个镜像的虚拟工作环境，同一个镜像可以创建多个容器，在容器进行相应的操作并不会影响底层镜像，（一般用作调试环境，调试成功后再通过Dockerfile修改底层镜像），你在内部对系统所做的任何操作都将在系统退出后丢失，但对用户根目录下（/ghome/username）的文件操作将不会丢失。

- 注意用户只能在计算节点 G101~G125 上启动 Docker 容器。G101 是测试节点，在此节点测试自己的程序，成功运行后，切换到gwork节点通过 Torque PBS 提交任务脚本。

**2. Start docker**

**登录测试节点：**

```
ssh g101
```

执行该指令后，可以看到主机名变成了G101，说明我们已经处于G101节点。

**相关查看指令：**

```python
# 查看可用的相关镜像
sudo docker images
# 查看正在运行的容器（UP 状态）
sudo docker ps
# 查看所有容器（包括已经关闭的）
sudo docker ps -a

# 查看用户有权使用的命令
sudo -l
"""
用户 wangshuai 可以在 G101 上运行以下命令：
(root) NOPASSWD: 
/usr/bin/docker images 
/usr/bin/docker ps*
/usr/bin/docker inspect*
/usr/bin/nvidia-docker run --rm -u [17]*
# 重新进入已打开的容器
/usr/bin/docker attach *
/usr/bin/docker start *
/usr/bin/nvidia-docker attach *
/usr/bin/docker stop *
"""
# 可以看到，依然保留了用户执行nvidia-docker的权利，只是强制加两个参数--rm（退出后即销毁容器） 和 -u，必须以此命令开头：
sudo nvidia-docker run --rm -u <your-id>
# 示例：
sudo nvidia-docker run --rm -u 17 -it bit:5000/deepo
# 更多用法学习nvidia docker的使用后再测试
```

**启动容器：**

```python
# 从bit:5000/deepo镜像启动一个容器并开启一个交互式终端（-it）
startdocker -u "-it" -c /bin/bash bit:5000/deepo
```

- startdocker是对nvidia-docker的再一次封装，主要为了配合torque pbs和避免root权限管理漏洞。
- bit:5000/deepo是一个集中了几乎所有深度学习框架的一个docker 镜像，更新版本的镜像为bit:5000/deepo_9
- 使用上述命令将进入docker container内部，你在内部对系统所做的任何操作都将在系统退出后丢失，但对用户根目录下（/ghome/<username>）的文件操作将不会丢失。 此容器也是退出后即销毁，如果不想销毁此容器，只有保持终端开启或强制直接关闭终端。
- 容器开启后系统处于根目录(/)下，系统默认挂载了整个用户根目录(/ghome/wangsuai)

**在容器中挂载更多目录:**

- 刚才这样使用 startdocker，你会发现在容器里只能访问外面的 `/ghome/用户名` 目录。如果要访问其他目录，需要加额外的参数，例如：

  ```python
  # 把 /gdata/用户名 和 /gpub 下的 ImageNet-Caffe 数据库挂载进容器中
  # 启动测试用的 bash 命令行
  startdocker -u "-it -v /gdata/用户名:/gdata/用户名 -v /gpub/ImageNet-Caffe:/gpub/ImageNet-Caffe" -c /bin/bash bit:5000/deepo
  
  # 直接在容器中运行 /ghome/用户名/run.sh（注意脚本必须有可执行权限）
  startdocker -u "-v /gdata/用户名:/gdata/用户名 -v /gpub/ImageNet-Caffe:/gpub/ImageNet-Caffe" -s /ghome/用户名/run.sh bit:5000/deepo
      
  # 如果需要挂载更多的目录，在 `-u` 后面的引号中增加新的 `-v /XXX:/XXX` 参数就行了（注意路径要用绝对路径）
  ```

- 另外，容器启动时工作目录是在容器内的根目录。你的程序里若使用的相对路径可能会出现无权限写文件的错误。可以在 `-u` 后面的引号里添加 `-w 工作目录` 设置容器中的初始工作目录（当然工作目录必须挂载到容器中）：

  ```python
  # 在容器中运行 /ghome/用户名/run.sh
  # 工作目录和容器外的当前工作目录一致（PWD 环境变量是当前工作目录路径）
  startdocker -u "-v /gdata/用户名:/gdata/用户名 -w ${PWD}" -s /ghome/用户名/run.sh bit:5000/deepo
  ```

  

**在容器中运行程序（非交互式，用于任务提交）：**

```python
startdocker  -D <my-data-path> -P <my-proc-config-path> -s <my-script-file> bit:5000/deepo
    
# 如 startdocker -s /ghome/username/mytest.py bit:5000/deepo
```

- -s 参数后的my-script-file 可以是shell脚本或python脚本，需要在第一行加解释器

   ```python
   shell脚本需要加：#!/bin/bash
   python脚本需要加：#!/usr/local/bin/python 
   由于镜像中的解释器位置并不一定固定，可以使用下面命令获得：
   startdocker  -c "which python" bit:5000/deepo
   
   -s还有一个替代命令-c，-c 是执行命令行，py文件可以不是脚本，和-s 的区别是本参数不会处理相对路径，只解释为命令行，和-s 两者中只能出现一个，如
   startdocker -c "python /ghome/<username>/mytest.py"  bit:5000/deepo
   ```

- -P 参数用来指定代码或配置文件所在的主目录，通常取配置文件和可执行文件的公共父目录

  ```python
  如配置文件是/gdata/userx/proc1/conf/my.conf，
   可执行文件/gdata/userx/proc1/shell/my.sh
  则可以取 -P /gdata/userx/proc1 
  那么指定脚本文件时就是 -s shell/my.sh 
  如果代码就在自己的home目录下，则此参数可以不用设置。
  ```

- -D 参数用来指定数据所在目录

  ```python
  如可以是 -D /gdata/userx/proj1
  ```

- 无论是-P 还是-D 参数指定的目录都是使用docker的-v参数进行了卷的挂载，如果两个挂载点依然不够用（比如我想要运行的程序存在两个文件夹下），或者挂载后容器内外目录结构不一样，可以利用提供的-u 参数进行扩展，

  ```python
  比如需要将/gdata/userx 挂载到内部的/userx,可以这样设置-u:
  -u "-v /gdata/userx:/userx"
  针对于两个挂载点不够用的情况，如果需要挂载更多的目录，在 `-u` 后面的引号中增加新的 `-v /XXX:/XXX` 参数就行了（注意路径要用绝对路径）
  -u 其实支持其他任意形式的docker参数的传入，如 -u "--ipc=host"等，
  另外，-D和-P参数不宜使用太高级别的目录如/gdata,  /ghome等，至少应该到用户的所在目录层次，如/gdata/userx等。
  ```

**例子：**

```python
startdocker -u "--ipc=host -v /gpub/leftImg8bit_sequence:/gpub/leftImg8bit_sequence --shm-size 8G" \
-D "/gdata/jinlk" -s "/ghome/jinlk/VSS/DVSNet_pytorch/scripts/train_feat2_df_with_scale_4.sh" bit:5000/deepo_9
```



**停止或重启容器:**

```python
# 停止正在运行的容器
sudo docker stop CONTAINER_ID/CONTAINER_NAME
# 重启已经关闭的容器
sudo docker start CONTAINER_ID/CONTAINER_NAME
```



## 4.通过 Torque PBS 提交任务

如果编写的脚本在测试节点上运行程序没有问题，就可以到正式节点上提交任务运行了

**！！注意只能在 gwork 节点上提交任务，不能在测试节点提交！**

一共有下面几个步骤

**1. 编写任务脚本：**

Torque 管理系统不能直接提交二进制可执行文件，需要编写一个文本的脚本文件，来描述相关参数情况。下面是一个示例脚本文件 myjob.pbs ：

```bash
#PBS    -N  testjob
#PBS    -o  /ghome/<username>/$PBS_JOBID.out
#PBS    -e  /ghome/<username>/$PBS_JOBID.err
#PBS    -l nodes=1:gpus=1:S
#PBS    -r y
#PBS    -q mcc
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

解释说明

- 脚本文件中定义的参数默认是以#PBS 开头的

- -N 定义的是 job 名称，可以随意

- -o 定义程序运行的标准输出文件，如程序中 printf 打印信息,相当于 stdout，注意<username>要改成自己的用户名，所有程序运行的结果会存在那个文件中

- -e 定义程序运行时的错误输出文件,相当于 stderr，注意<username>要改成自己的用户名，所有程序输出的错误信息会存在那个文件中

- -l 定义了申请的结点数和 gpus 数量

  nodes=1 代表一个结点，一般申请一个结点，除非用 mpi 并行作业

  gpus=1 定义了申请的 GPU 数量,根据应用实际使用的 gpu数量来确定

  S 表示 job 类型，

  - [ ] 使用单核的 job 必须加参数 S,如：#PBS -l nodes=1:gpus=1:S
  - [ ] 使用双核的 job 必须加参数 D,如：#PBS -l nodes=1:gpus=2:D
  

队列系统的默认 job 请求时间是一周，如果运行的 job 时间估计会超过，则可以使用下面的参数：表示请求 300 小时的 job 时间

```
  #PBS -l nodes=1:gpus=1:S,walltime=300:00:00
```

- -r y 指明作业是否可运行，y 为可运行，n 为不可运行

- -q 表示排在哪个队列中，默认是排在batch队列中，后面加上mcc表示排在mcc队列中

- 后面的 cat 和 echo 信息是打印出时间、路径、运行节点及GPU 分配情况等信息，便于调试

- 最后一行是执行自己程序的命令，详细解释请看上节，更常用的是

  ```python
  startdocker -u "-v /gdata/$USER:/gdata/$USER -w /ghome/$USER" -s /ghome/$USER/run_in_docker.py bit:5000/deepo
  ```



**2. 提交任务**

用下面的指令提交任务，注意要在gwork节点提交任务

```
qsub myjob.pbs
```

返回的结果是自己的任务编号

**3. 其他指令**

![a6](imgs/a10.png)




## 5.自定义docker 镜像文件

集群系统已经提供了大多数常用的深度学习框架，deepo就是MIT创建的一个包含了几乎全部主流深度学习框架的镜像。请仔细查看集群上所有的镜像，如果仍然不能满足用户需求， 则用户可以申请创建自己的镜像，向管理员提出申请有两种方式：

- 第一种方式是提供docker hub上的镜像名称，由管理员下载到本地仓库。

  - docker hub上包含许多别人已经制作好的镜像，可以直接下载后使用。你也可以将自己经常使用的环境制作成镜像后上传到docker hub，由管理员下载到本地。--疑问：我们在自己的电脑使用docker之前，已经安装好了显卡驱动，制作自己的镜像时，进行的是和硬件平台无关的操作：如安装某一版本的cuda，安装tensorflow等，因此，应该不用担心你自己在你电脑上制作的镜像装到集群上不能用，除非集群上的显卡驱动版本过低，你要使用的cuda版本和此显卡驱动不兼容。另外，应该有英伟达官方出的基于自己的显卡安装好了tensorflow等环境的相关镜像，我们应该避免自己完全从零开始制作镜像。--具体步骤可以研究nvidia-docker之后进行尝试。

-  第二种方式是基于集群上的镜像使用Dockerfile build生成，申请时需要给出基于的镜像名称和要增加的内容。

  ```
  1. 为了方便Dockerfile的编写，首先告诉管理员你要基于哪个镜像进行修改，请管理员用root身份给你开启一个基于此镜像的container
  2. 用户用 sudo docker attach <容器名> 命令连接到container内部，边安装/配置环境，边做步骤记录，以便于形成Dockerfile
  3. 申请者需要在自己的根目录下面建立一个dockertmp子目录，如/ghome/userx/dokcertmp,在里面编辑一个Dockerfile的文本文件，注意D大写，build过程中需要用到的文件都copy到此目录下，因为在镜像build过程中是不支持挂着卷或文件系统的。（如果你只是进行一些在线安装操作，不需要用到某些自定义的文件，应该只提供一个Dockerfile就可以了）
  3.1 因此在build过程中如果需要使用文件系统中已经有的文件可以使用COPY或ADD命令，COPY和ADD命令不支持绝对路径，只能放在build目录（/ghome/userx/dokcertmp）下面，例如，需要安装my.zip 文件，可以使用 COPY  ./my.zip /userx 事先放在build目录的my.zip则会在镜像build过程中被复制到镜像内的/userx目录下。
  4. 完成编写Dockerfile以后，提交管理员build生成新的镜像。
  5. 需要注意的时，原镜像内部一般都有很多关于已安装软件的版本信息的ENV变量，如果用户新生成的镜像做了软件修改，务必同时修改ENV，以保证ENV和实际软件版本一致，便于后期使用inspect命令查看。
  ```

- 下面是一个Dockerfile示例(cuda8-cudnn5.tar需事先copy到/ghome/userx/dockertmp下)：

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

bit:5000/deep_9

```
absl-py          0.2.2                 
astor            0.6.2                 
backcall         0.1.0                 
bleach           1.5.0                 
certifi          2018.4.16             
chainer          4.0.0                 
chardet          3.0.4                 
cntk-gpu         2.5.1                 
cupy             4.0.0                 
cycler           0.10.0                
Cython           0.28.2                
decorator        4.3.0                 
dm-sonnet        1.20                  
fastrlock        0.3                   
filelock         3.0.4                 
future           0.16.0                
gast             0.2.0                 
graphviz         0.8.3                 
grpcio           1.12.0                
h5py             2.7.1                 
html5lib         0.9999999             
idna             2.6                   
ipython          6.4.0                 
ipython-genutils 0.2.0                 
jedi             0.12.0                
Keras            2.1.6                 
kiwisolver       1.0.1                 
Lasagne          0.2.dev1              
leveldb          0.194                 
Mako             1.0.7                 
Markdown         2.6.11                
MarkupSafe       1.0                   
matplotlib       2.2.2                 
mxnet-cu90       1.2.0                 
networkx         2.1                   
nose             1.3.7                 
numpy            1.14.3                
pandas           0.23.0                
parso            0.2.1                 
pexpect          4.5.0                 
pickleshare      0.7.4                 
Pillow           5.1.0                 
pip              10.0.1                
prompt-toolkit   1.0.15                
protobuf         3.5.2.post1           
ptyprocess       0.5.2                 
pycurl           7.43.0                
Pygments         2.2.0                 
pygobject        3.20.0                
pygpu            0.7.6                 
pyparsing        2.2.0                 
python-apt       1.1.0b1+ubuntu0.16.4.1
python-dateutil  2.7.3                 
python-gflags    3.1.2                 
pytz             2018.4                
PyWavelets       0.5.2                 
PyYAML           3.12                  
requests         2.18.4                
scikit-image     0.13.1                
scikit-learn     0.19.1                
scipy            1.1.0                 
setuptools       39.2.0                
simplegeneric    0.8.1                 
six              1.11.0                
tensorboard      1.8.0                 
tensorflow-gpu   1.8.0                 
termcolor        1.1.0                 
Theano           1.0.1                 
torch            0.4.0                 
torchvision      0.2.1                 
traitlets        4.3.2                 
urllib3          1.22                  
wcwidth          0.1.7                 
Werkzeug         0.14.1                
wheel            0.31.1   
```

```
pip install keras_applications==1.0.7 --no-deps
pip install keras_preprocessing==1.0.9 --no-deps
pip install h5py==2.9.0
```





安装putty

<https://blog.csdn.net/skypeGNU/article/details/11655713>

启用putty keepalive

putty -> Connection -> Seconds between keepalives ( 0 to turn off ), 默认为0, 改为60



## FAQ

- 为什么在gwork 控制台不能运行docker命令？

```
gwork 是操作控制台，并没有GPU卡 所以也不需要启用docker,要调试代码，请从gwork ssh到G101。
```

- 如果没有任务运行，处于gwork控制台的时间有限制吗？是不是超过一定时间会被退出？

```
没有刻意限制，但某些网络设备若检测不到终端有通信，就会清理掉session 连接，可以启用ssh client端软件的anti-idle功能避免超时退出。
```

- 使用调试命令打开自己的一个container后，会提示i have no name，会有问题吗？

```
这是因为新创建的container中没有对应于系统的帐号引起，不会影响任何使用，是正常现象。
```

- 提交job后，startdocker返回错误“exec format error”是什么原因？

```
是因为没有在script脚本中指定解释器，如/bin/bash,/usr/local/bin/python等，参看前面【3.使用】部分内容。
```

- 提交job后，startdocker返回错误“exec permission denied”是什么原因？

```
是因为提供的脚本不具有可执行权限，可以使用命令chmod +x <scriptfile> 增加执行权。
```

- 在用pyTorch 时老是报错：unable to write to file ,网上说用docker时需要添加 --ipc=host 这条指令, 如何做？

```
在startdocker时 使用 -u "--ipc=host".
```

- 如何查看结点的GPU/CPU和内存等使用情况？

```
使用chk_gpuused <结点名>查看GPU使用情况，使用sudo chk_res <结点名> <用户名> 查看cpu和内存等资源使用情况。
```

- 使用tensorflow为什么CPU利用率非常高？

```
可能是配置了inter_op_parallelism_threads和intra_op_parallelism_threads这两个参数，这两个参数会提高CPU的并行利用率，会导致CPU利用率高,每个job的CPU利用率最好不要超过200%。
```

- 如果自己的python程序需要用到镜像里没有的、比较小的package，是否需要构建新的镜像？

```
不需要构建镜像。以numpy为例：
1. 下载numpy的源码（一般能从个人PC上拷贝），放入服务器上的某个目录，比如/gdata/xxx/pylib
2. 挂载目录，-v /gdata/xxx/pylib:/data/pylib
3. 将挂载后的目录加入python import的搜索路径，两个方法
   a. 指定容器的PYTHONPATH, 在startdocker的-u里面加入-e PYTHONPATH=/data/pylib，此时/data/pylib的
      优先级可能次于系统自带路径（没测过），import numpy 会导入系统的numpy
   b. 在程序的一开始加入下面两行代码
      import sys
      sys.path.insert(1, '/data/pylib')
      1表示/data/pylib的优先级仅次于程序所在目录
      此时/data/pylib的优先级高于系统自带路径，import numpy 会导入/data/pylib的numpy
```

- 有时候代码使用的显存会超出实际显存大小，能否加以限制？

```
显式控制框架对显存申请其实很简单，比如在tf里使用下面的语句就可以避免过渡使用显存。
tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
```

- 如何知道自己的代码有没引起kernel错误？

```
常见比较严重的kernel错误是：SLUB: Unable to allocate memory on node -1 (gfp=0x20)
该错误很可能会引起结点重启，可以使用命令k_log 查看5分钟内有没有输出kernel的log。
如发现该错误，请及时通知管理员。
```

- 某些镜像内的时间和当前时间不一致，log中显示的时间也就不对，怎么解决？

```
是因为镜像内时区设置有问题，可以通过在镜像内运行的shell中设置一个环境变量实现，
export TZ=Asia/Shanghai
```

- 运行docker时会报 dial unix /var/run/docker.sock: connect: permission denied错误，是什么原因？

```
可能是帐号从docker组中脱离，联系管理员加入docker group。
```

- 在docker中使用python调用getpass.getuser()函数时报错：KeyError:'getpwuid():uid not found:****。

```
可以通过申明环境变量LOGNAME，USER等来禁止调用pwd.getpwuid(os.getuid())，因此在~/.bashrc中加入两行:
export LOGNAME=coder
export USER=coder
然后source ~/.bashrc即可解决问题。
```

## 参考文献

<http://mccipc.ustc.edu.cn/mediawiki/index.php/Main_Page>

<https://github.com/jinlukang1/issue-Notebook/issues/15>

<http://mccipc.ustc.edu.cn/mediawiki/index.php/Gpu-cluster-manual>

<https://wmg.ustc.edu.cn/wiki/index.php/%E9%9B%86%E7%BE%A4:%E4%BF%A1%E9%99%A2_GPU_%E9%9B%86%E7%BE%A4#.E6.96.87.E4.BB.B6.E7.B3.BB.E7.BB.9F.E7.BB.93.E6.9E.84>