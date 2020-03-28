## AutoML：无人驾驶机器学习模型设计自动化

原创： 雷锋字幕组 [AI研习社](javascript:void(0);) *3月10日*

![img](https://mmbiz.qpic.cn/mmbiz_png/bicdMLzImlibR5HaSmNUaVbMP6LudaUQibuaGL9UVR1WRKaoCnywCOj23ZrTgmWV2Rs39PicCpyonohs1TKjbCZ8rQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

> 本文为 AI 研习社编译的技术博客，原标题 ：
>
> AutoML: Automating the design of machine learning models for autonomous driving
>
> 作者 | *Waymo TeamFollow*
>
> 翻译 | ciky奇、Ophria、谢玄xx           
>
> 校对 | 邓普斯•杰弗        审核 | 酱番梨       整理 | 立鱼王
>
> 原文链接：
>
> https://medium.com/waymo/automl-automating-the-design-of-machine-learning-models-for-autonomous-driving-141a5583ec2a
>
> 注：本文的相关链接请访问文末二维码



作者: Shuyang Cheng and Gabriel Bender*

在Waymo，机器学习几乎在自动驾驶系统的每个部分都起着关键作用。它可以让汽车看到周围环境，感知和了解世界，预测其他人的行为方式，并决定他们的下一步行动。
感知：我们的系统采用神经网络的组合，以便我们的车辆能够识别传感器数据、识别物体并随着时间的推移跟踪它们，以便它能够深入了解周围的世界。这些神经网络的构建通常是一项耗时的任务;优化神经网络架构以实现在自动驾驶汽车上运行所需的质量和速度是一个复杂的微调过程，我们的工程师要完成一项新任务可能要花费数月时间。
现在，通过与Brain团队的谷歌AI研究人员合作，我们将前沿研究付诸实践，用来自动生成神经网络。更重要的是，这些最先进的神经网络比工程师手动微调的质量更高，速度更快。
为了将我们的自动驾驶技术带到不同的城市和环境，我们需要以极快的速度优化我们的模型以适应不同的场景。AutoML使我们能够做到这一点，高效，持续地提供大量的ML解决方案。



##    **传输学习：使用现有的AutoML架构**

我们的合作始于一个简单的问题：AutoML能否为汽车生成高质量和低延迟的神经网络？

“质量”可以衡量神经网络产生结果的准确性。“延迟”可以测量网络提供答案的速度，也称为“推理时间”。由于驾驶是一项活动，要求我们的车辆使用实时反馈并考虑到我们系统的安全--关键性质，因此我们的神经网络需要低延迟运行。大多数直接在我们车上运行的网络都可以在不到10毫秒的时间内提供结果，这比在数千台服务器上运行的数据中心部署的许多网络更快。

在他们最初的AutoML论文[1]中，我们的Google AI同事能够自动探索出超过12,000种架构来解决CIFAR-10这种经典图像识别任务：将一张小图像识别为10个类别中的一种，如一辆汽车，一架飞机，一只狗等。在后续的一篇文章[2]中，他们发现了一系列神经网络构建块，称为NAS单元，它可自动构建出比手工制作的用于CIFAR-10以及相似任务更好的网络。通过这次合作，我们的研究人员决定使用这些单元自动构建用于自动驾驶特定任务的新模型，从而将在CIFAR-10上学习到的知识迁移到我们领域中。我们的第一个实验是使用语义分割任务：将LiDAR点云中的每个点识别为汽车，行人，树等。

![img](https://mmbiz.qpic.cn/mmbiz_png/bicdMLzImlibQBtwm3tZ61JYUet7PKs6ADHvuujZEf4mOW7wjhsSicic8SNl35T5YicsoIIb0orU2TAA0S9QC0x4Tcg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

One example of a NAS cell. This cell processes inputs from the two previous layers in a neural net.

为此，我们的研究人员搭建了一种自动搜索算法，用于在卷积网络架构（CNN）中探索数百种不同NAS单元的组合，为我们的LiDAR分割任务训练和评估模型。当我们的工程师手工对这些网络进行微调时，他们只能探索有限数量的架构，但是通过搭建自动搜索算法，我们可以自动探索数百种架构。我们发现模型较之前手工制作的模型有两方面的改进：

- 有些在相似质量情况下具有显著低延迟;
- 其他在相似延迟情况下具有更高的质量；

鉴于此初步的成功，我们将相同的搜索算法应用于与交通车道检测和定位相关的两个附加任务中。迁移学习技术也适用于这些任务，我们能在汽车上部署三个新训练和改进的神经网络。



##    **端到端搜索：从头开始搜索新架构**

我们受初步实验结果鼓舞，因此我们决定进一步更广泛地搜索可以提供更好结果的全新架构。不将自己限制于已经发现的NAS单元中，我们可以直接考虑那些有严格延迟性要求考虑的架构。

进行端到端搜索通常需要手动探索数千种架构，这会带来很大计算成本。探索单一架构需要在具有多GPU卡的数据中心计算机上进行数天训练，这意味着需要数千天的计算才能搜索这个单一任务。相反，我们设计了一个代理任务：减轻LiDAR分割任务，使任务可以在几个小时内解决。

团队必须克服的一个挑战是找到一个与原始分割任务足够相似的代理任务。在确定该任务的体系结构质量与原始任务的体系结构质量之间的良好相关性之前，我们尝试了几种代理任务设计。然后我们启动了一个类似于原始AutoML论文的搜索，但现在在代理任务：一个代理端到端的搜索。这是首次将此概念应用于激光雷达数据。

![img](https://mmbiz.qpic.cn/mmbiz_png/bicdMLzImlibQBtwm3tZ61JYUet7PKs6ADU5liaVFuff87yDGQmSaeupjXPLia0vHDjJic7ibkgVNiaFSyibtZkLcH3hUQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

代理端到端搜索：在缩小的代理任务上探索数千种架构，将100种最佳架构应用于原始任务，验证和部署汽车上最好的架构。

我们使用了几种搜索算法，对质量和延迟进行了优化，因为这对车辆至关重要。我们观察了不同类型的CNN架构，并使用了不同的搜索策略，例如随机搜索和强化学习，我们能够为代理任务探索超过10000种不同的架构。通过使用代理任务，在一个Google TPU集群上花费一年的计算时间只花了两周。我们发现了比我们刚转移NAS细胞时更好的网络：

- 神经网络具有20-30%的低潜伏期和相同质量的结果。
- 神经网络的质量更高，错误率低至8-10%，与以前的体系结构具有相同的延迟。

![img](https://mmbiz.qpic.cn/mmbiz_png/bicdMLzImlibQBtwm3tZ61JYUet7PKs6ADEGfTzOeJfB8o6THwzDXXibaSJHO9rtnsvR5ccqS7YhWxZh8JbSCpGuA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![img](https://mmbiz.qpic.cn/mmbiz_png/bicdMLzImlibQBtwm3tZ61JYUet7PKs6ADgpUic3hHT53bcb5FoMG2ia9nSlZO8y5JjichQ8LLBEY5cIhW7SIBugkRg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

1）第一个图显示了在一组简单的架构上随机搜索发现的大约4000个架构。每一点都是经过培训和评估的体系结构。实线标记了不同推理时间约束下的最佳体系结构。红点表示使用传输学习构建的网络的延迟和性能。在这个随机搜索中，网络不如转移学习中的网络好。

2）在第二个图中，黄色和蓝色点显示了其他两种搜索算法的结果。黄色的是对一组精炼的架构的随机搜索。Blue One使用了如[1]所示的强化学习，探索了6000多个架构。结果最好。这两个额外的搜索发现网络比转移学习的网络要好得多。

搜索中发现的一些架构显示了卷积、池和反卷积操作的创造性组合，如下图所示。这些架构最终非常适合我们最初的激光雷达分割任务，并将部署在Waymo的自动驾驶车辆上。

![img](https://mmbiz.qpic.cn/mmbiz_png/bicdMLzImlibQBtwm3tZ61JYUet7PKs6ADruhpNpWQqko3iaEBOC7mG70kC75yCecdFUVqenRxsBvv5v0YskTkHqQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

代理端到端搜索发现的一种神经网络结构。



##    **下一步是什么**

我们的汽车试验才刚刚开始。对于我们的激光雷达分割任务，传输学习和代理端到端搜索都提供了优于手工制作的网络。我们现在也有机会将这些机制应用于新类型的任务，这可以改善许多其他神经网络。

这一发展为我们未来的ML工作开辟了新的令人兴奋的途径，并将提高我们的自驱动技术的性能和能力。我们期待着继续我们的工作与谷歌人工智能，所以请继续关注更多！



##   **参考**

[ 1 ] Barret Zoph and Quoc V. Le. 搜索的神经结构与强化学习。ICLR，2017年。

[ 2 ] Barret Zoph, Vijay Vasudevan, Jonathon Shlens, Quoc V. Le, 学习可扩展图像识别的可转移体系结构。CVPR，2018年。



##   ***致谢**

Waymo和Google之间的合作是由Waymo的Matthieu Devin和Google的Quoc Le发起和赞助的。这项工作由Waymo的Shuyang Cheng和Google的Gabriel Bender和Pieter Jan Kindermans共同完成。多谢维希·提鲁马拉什蒂的支持。

![img](https://mmbiz.qpic.cn/mmbiz_png/bicdMLzImlibQBtwm3tZ61JYUet7PKs6ADsHkYQ813wUGMnMUwsc3EVBfUkgd92yWVoS55PLfdfaqu4V2ng8wibWQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

Waymo和Google团队成员（左起）：Gabriel Bender、Shuyang Cheng、Matthieu Devin和Quoc Le

想要继续查看该篇文章相关链接和参考文献？

点击底部【阅读原文】或长按下方地址/二维码访问：

https://ai.yanxishe.com/page/TextTranslation/1441

![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

[![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)](https://mp.weixin.qq.com/s?__biz=MjM5ODU3OTIyOA==&mid=2650675473&idx=1&sn=aa980aa3fd0084f004d5525a3e6d436c&chksm=bec2246289b5ad7479602ee017e215f8b356f3b211c40a78093a3b2962309615d322f5669d38&mpshare=1&scene=24&srcid=&pass_ticket=L9MmezP0euAiloYx0ZpV0zn%2FH1NZipsFj7Qrzod8QKwih5XAiPWVLPugpr7yoxrL)

[阅读原文](https://mp.weixin.qq.com/s?__biz=MjM5ODU3OTIyOA==&mid=2650675473&idx=1&sn=aa980aa3fd0084f004d5525a3e6d436c&chksm=bec2246289b5ad7479602ee017e215f8b356f3b211c40a78093a3b2962309615d322f5669d38&mpshare=1&scene=24&srcid=&pass_ticket=L9MmezP0euAiloYx0ZpV0zn%2FH1NZipsFj7Qrzod8QKwih5XAiPWVLPugpr7yoxrL##)



<



https://medium.com/waymo/automl-automating-the-design-of-machine-learning-models-for-autonomous-driving-141a5583ec2a>



[Homepage](https://medium.com/)

[Sign in](https://medium.com/m/signin?redirect=https%3A%2F%2Fmedium.com%2Fwaymo%2Fautoml-automating-the-design-of-machine-learning-models-for-autonomous-driving-141a5583ec2a&source=--------------------------nav_reg&operation=login)[Get started](https://medium.com/m/signin?redirect=https%3A%2F%2Fmedium.com%2Fwaymo%2Fautoml-automating-the-design-of-machine-learning-models-for-autonomous-driving-141a5583ec2a&source=--------------------------nav_reg&operation=register)

[![Waymo](https://cdn-images-1.medium.com/letterbox/112/72/50/50/1*0LFPBnvSYQE3mQJti04cAg.png?source=logoAvatar-lo_Y5fGDgoWbF9c---7075a35566d9)](https://medium.com/waymo?source=logo-lo_Y5fGDgoWbF9c---7075a35566d9)

- [COMPANY NEWS](https://medium.com/waymo/company/home)
- [TECHNOLOGY](https://medium.com/waymo/technology/home)
- [RIDER EXPERIENCE](https://medium.com/waymo/riderexperience/home)
- [LIFE AT WAYMO](https://medium.com/waymo/lifeatwaymo/home)
- 

- [RIDE WITH US](http://www.waymo.com/apply)



# **AutoML: Automating the design of machine learning models for autonomous driving**

[![Go to the profile of Waymo Team](https://cdn-images-1.medium.com/fit/c/100/100/1*GqcDSHJfweuFsivIcIHkHA.png)](https://medium.com/@waymo?source=post_header_lockup)

[Waymo Team](https://medium.com/@waymo)Follow

Jan 15

By: Shuyang Cheng and Gabriel Bender*

At Waymo, machine learning plays a key role in nearly every part of our self-driving system. It helps our cars see their surroundings, make sense of the world, predict how others will behave, and decide their next best move.

Take perception: our system employs a combination of neural nets that enables our vehicles to interpret sensor data to identify objects and track them over time so it can have a deep understanding of the world around it. The creation of these neural nets is often a time-consuming task; optimizing neural net architectures to achieve both the quality and speed needed to run on our self-driving cars is a complex process of fine-tuning that can take our engineers months for a new task.

Now, through a collaboration with Google AI researchers from the Brain team, we’re putting cutting-edge research into practice to automatically generate neural nets. What’s more, these state-of-the-art neural nets are higher quality and quicker than the ones manually fine-tuned by engineers.

To bring our self-driving technology to different cities and environments, we will need to optimize our models for different scenarios at a great velocity. AutoML enables us to do just that, providing a large set of ML solutions efficiently and continuously.

**Transfer Learning: Using existing AutoML architectures**

Our collaboration started out with a simple question: could AutoML generate a high quality and low latency neural net for the car?

*Quality* measures the accuracy of the answers produced by the neural net. *Latency* measures how fast the net provides its answers, which is also called the *inference time*. Since driving is an activity that requires our vehicles to use real-time answers and given the safety-critical nature of our system, our neural nets need to operate with low latency. Most of our nets that run directly on our vehicles provide results in less than 10ms, which is quicker than many nets deployed in data centers that run on thousands of servers.

In their [original AutoML paper](https://arxiv.org/abs/1611.01578)[1], our Google AI colleagues were able to automatically explore more than 12,000 architectures to solve the classic image recognition task of CIFAR-10: identify a small image as representative of one of ten categories, such as a car, a plane, a dog, etc. In a [follow-up paper](https://arxiv.org/pdf/1707.07012.pdf)[2], they discovered a family of neural net building blocks, called *NAS cells,* that could be composed to automatically build better than hand-crafted nets for CIFAR-10 and similar tasks. With this collaboration, our researchers decided to use these cells to automatically build new models for tasks specific to self-driving, thus transferring what was learned on CIFAR-10 to our field. Our first experiment was with a *semantic segmentation* task: identify each point in a LiDAR point cloud as either a car, a pedestrian, a tree, etc.



![img](https://cdn-images-1.medium.com/max/1600/1*CT9kBawjAO2oRN2OGc7MuQ.png)

*One example of a NAS cell. This cell processes inputs from the two previous layers in a neural net.*

To do this, our researchers set up an automatic search algorithm to explore hundreds of different NAS cell combinations within a convolutional net architecture (CNN), training and evaluating models for our LiDAR segmentation task. When our engineers fine-tune these nets by hand, they can only explore a limited amount of architectures, but with this method, we automatically explored hundreds. We found models that improved the previously hand-crafted ones in two ways:

- Some had a significantly lower latency with a similar quality.
- Others had an even higher quality with a similar latency.

Given this initial success, we applied the same search algorithm to two additional tasks related to the detection and localization of traffic lanes. The transfer learning technique also worked for these tasks, and we were able to deploy three newly-trained and improved neural nets on the car.

**End-to-End Search: Searching for new architectures from scratch**

We were encouraged by these first results, so we decided to go one step further by looking more widely for completely new architectures that could provide even better results. By not limiting ourselves to combining the already discovered NAS cells, we could look more directly for architectures that took into account our strict latency requirements.

Conducting an *end-to-end search* ordinarily requires exploring thousands of architectures manually, which carries large computational costs. Exploring a single architecture requires several days of training on a data center computer with multiple GPU cards, meaning it would take thousands of days of computation to search for a single task. Instead, we designed a *proxy task:* a scaled-down LiDAR segmentation task that could be solved in just a matter of hours.

One challenge that the team had to overcome was finding a proxy task similar enough to our original segmentation task. We experimented with several proxy task designs before we could ascertain a good correlation between the quality of architectures on that task and those found on the original task. We then launched a search similar to the one from the original AutoML paper but now on the proxy task: a proxy end-to-end search. This was the first time this concept has been applied for use on LiDAR data.



![img](https://cdn-images-1.medium.com/max/1600/1*JCPSzb1GEvUJkgrRLXqXfw.png)

*Proxy end-to-end search: Explore thousands of architecture on a scaled-down proxy task, apply the 100 best ones to the original task, validate and deploy the best of the best architectures on the car.*

We used several search algorithms, optimizing for quality and latency, as this is critical on the vehicle. Looking at different types of CNN architectures and using different search strategies, such as random search and reinforcement learning, we were able to explore more than 10,000 different architectures for the proxy task. By using the proxy task, what would have taken over a year of computational time on a Google TPU cluster only took two weeks. We found even better nets than we had before when we had just transferred the NAS cells:

- Neural nets with 20–30% lower latency and results of the same quality.
- Neural nets of higher quality, with an 8–10% lower error rate, at the same latency as the previous architectures.



![img](https://cdn-images-1.medium.com/max/1200/1*pzaDWldooweo5ToaWnxILQ.png)![img](https://cdn-images-1.medium.com/max/1000/1*pzaDWldooweo5ToaWnxILQ.png)



![img](https://cdn-images-1.medium.com/max/1200/1*yPcHE6Ib3lKBQBEQQxtg4Q.png)

1) The first graph shows a*bout 4,000 architectures discovered with a random search on a simple set of architectures. Each point is an architecture that was trained and evaluated. The solid line marks the best architectures at different inference time constraints. The red dot shows the latency and performance of the net built with transfer learning. In this random search, the nets were not as good as the one from transfer learning. 2) In the second graph, the yellow and blue points show the results of two other search algorithms. The yellow one was a random search on a refined set of architectures. The blue one used reinforcement learning as in [1] and explored more than 6,000 architectures. It yielded the best results. These two additional searches found nets that were significantly better than the net from transfer learning.*

Some of the architectures found in the search showed creative combinations of convolutions, pooling, and deconvolution operations, such as the one in the figure below. These architectures ended up working very well for our original LiDAR segmentation task and will be deployed on Waymo’s self-driving vehicles.



![img](https://cdn-images-1.medium.com/max/1600/1*kXgRb7KIpuotmg1YZQu3BQ.png)

*One of the neural net architectures discovered by the proxy end-to-end search.*

**What’s next**

Our AutoML experimentations are just the beginning. For our LiDAR segmentation tasks, both transfer learning and proxy end-to-end search provided nets that were better than hand-crafted ones. We now have the opportunity to apply these mechanisms to new types of tasks too, which could improve many other neural nets.

This development opens up new and exciting avenues for our future ML work and will improve the performance and capabilities of our self-driving technology. We look forward to continuing our work with Google AI, so stay tuned for more!

**References**

[1] *Barret Zoph and Quoc V. Le*. Neural architecture search with reinforcement learning. ICLR, 2017.

[2] *Barret Zoph, Vijay Vasudevan, Jonathon Shlens, Quoc V. Le,* Learning Transferable Architectures for Scalable Image Recognition. CVPR, 2018.

***Acknowledgements**

This collaboration between Waymo and Google was initiated and sponsored by Matthieu Devin of Waymo and Quoc Le of Google. The work was conducted by Shuyang Cheng of Waymo and Gabriel Bender and Pieter-jan Kindermans of Google. Extra thanks for the support of Vishy Tirumalashetty.



![img](https://cdn-images-1.medium.com/max/1600/1*kVljrdIhXvdoVRXr_n3l-g.jpeg)

*Members of the Waymo and Google teams (from left): Gabriel Bender, Shuyang Cheng, Matthieu Devin, and Quoc Le*



- [Artificial Intelligence](https://medium.com/tag/artificial-intelligence?source=post)
- [Technology](https://medium.com/tag/technology?source=post)
- [Research](https://medium.com/tag/research?source=post)



829 claps

2

Follow

[![Go to the profile of Waymo Team](https://cdn-images-1.medium.com/fit/c/120/120/1*GqcDSHJfweuFsivIcIHkHA.png)](https://medium.com/@waymo?source=footer_card)

### [Waymo Team](https://medium.com/@waymo)

Follow

[![Waymo](https://cdn-images-1.medium.com/fit/c/120/120/1*GqcDSHJfweuFsivIcIHkHA.png)](https://medium.com/waymo?source=footer_card)

### [Waymo](https://medium.com/waymo?source=footer_card)

Waymo — formerly the Google self-driving car project — is making our roads safer and easier to navigate for all. One step at a time.





More from Waymo

Waymo’s early rider program, one year in

[![Go to the profile of Waymo Team](https://cdn-images-1.medium.com/fit/c/36/36/1*GqcDSHJfweuFsivIcIHkHA.png)](https://medium.com/@waymo)

Waymo Team

[Jun 14, 2018](https://medium.com/waymo/waymos-early-rider-program-one-year-in-3a788f995a9c?source=placement_card_footer_grid---------0-41)



2.2K







More from Waymo

Where the next 10 million miles will take us

[![Go to the profile of Waymo Team](https://cdn-images-1.medium.com/fit/c/36/36/1*GqcDSHJfweuFsivIcIHkHA.png)](https://medium.com/@waymo)

Waymo Team

[Oct 10, 2018](https://medium.com/waymo/where-the-next-10-million-miles-will-take-us-de51bebb67d3?source=placement_card_footer_grid---------1-41)



1.1K









Related reads

Multi-Sensor Data Fusion (MSDF) for Driverless Cars, An Essential Primer

[![Go to the profile of Lance Eliot](https://cdn-images-1.medium.com/fit/c/36/36/1*k8hHy6FENBJHF-my2rHy7g.jpeg)](https://medium.com/@lance.eliot)

Lance Eliot

[Apr 2](https://medium.com/@lance.eliot/multi-sensor-data-fusion-msdf-for-driverless-cars-an-essential-primer-a1948bb8b57c?source=placement_card_footer_grid---------2-60)



104





Responses

Write a response…



Show all responses







[![Waymo](https://cdn-images-1.medium.com/fit/c/80/80/1*GqcDSHJfweuFsivIcIHkHA.png)](https://medium.com/waymo)

Never miss a story from **Waymo**, when you sign up for Medium. [Learn more](https://medium.com/@Medium/personalize-your-medium-experience-with-users-publications-tags-26a41ab1ee0c#.hx4zuv3mg)