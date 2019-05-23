## 入门指南 | 人工智能的新希望-强化学习全解

原创： 大数据文摘 [大数据文摘](javascript:void(0);) *2017-02-20*

![img](https://mmbiz.qpic.cn/mmbiz_jpg/wc7YNPm3YxWTZJlu9RowP6tq8qWic2ibweCPtFEBzmFYxBnwtz5XAFyrHz7EedDKsJkDic45O1icoFHC1m74vOzNfw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1&retryload=1)

大数据文摘作品，转载具体要求见文末

编译团队 | Jennifer Zhu 赖小娟 张礼俊

作者 |  FAIZAN SHAIKH



很多人说，**强化学习被认为是真正的人工智能的希望**。本文将从7个方面带你入门强化学习，读完本文，希望你对强化学习及实战中实现算法有着更透彻的了解。



介绍



许多科学家都在研究的一个最基本的问题是“人类如何学习新技能？”。 理由显而易见– 如果我们能解答这个问题，人类就能做到很多我们以前没想到的事情。 另一种可能是我们训练机器去做更多的“人类”任务，创造出真正的人工智能。



虽然我们还没有上述问题的全部答案，但有一些事情是清楚的。不论哪种技能，我们都是先通过与环境的互动来学习它。无论是学习驾驶汽车还是婴儿学步，我们的学习都是基于与环境的互动。 从这些互动中学习是所有关于学习与智力的理论的基础概念。



### 强化学习



今天我们将探讨强化学习（Re-inforcement Learning） 一种基于与环境互动的目标导向的学习。**强化学习被认为是真正的人工智能的希望。**我们认为这是正确的说法，因为强化学习拥有巨大的潜力。



强化学习正在迅速发展。它已经为不同的应用构建了相应的机器学习算法。因此，熟悉强化学习的技术会对深入学习和使用机器学习非常有帮助。如果您还没听说过强化学习，我建议您阅读我之前关于强化学习和开源强化学习（RL）平台的介绍文章（https://www.analyticsvidhya.com/blog/2016/12/getting-ready-for-ai-based-gaming-agents-overview-of-open-source-reinforcement-learning-platforms/）。



如果您已经了解了一些强化学习的基础知识，请继续阅读本文。读完本文，您将会对强化学习及实战中实现算法有着更透彻的了解。



附：下面这些算法实现的讲解中，我们将假设您懂得Python的基本知识。如果您还不知道Python，建议可以先看看这个Python教程（https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-learn-data-science-python-scratch-2/）。



轻松搞定强化学习



（1-4是强化学习的步骤，5-7是其他资源）



\1.      提出一个强化学习的问题

\2.      强化学习 v.s. 其他机器学习方法

\3.      解决强化学习问题的基本框架

\4.      用python实现强化学习算法

\5.      更复杂的应用

\6.      强化学习的最新进展

\7.      其他强化学习的资源

 

**1. 提出一个强化学习的问题**



强化学习的目的是学习如何做一件事情，以及如何根据不同的情况选择不同的行动。 它的最终结果是为了实现数值回报信号的最大化。强化学习并不告诉学习者采取哪种行动，而是让学习者去发现采取哪种行动能产生最大的回报。 下面让我们通过一个孩子学走路的简单例子（下图）来解释什么是强化学习。



![img](https://mmbiz.qpic.cn/mmbiz_png/wc7YNPm3YxWTZJlu9RowP6tq8qWic2ibwesf1icn8wMficmibzL4U1ibu20kw4dXzgUwFTUEXQCGVvPynuppjJy2ZChQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1&retryload=1)

上图：孩子学走路。

 



**以下是孩子在学习走路时要采取的步骤：**



1. 首先孩子将观察你是如何行走的。你用两条腿，一步一步走。得到这个概念后，孩子试图模仿你走路的样子。
2. 但孩子很快发现，走路之前必须站起来！这是一个试图走路必经的挑战。所以现在孩子试图先站起来，虽然经历挣扎和滑倒，但仍然决心站起来。
3. 然后还有另一个挑战要应付：站起来很容易，但要保持站立又是另一项挑战！孩子挥舞着双手，似乎是想找到能支撑平衡的地方，设法保持着站立。
4. 现在孩子开始他／她真正的任务––走路。这是件说比做容易的事。要记住很多要点，比如平衡体重，决定先迈哪个脚，把脚放在哪里。



这听起来像一个困难的任务吗？实际上站起来和开始走路确实有点挑战性，但当你走熟练了就不会再觉得走路难。不过通过我们的分析，现在的您大概明白了一个孩子学走路的困难点。



让我们把上面的例子描述成一个强化学习的问题（下图）。这个例子的“问题”是走路，这个过程中孩子是一个试图通过采取行动（行走）来操纵环境（孩子行走的表面）的智能体（agent）。他/她试图从一个状态（即他/她采取的每个步骤）到另一个状态。当他/她完成任务的子模块（即采取几个步骤）时，孩子将得到奖励（让我们说巧克力）。但当他/她不能完成走几步时，他/她就不会收到任何巧克力（亦称负奖励）。这就是对一个强化学习问题的简单描述。



 

![img](https://mmbiz.qpic.cn/mmbiz_png/wc7YNPm3YxWTZJlu9RowP6tq8qWic2ibwefOu7J6ib0W6enicSvXeDcgHNibxnWhibGwC0ibicb9xHvyBuorVvIndnFIicw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1&retryload=1)

![img](https://mmbiz.qpic.cn/mmbiz_png/wc7YNPm3YxWTZJlu9RowP6tq8qWic2ibweYeg8JCiahPQSzxPND9ACiayTYJMibFZtKJwxNbAl7RianMCP4qvtup9DlQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1&retryload=1)上图：把小孩子学走路的过程（图下方）归纳成一个强化学习的问题（图上方）。

 

这里我们还推荐一个不错的对强化学习的视频介绍（https://www.youtube.com/watch?v=m2weFARriE8）。



\2. 强化学习 v.s. 其他机器学习方法



强化学习是机器学习算法的一个大的类型。下图描述了机器学习方法的类型。





![img](https://mmbiz.qpic.cn/mmbiz_png/wc7YNPm3YxWTZJlu9RowP6tq8qWic2ibweDaYZEI3TRn9rSlyJscCnyD8mwE7icT42xcYbaL47ic0LLiaQickVic0uPyQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1&retryload=1)

上图：机器学习的方法分类：蓝色方框从左到右依次为监督学习，无监督学习和强化学习。

 

**让我们来比较一下强化学习和其他种类机器学习方法：**



- 监督学习（supervised learning）v.s. 强化学习：在监督学习中，有一个外部“监督者”（supervisor）。“监督者”了解环境，并与智能体共享环境信息以完成任务。但这其中存在一些问题，智能体可以通过执行许多种不同子任务的组合来达到目标。所以创建一个“监督者””几乎是不切实际的。例如在象棋游戏中，有成千上万种走法。因此，创建一个可以下象棋的知识库是一个单调乏味的任务。在这样的问题中，从经验中学习更为可行。这可以说是强化学习和监督学习的主要区别。在监督学习和强化学习中，输入和输出之间都存在映射（mapping）。但在强化学习中，还存在对智能体进行反馈的奖励函数，这在监督学习中是不存在的。



- 无监督学习（unsupervised learning） v.s. 强化学习：在强化学习中，有一个从输入到输出的映射。这种映射在无监督学习中并不存在。在无监督学习中，主要任务是找到数据本身的规律而不是映射。例如，如果任务是向用户建议新闻文章，则无监督学习算法将查看该人先前读过的文章并向他们建议类似的文章。而强化学习算法将通过建议少量新闻文章给用户，从用户获得不断的反馈，然后构建一个关于人们喜欢哪些文章的“知识图”。



此外，还有第四种类型的机器学习方法，称为半监督学习（semi-supervised learning），其本质上是监督学习和无监督学习的结合（利用监督学习的标记信息，利用未标记数据的内在特征）。它类似于监督学习和半监督学习，不具有强化学习具备的反馈机制（奖赏函数）。（译者注：这里应该是原文作者的笔误，强化学习有映射，映射是每一个状态对应值函数。而无监督学习没有标记信息，可以说是没有映射的。我想这里作者想要表达的是半监督学习区别于强化学习的地方是半监督学习没有强化学习的反馈这个机制。）



\3. 解决强化学习问题的基本框架



为了了解如何解决强化学习问题，我们将分析一个强化学习问题的经典例子––多摇臂老虎机问题。 首先，我们将去回答探索 v.s. 利用的根本问题，然后继续定义基本框架来解决强化学习的问题。

 



![img](https://mmbiz.qpic.cn/mmbiz_png/wc7YNPm3YxWTZJlu9RowP6tq8qWic2ibweibVbKibUfstmTnQ8vWr8HibA0NK7OiaSKRu4E6TVPpXLM49nGdDfEY48pQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1&retryload=1)

![img](https://mmbiz.qpic.cn/mmbiz_gif/wc7YNPm3YxWTZJlu9RowP6tq8qWic2ibweHfzWnBMmibDC3Ec0JxNyoFFDjopPUBgRYuNLWCria5mRIQ3eu7TvR5lQ/0?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&retryload=1)

上图：赌场里的“老虎机”。



假设你有很多吐出随机奖金的老虎机（即投币式游戏机，见上图）。



现在你想尽可能快地从老虎机获得最多的奖金。你会怎么做？



一个幼稚的方法可能是只选择一个老虎机，并拉一整天的杠杆。听起来好无聊，但这种方法可能会给你赢点小钱。你也有可能会中大奖（几率接近0.00000 ... .1），但大多数时候你可能只是坐在老虎机面前亏钱。这种方法的正式定义是一种纯利用（pureexploitation）的方法。这是我们的最佳选择吗？答案是不。



让我们看看另一种方法。我们可以拉每个老虎机的杠杆，并向上帝祈祷，至少有一个会中奖。这是另一个幼稚的方法，能让你拉一整天的杠杆，但老虎机们只会给你不那么好的收获。正式地，这种方法也被正式定义为一种纯探索（pureexploration）的方法。



这两种方法都不是最优的方法。我们得在它们之间找到适当的平衡以获得最大的回报。这被称为强化学习的探索与利用困境。



首先，我们要正式定义强化学习问题的框架，然后列出可能的解决方法。



### 马尔可夫决策过程：

在强化学习中定义解法的数学框架叫做马尔可夫决策过程（Markov Decision Process）。 它被设计为：

●       一系列状态的集合（Set of states），S

●       一系列行动的集合（Set of actions），A

●       奖励函数（Reward function），R

●       策略（Policy），π

●       价值（Valu），V



我们必须采取行动（A）从我们的开始状态过渡到我们的结束状态（S）。我们采取的每个行动将获得奖励（R）。 我们的行为可以导致正奖励或负奖励。



我们采取的行动的集合（A）定义了我们的策略（π），我们得到的奖励（R）定义了我们的价值（V）。 我们在这里的任务是通过选择正确的策略来最大化我们的奖励。 所以我们必须对时间t的所有可能的S值最大化。

 

旅行推销员问题



让我们通过另一个例子来进一步说明如何定义强化学习问题的框架。



![img](https://mmbiz.qpic.cn/mmbiz_png/wc7YNPm3YxWTZJlu9RowP6tq8qWic2ibweHdXm8dyEmLd8lsypLuyXYfjtyqBO6ibFu8DOXZgAgMx8lQfs24ddBPA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1&retryload=1)上图：旅行推销员的例子。A–F表示地点，之间的连线上的数字代表在两个地点间的旅行成本。

 

这显示的是旅行推销员问题。推销员的任务是以尽可能低的成本从地点A到地点F。 这两个位置之间的每条连线上的数字表示旅行这段距离所需花费的成本。负成本实际上是一些出差的收入。 我们把当推销员执行一个策略累积的总奖励定义为价值。



这里，

●       一系列状态的集合是那些节点，即{A，B，C，D，E，F}

●       采取的行动的集合是从一个地方到另一个地方，即{A→B，C→D等}

●       奖励函数是节点的连线上的值，即成本

●       策略是完成任务的“方式”，即{A - > C - > F}



现在假设你在位置A，在这个平台上唯一可见路径是你下一目的地的（亦称可观测的空间），除此之外所有都是未知的。



当然你可以用贪婪算法选择下一步最有可能的，从{A -> (B, C, D, E)}子集中选出{A -> D}。同样的你在位置D，想要到达F，你可以从{D -> (B, C, F)}中选择，可以看出由于{D -> F}路径花费最小，选择此路径。



到此为止，我们的规则是{A -> D -> F}，价值为-120.



恭喜你！你刚刚完成了一个强化学习算法。这个算法被称作ε-贪心算法，以贪心方式解决问题。现在如果你（销售人员）想要再次从位置A到F，你总是会选择相同的策略。



**其他的旅行方式？**



你可以猜测到我们的策略属于哪一个类别么（例如，纯探索vs纯开发）？



可以看出我们选择的并不是最优策略，我们必须去一点点“探索”来发现最优策略。在这里我们使用的方法是基于策略的学习，我们的任务是在所有可能策略中发现最优策略。解决这个问题有很多不同的方式，简单列举主要类别如下：



●    基于策略，重点是找到最优策略

●    基于价值，重点是找到最优价值，例如，累计奖励

●    基于动作，重点是在执行每一步动作时，确定什么是最优动作

 

我会尝试在以后的文章中更深入地讲述强化学习算法，那时，你们就可以参考这篇强化学习算法调查的文章(https://www.jair.org/media/301/live-301-1562-jair.pdf)。（译者注：这里是原文作者的一个笔误。Q-learning，它可以用一个线性函数作为function approximator, 也可以通过列举每一个q-state的值来做。用神经网络来做Q-learning的function approximator应该是15年Google Deepmind发表在Nature的文章开始的，那篇文章中称该算法为deep-Q-network，后来统称为deep q learning）



4.强化学习的实践案例



我们会使用深度Q学习算法，Q学习是基于策略的，用神经网络来近似值函数的学习算法。Google使用该算法在Atari游戏中击败了人类。



让我们看看Q学习的伪代码：

1. 初始化价值表‘Q(s,a)’.
2. 观测到当前状态点’s’.
3. 基于策略选择该状态下的行动’a’(例如，ε-贪心)
4. 采取行动并观察奖励值’r’及新状态点’s’
5. 根据上面描述的公式及参数，用观测到的奖励值及下一状态可能的最大奖励值更新状态点新值。
6. 设置新状态，重复此流程直至到达最后目标点。

 

Q学习算法的简单描述可以总结如下：



![img](https://mmbiz.qpic.cn/mmbiz_png/wc7YNPm3YxWTZJlu9RowP6tq8qWic2ibwe9DhicBbEF8WaKF2T0A8yst67gQYriaOsbhuQA6e3OrPMicdDN5KsyPN5Q/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1&retryload=1)

该图是Q学习算法流程图



我们先看看什么是Cartpole问题，再继续编程提供解决方案



当我还是一个小孩的时候，我记得我会捡一根棍子试着用一只手让它保持平衡。我和我的朋友们一起比赛看谁让棍子保持平衡的时间最长就可以得到“奖励”，一块巧克力！



开始我们的代码前，我们需要先安装一些东西，



**步骤1:安装keras-rl库**

从终端运行以下命令：

git clone https://github.com/matthiasplappert/keras-rl.git
cd keras-rl
python setup.py install



**步骤2:安装CartPole环境组件**

假设你已经安装了pip，使用pip命令安装以下库

pip install h5py
pip install gym



**步骤3:启动**

首先我们要导入所需模块

import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory



然后设置相关变量

ENV_NAME = 'CartPole-v0'

\# Get the environment and extract the number of actions available in theCartpole problem
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n





下一步，我们创建一个简单的单隐层神经网络模型。

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())



接下来，配置并编译我们的代理端。我们将策略设成ε-贪心算法，并且将存储设置成顺序存储方式因为我们想要存储执行操作的结果和每一操作得到的奖励。

policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,nb_steps_warmup=10,
target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.fit(env, nb_steps=5000, visualize=True, verbose=2)



现在测试强化学习模型

dqn.test(env, nb_episodes=5, visualize=True)

This will be the output of our model:

这就是模型输出结果：



![img](https://mmbiz.qpic.cn/mmbiz_png/wc7YNPm3YxWTZJlu9RowP6tq8qWic2ibweIgUicscgkfhXkvJjuJibeib9Nd3zE0EQs5gy5m2dFZPd678H5guUSG48g/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1&retryload=1)瞧！你构建了一个强化模型的雏形！





 **5.增加复杂性**



现在你已经有了一个强化学习的基础成品，让我们来进一步的每次增加一点点复杂度以解决更多的问题。



问题－汉诺塔

 ![img](https://mmbiz.qpic.cn/mmbiz_png/wc7YNPm3YxWTZJlu9RowP6tq8qWic2ibwe3xla3HEcib0tWriaRFDR7LenUMOPMF3JjoZZQcw8AJHX7icuibNia7OibdUg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1&retryload=1)



对于不知道该游戏的人简单说明一下——发明于1883年，由3根杆及一些逐增大小的圆盘（如上图中所示的3个一样）从最左边的杆开始，目标是从选择最小移动次数将所有圆盘从最左边移动到最右边（你可以从维基百科得到更多讯息(https://en.wikipedia.org/wiki/Tower_of_Hanoi)）。



如果我们要映射这个问题，从定义状态开始：

●    开始状态 – 3个圆盘都在最左边杆上（从上到下依次为1、2、3）

●    结束状态 – 3个圆盘都在最右边杆上（从上到下依次为1、2、3）

 

所有可能的状态：

列举可能的27个状态：



| All disks  in a rod | One disk  in a Rod | (13) disks  in a rod | (23) disks  in a rod | (12) disks  in a rod |
| ------------------- | ------------------ | -------------------- | -------------------- | -------------------- |
| (123)**             | 321                | (13)2*               | (23)1*               | (12)3*               |
| *(123)*             | 312                | (13)*2               | (23)*1               | (12)*3               |
| **(123)             | 231                | 2(13)*               | 1(23)*               | 3(12)*               |
|                     | 132                | *(13)2               | *(23)1               | *(12)3               |
|                     | 213                | 2*(13)               | 1*(23)               | 3*(12)               |
|                     | 123                | *2(13)               | *1(23)               | *3(12)               |

图中(12)3*代表的是圆盘1和圆盘2依次在最左边杆上（从上到下），圆盘3在中间杆上，＊表示最右边杆为空



数字奖励：

因为我们想以最少步数来解决问题，我们可以设定每一步的奖励为-1。



规则：

现在，不考虑任何技术细节，我们可以标记出在以上状态间可能出现的转移。例如从奖励为-1的状态(123)** 到状态 (23)1*，也可以是到状态(23)*1。



同样地，你看出了上面提到的27个状态的每一个都类似于之前销售人员旅行的示意图。我们可以根据之前的经验找出最优解决方案选择不同状态和路径。



问题 － 3 x 3 魔方



当我在为你解决这个问题的同时，也想要你自己也做一做。遵照我上面使用的相同步骤，你可以更好的理解和掌握。



从定义开始和结束状态开始，接下来，定义所有可能的状态和相应的状态转移奖励和规则。最后，使用相同的方法你可以提供解决魔方问题的方案。



**6.强化学习的研究现状**



你已经意识到了魔方问题的复杂度比汉诺塔高了好几个倍，也明白每次可选择的操作数是怎么增长的。现在想想围棋游戏里面状态数和选择，行动起来吧！最近谷歌DeepMind创建了一个深度强化学习算法打败了李世石！



随着近来涌现的深度学习成功案例，焦点慢慢转向了应用深度学习解决强化学习问题。李世石被谷歌deepmind开发的深度强化学习算法开打败的新闻铺天盖地袭来。同样的突破也出现在视频游戏中，已经逼近甚至超出人类级别的准确性。研究仍然同等重要，不管是行业还是学术界的翘楚都在共同完成这个构建更好的自我学习机器的目标。

 



![img](https://mmbiz.qpic.cn/mmbiz_png/wc7YNPm3YxWTZJlu9RowP6tq8qWic2ibwetVHZQqJM1C5kKRpiaYuoxflhicLHUddslJvuc66JO19CoS5fTGaj1hrw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1&retryload=1)

图为李世石与AlphaGo参与围棋人机大战中



深度学习应用的主要领域如下：



●       游戏原理及多智能体交互

●       机器人学

●       计算机网络

●       车辆导航

●       医药学

●       行业物流

 

随着近期将深度学习应用于强化学习的热潮，毫无疑问还有许多未探索的事在等待着更多的突破来临！



其中一条最近的新闻：



![img](https://mmbiz.qpic.cn/mmbiz_png/wc7YNPm3YxWTZJlu9RowP6tq8qWic2ibwenRicibZb7hJxKWdwlbEFnZUa2b3WxFrQBX4C6KP45XsAXuDOgT5jydibA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1&retryload=1)



**7.其他资源**



我希望现在你已经对强化学习怎么运行有了一个深入的了解。列举了一些可以帮你探索更多有关强化学习的其他资源：



- 强化学习视频（https://www.analyticsvidhya.com/blog/2016/12/21-deep-learning-videos-tutorials-courses-on-youtube-from-2016/）
- 介绍强化学习的书籍（https://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf）
- Github上强化学习的优秀资源（https://github.com/aikorea/awesome-rl）
- David Silver强化学习课程（https://www.youtube.com/playlist?list=PLV_1KI9mrSpGFoaxoL9BCZeen_s987Yxb）

 

结束语



我希望你们能喜欢阅读这篇文章，如果你们有任何疑虑和问题，请在下面提出。如果你们有强化学习的工作经验请在下面分享出来。通过这篇文章我希望能提供给你们一

个强化学习的概况，以及算法如何实际实施的，希望对你们有用。





原文链接：https://www.analyticsvidhya.com/blog/2017/01/introduction-to-reinforcement-learning-implementation/?winzoom=1









```
关于转载如需转载，请在开篇显著位置注明作者和出处（转自：大数据文摘 |bigdatadigest），并在文章结尾放置大数据文摘醒目二维码。无原创标识文章请按照转载要求编辑，可直接转载，转载后请将转载链接发送给我们；有原创标识文章，请发送【文章名称-待授权公众号名称及ID】给我们申请白名单授权。未经许可的转载以及改编者，我们将依法追究其法律责任。联系邮箱：zz@bigdatadigest.cn。
```



志愿者介绍

**回复“志愿者”了解如何加入我们**





![img](https://mmbiz.qpic.cn/mmbiz_png/wc7YNPm3YxV6dmia5icw8OvBBqANAABpbzRUOdWB3ib9pY0jT4gmNZicW7icxeUElvbyDn5LiaWOCFicouBnmmFgvEKicg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1&retryload=1)

![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)





**往期精彩文章**



**点击图片阅读文章**

**TED | 数学告诉你，完美伴侣如何选择**[![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)](http://mp.weixin.qq.com/s?__biz=MjM5MTQzNzU2NA==&mid=2651641272&idx=1&sn=f205ff77fd85d82770a5f9e922003248&scene=21#wechat_redirect)

![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)



![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)