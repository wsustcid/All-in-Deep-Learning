## Overview

The Course Project is an opportunity for you to apply what you have learned in class to a problem of your interest. Potential projects usually fall into these two tracks:

- **Applications.** If you're coming to the class with a specific background and interests (e.g. biology, engineering, physics), we'd love to see you apply ConvNets to problems related to your particular domain of interest. Pick a real-world problem and apply ConvNets to solve it.
- **Models.** You can build a new model (algorithm) with ConvNets, or a new variant of existing models, and apply it to tackle vision tasks. This track might be more challenging, and sometimes leads to a piece of publishable work.

One **restriction** to note is that this is a Computer Vision class, so your project should involve pixels of visual data in some form somewhere. E.g. a pure NLP project is not a good choice, even if your approach involves ConvNets.

To get a better feeling for what we expect from CS231n projects, we encourage you to take a look at the project reports from previous years:

- [Spring 2017](http://cs231n.stanford.edu/2017/reports.html)
- [Winter 2016](http://cs231n.stanford.edu/2016/reports.html)
- [Winter 2015](http://cs231n.stanford.edu/2015/reports.html)



To inspire ideas, you might also look at recent deep learning publications from top-tier conferences, as well as other resources below.

- [CVPR](http://openaccess.thecvf.com/CVPR2017.py): IEEE Conference on Computer Vision and Pattern Recognition
- [ICCV](http://openaccess.thecvf.com/ICCV2017.py): International Conference on Computer Vision
- [ECCV](http://www.eccv2016.org/main-conference/): European Conference on Computer Vision
- [NIPS](https://papers.nips.cc/): Neural Information Processing Systems
- [ICLR](https://openreview.net/group?id=ICLR.cc/2018/Conference): International Conference on Learning Representations
- [ICML](https://icml.cc/Conferences/2017/Schedule?type=Poster): International Conference on Machine Learning
- Publications from the [Stanford Vision Lab](http://vision.stanford.edu/publications.html)
- [Awesome Deep Vision](https://github.com/kjw0612/awesome-deep-vision)
- [Past CS229 Projects](http://cs229.stanford.edu/projects.html): Example projects from Stanford's machine learning class
- [Kaggle challenges](http://www.kaggle.com/): An online machine learning competition website. For example, a [Yelp classification challenge](https://www.kaggle.com/c/yelp-restaurant-photo-classification).

For applications, this type of projects would involve careful data preparation, an appropriate loss function, details of training and cross-validation and good test set evaluations and model comparisons. Don't be afraid to think outside of the box. Some successful examples can be found below:

- [Teaching Deep Convolutional Neural Networks to Play Go](http://arxiv.org/abs/1412.3409)
- [Playing Atari with Deep Reinforcement Learning](http://arxiv.org/abs/1312.5602)
- [Winning the Galaxy Challenge with convnets](http://blog.kaggle.com/2014/04/18/winning-the-galaxy-challenge-with-convnets/)

ConvNets also run in real time on mobile phones and Raspberry Pi's - building an interesting mobile application could be a good project. If you want to go this route you might want to check out [TensorFlow Mobile / Lite](https://www.tensorflow.org/mobile/) or [Caffe2 iOS/Android integration](https://caffe2.ai/docs/mobile-integration.html).

For models, ConvNets have been successfully used in a variety of computer vision tasks. This type of projects would involve understanding the state-of-the-art vision models, and building new models or improving existing models for a vision task. The list below presents some papers on recent advances of ConvNets in the computer vision community.

- **Image Classification**: [[Krizhevsky et al.\]](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf), [[Russakovsky et al.\]](http://arxiv.org/abs/1409.0575), [[Szegedy et al.\]](http://arxiv.org/abs/1409.4842), [[Simonyan et al.\]](http://arxiv.org/abs/1409.1556), [[He et al.\]](http://arxiv.org/abs/1406.4729), [[Huang et al.\]](https://arxiv.org/abs/1608.06993), [[Hu et al.\]](https://arxiv.org/abs/1709.01507) [[Zoph et al.\]](https://arxiv.org/abs/1707.07012)
- **Object detection**: [[Girshick et al.\]](http://arxiv.org/abs/1311.2524), [[Ren et al.\]](https://arxiv.org/abs/1506.01497), [[He et al.\]](https://arxiv.org/abs/1703.06870)
- **Image segmentation**: [[Long et al.\]](http://arxiv.org/abs/1411.4038) [[Noh et al.\]](https://arxiv.org/abs/1505.04366) [[Chen et al.\]](http://ieeexplore.ieee.org/abstract/document/7913730/)
- **Video classification**: [[Karpathy et al.\]](http://cs.stanford.edu/people/karpathy/deepvideo/), [[Simonyan and Zisserman\]](http://arxiv.org/abs/1406.2199) [[Tran et al.\]](https://arxiv.org/abs/1412.0767) [[Carreira et al.\]](https://arxiv.org/abs/1705.07750) [[Wang et al.\]](https://arxiv.org/abs/1711.07971)
- **Scene classification**: [[Zhou et al.\]](http://places.csail.mit.edu/)
- **Face recognition**: [[Taigman et al.\]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Taigman_DeepFace_Closing_the_2014_CVPR_paper.pdf) [[Schroff et al.\]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf) [[Parkhi et al.\]](http://cis.csuohio.edu/~sschung/CIS660/DeepFaceRecognition_parkhi15.pdf)
- **Depth estimation**: [[Eigen et al.\]](http://www.cs.nyu.edu/~deigen/depth/)
- **Image-to-sentence generation**: [[Karpathy and Fei-Fei\]](http://cs.stanford.edu/people/karpathy/deepimagesent/), [[Donahue et al.\]](http://arxiv.org/abs/1411.4389), [[Vinyals et al.\]](http://arxiv.org/abs/1411.4555) [[Xu et al.\]](https://arxiv.org/pdf/1502.03044.pdf) [[Johnson et al.\]](https://arxiv.org/abs/1511.07571)
- **Visualization and optimization**: [[Szegedy et al.\]](http://arxiv.org/pdf/1312.6199v4.pdf), [[Nguyen et al.\]](http://arxiv.org/abs/1412.1897), [[Zeiler and Fergus\]](http://arxiv.org/abs/1311.2901), [[Goodfellow et al.\]](http://arxiv.org/abs/1412.6572), [[Schaul et al.\]](http://arxiv.org/abs/1312.6055)

You might also gain inspiration by taking a look at some popular computer vision datasets:



- [Meta Pointer: A large collection organized by CV Datasets.](http://www.cvpapers.com/datasets.html)
- [Yet another Meta pointer](http://riemenschneider.hayko.at/vision/dataset/)
- [ImageNet](http://http//image-net.org/): a large-scale image dataset for visual recognition organized by [WordNet](http://wordnet.princeton.edu/) hierarchy
- [SUN Database](http://groups.csail.mit.edu/vision/SUN/): a benchmark for scene recognition and object detection with annotated scene categories and segmented objects
- [Places Database](http://places.csail.mit.edu/): a scene-centric database with 205 scene categories and 2.5 millions of labelled images
- [NYU Depth Dataset v2](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html): a RGB-D dataset of segmented indoor scenes
- [Microsoft COCO](http://mscoco.org/): a new benchmark for image recognition, segmentation and captioning
- [Flickr100M](http://yahoolabs.tumblr.com/post/89783581601/one-hundred-million-creative-commons-flickr-images): 100 million creative commons Flickr images
- [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/): a dataset of 13,000 labeled face photographs
- [Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/): a benchmark for articulated human pose estimation
- [YouTube Faces DB](http://www.cs.tau.ac.il/~wolf/ytfaces/): a face video dataset for unconstrained face recognition in videos
- [UCF101](http://crcv.ucf.edu/data/UCF101.php): an action recognition data set of realistic action videos with 101 action categories
- [HMDB-51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/): a large human motion dataset of 51 action classes
- [ActivityNet](http://activity-net.org/): A large-scale video dataset for human activity understanding
- [Moments in Time](http://moments.csail.mit.edu/): A dataset of one million 3-second videos



## Collaboration Policy

You can work in teams of up to **3** people. We do expect that projects done with 3 people have more impressive writeup and results than projects done with 2 people. To get a sense for the scope and expectations for 2-people projects have a look at project reports from previous years.

## Honor Code

You may consult any papers, books, online references, or publicly available implementations for ideas and code that you may want to incorporate into your strategy or algorithm, so long as you clearly cite your sources in your code and your writeup. However, under no circumstances may you look at another group’s code or incorporate their code into your project.

If you are combining your course project with the project from another class, you must receive permission from the instructors, and clearly explain in the Proposal, Milestone, and Final Report the exact portion of the project that is being counted for CS 231n. In this case you must prepare separate reports for each course, and submit your final report for the other course as well.

## Important Dates

Unless otherwise noted, all project items are due by 11:59 pm Pacific Time.

- Project proposal: due Wednesday, April 25.
- Project milestone: due Wednesday, May 16.
- Final report: due Thursday, June 7. **No late days.**
- Poster PDF: due Monday, June 11. **No late days.**
- Poster session: Tuesday, June 12

## Project Proposal

The project proposal should be one paragraph (200-400 words). Your project proposal should describe:

- What is the problem that you will be investigating? Why is it interesting?
- What reading will you examine to provide context and background?
- What data will you use? If you are collecting new data, how will you do it?
- What method or algorithm are you proposing? If there are existing implementations, will you use them and how? How do you plan to improve or modify such implementations? You don't have to have an exact answer at this point, but you should have a general sense of how you will approach the problem you are working on.
- How will you evaluate your results? Qualitatively, what kind of results do you expect (e.g. plots or figures)? Quantitatively, what kind of analysis will you use to evaluate and/or compare your results (e.g. what performance metrics or statistical tests)?

**Submission:** Please submit your proposal as a PDF on Gradescope. **Only one person on your team should submit.** Please have this person add the rest of your team as collaborators as a "Group Submission".

## Project Milestone

Your project milestone report should be between 2 - 3 pages using the 

provided template

. The following is a suggested structure for your report:



- Title, Author(s)
- Introduction: this section introduces your problem, and the overall plan for approaching your problem
- Problem statement: Describe your problem precisely specifying the dataset to be used, expected results and evaluation
- Technical Approach: Describe the methods you intend to apply to solve the given problem
- Intermediate/Preliminary Results: State and evaluate your results upto the milestone



**Submission**: Please submit your milestone as a PDF on Gradescope. **Only one person on your team should submit.** Please have this person add the rest of your team as collaborators as a "Group Submission".

## Final Report

Your final write-up is required to be between **6 - 8** pages using the [provided template](http://www.pamitc.org/cvpr15/files/cvpr2015AuthorKit.zip), structured like a paper from a computer vision conference (CVPR, ECCV, ICCV, etc.). Please use this template so we can fairly judge all student projects without worrying about altered font sizes, margins, etc. After the class, we will post all the final reports online so that you can read about each others' work. If you do not want your writeup to be posted online, then please let us know at least a week in advance of the final writeup submission deadline.

The following is a suggested structure for your report, as well as the rubric that we will follow when evaluating reports. You don't necessarily have to organize your report using these sections in this order, but that would likely be a good starting point for most projects.

- **Title, Author(s)**

- **Abstract**: Briefly describe your problem, approach, and key results. Should be no more than 300 words.

- **Introduction (10%)**: Describe the problem you are working on, why it's important, and an overview of your results

- **Related Work (10%)**: Discuss published work that relates to your project. How is your approach similar or different from others?

- **Data (10%)**: Describe the data you are working with for your project. What type of data is it? Where did it come from? How much data are you working with? Did you have to do any preprocessing, filtering, or other special treatment to use this data in your project?

- **Methods (30%)**: Discuss your approach for solving the problems that you set up in the introduction. Why is your approach the right thing to do? Did you consider alternative approaches? You should demonstrate that you have applied ideas and skills built up during the quarter to tackling your problem of choice. It may be helpful to include figures, diagrams, or tables to describe your method or compare it with other methods.

- **Experiments (30%)**: Discuss the experiments that you performed to demonstrate that your approach solves the problem. The exact experiments will vary depending on the project, but you might compare with previously published methods, perform an ablation study to determine the impact of various components of your system, experiment with different hyperparameters or architectural choices, use visualization techniques to gain insight into how your model works, discuss common failure modes of your model, etc. You should include graphs, tables, or other figures to illustrate your experimental results.

- **Conclusion (5%)** Summarize your key results - what have you learned? Suggest ideas for future extensions or new applications of your ideas.

- **Writing / Formatting (5%)** Is your paper clearly written and nicely formatted?

- Supplementary Material

  , not counted toward your 6-8 page limit and submitted as a separate file. Your supplementary material might include:

  - Source code (if your project proposed an algorithm, or code that is relevant and important for your project.).
  - Cool videos, interactive visualizations, demos, etc.

  Examples of things to not put in your supplementary material:

  - The entire PyTorch/TensorFlow Github source code.
  - Any code that is larger than 10 MB.
  - Model checkpoints.
  - A computer virus.

**Submission**: You will submit your final report as a PDF and your supplementary material as a separate PDF or ZIP file. We will provide detailed submission instructions as the deadline nears.

**Additional Submission Requirements**: We will also ask you do do the following when you submit your project report:

- **Your report PDF should list all authors who have contributed to your work; enough to warrant a co-authorship position.** This includes people not enrolled in CS 231N such as faculty/advisors if they sponsored your work with funding or data, significant mentors (e.g., PhD students or postdocs who coded with you, collected data with you, or helped draft your model on a whiteboard). All authors should be listed directly underneath the title on your PDF. Include a footnote on the first page indicating which authors are not enrolled in CS 231N. All co-authors should have their institutional/organizational affiliation specified below the title.

- - If you have non-231N contributors, you will be asked to describe the following:
  - **Specify the involvement of non-CS 231N contributors** (discussion, writing code, writing paper, etc). For an example, please see the author contributions for [AlphaGo (Nature, 2016)](https://www.nature.com/nature/journal/v529/n7587/full/nature16961.html#author-information).
  - **Specify whether the project has been submitted to a peer-reviewed conference or journal.** Include the full name and acronym of the conference (if applicable). For example: Neural Information Processing Systems (NIPS). This only applies if you have already *submitted* your paper/manuscript and it is under review as of the report deadline.

- **Any code that was used as a base for projects must be referenced and cited in the body of the paper.** This includes CS 231N assignment code, finetuning example code, open-source, or Github implementations. You can use a footnote or full reference/bibliography entry.

- **If you are using this project for multiple classes, submit the other class PDF as well.** Remember, it is an honor code violation to use the same final report PDF for multiple classes.

In summary, include all contributing authors in your PDF; include detailed non-231N co-author information; tell us if you submitted to a conference, cite any code you used, and submit your dual-project report (e.g., CS 230, CS 231A, CS 234).

## Poster Session

We will hold a poster session in which you will present the results of your projects is form of a poster.

- **Date:** Tuesday, June 12, 2018
- **Time:** 12:00 pm to 3:15 pm
- **Location:** [/a>Jen-Hsun Huang Engineering Center](https://engineering.stanford.edu/location)a
- **Who:** All on-campus students are required to attend. Local SCPD students are highly recommended to attend. Stanford students, faculty, and guests from industry are welcome!
- **Food:** Food and light refreshments will be provided.

Students: We will provide foam poster boards and easels. Please print your poster on a 20 inch by 30 inch poster in either landscape or portrait format. Posters larger than 24 inch by 36 inches may not fit on our poster boards. All students are required to submit a PDF of their poster before the event. See Piazza for details. Caution: Do not wait until the day before the event to print your poster. Many on-campus printers (e.g., EE, BioX) run out of paper or toner during the last week of classes. Many other courses also have poster presentations or academic conferences take place during this week and there is no guarantee they will be able to rush-print your order.

Frequently Asked Questions



- **I can only attend part of the poster session. Is that okay?** Yes, but you will lose all points for the project poster. Your team's scores will be unaffected. We will be taking attendance and TAs/graders/instructors may visit your poster at any time.
- **At the event, can I leave my poster and walk around?** Yes. We encourage you to visit other posters and learn about the cutting-edge projects other students are working on. However, we do ask that you periodically return to your poster in case a TA or instructor needs to grade your poster.
- **I am a local SCPD student, is attendance required?** No. However, the poster session is an immensely valuable opportunity to network with on-campus students, the course staff, and various industry representatives (e.g., investors and recruiters). We strongly recommend you attend if possible. Many SCPD students cite the poster session as their favorite part of the class.
- **I am a non-local SCPD student and cannot attend. Do I have to make a poster?** Yes. You are required to submit your poster as a PDF to Gradescope with the same deadline as on-campus students. We may require you to record a video of yourself presenting or conduct a presentation over video/conference call at a different time; check Piazza for details closer to the event.
- **Can I print my poster on 8.5"x11" pieces of paper and tape it together?** Yes - we will not deduct points if you choose to do this. However we recommend you print a full-sized poster if possible. Not only can you fit more content on a poster, but the visual appeal can help attract visitors and spur additional research discussions about your project.
- **Will you reimburse for poster printing costs?** Unfortunately no. Several departments at Stanford offer free or discounted poster printing to students. Many local businesses (e.g., Staples, FedEx, etc.) offer same-day printing services at reasonable prices. [Lathrop Library](https://vptl.stanford.edu/student-resources/computers-printing/poster-printing-lathrop) offers on-campus poster printing services.
- **I'm part of an organization and we'd like to sponsor or help contribute to the event. How can we get involved?** Please send an email to the course staff at cs231n-spring1718-staff@lists.stanford.edu. We have several sponsorship levels available.