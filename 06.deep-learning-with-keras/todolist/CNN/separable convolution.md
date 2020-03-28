We remark that this extreme form of an Inception module is almost identical to a depthwise sepa- rable convolution, an operation that has been used in neural network design as early as 2014 [15] and has become more popular since its inclusion in the TensorFlow framework [1] in 2016. 



A depthwise separable convolution, commonly called
“separable convolution” in deep learning frameworks such as TensorFlow and Keras, consists in a depthwise convolution, i.e. a spatial convolution performed independently over each channel of an input, followed by a pointwise convolution, i.e. a 1x1 convolution, projecting the channels output by the depthwise convolution onto a new channel space. This is not to be confused with a spatially separable convolution, which is also commonly called “separable convolution” in the image processing community.





[15] L. Sifre. Rigid-motion scattering for image classification, 2014. Ph.D. thesis.