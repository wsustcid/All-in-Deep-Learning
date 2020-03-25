<https://towardsdatascience.com/displaying-convnets-feature-maps-on-real-time-video-with-keras-and-opencv-418b986adda7>



[Homepage](https://medium.com/)



[![Towards Data Science](https://cdn-images-1.medium.com/letterbox/194/72/50/50/1*5EUO1kUYBthpOCPzRj_l2g.png?source=logoAvatar-22bd1c4e6cea---7f60cf5620c9)](https://towardsdatascience.com/?source=logo-22bd1c4e6cea---7f60cf5620c9)

- [DATA SCIENCE](https://towardsdatascience.com/data-science/home)
- [MACHINE LEARNING](https://towardsdatascience.com/machine-learning/home)
- [PROGRAMMING](https://towardsdatascience.com/programming/home)
- [VISUALIZATION](https://towardsdatascience.com/data-visualization/home)
- [AI](https://towardsdatascience.com/artificial-intelligence/home)
- [JOURNALISM](https://towardsdatascience.com/data-journalism/home)
- [PICKS](https://towardsdatascience.com/editors-picks/home)
- 

- [CONTRIBUTE](https://towardsdatascience.com/contribute/home)



# Visualization of Deep Learning Feature Maps in Mini Autonomous Vehicles

[![Go to the profile of Nelson Fernandez](https://cdn-images-1.medium.com/fit/c/100/100/1*4xs-KRIrBSHHKhR-hEz26g.png)](https://towardsdatascience.com/@nelson.fernandez?source=post_header_lockup)

[Nelson Fernandez](https://towardsdatascience.com/@nelson.fernandez)Follow

Apr 20, 2018

It’s been a few months since we started building The Axionaut, a [mini-autonomous radio controlled (RC) car](https://www.axionable.com/hello-axionaut-1er-prototype-de-vehicule-autonome-daxionable/) and raced it in some competitions in Paris. So far so good, we’ve managed to obtain good positions. However, there is always some curiosity about what could be actually going on inside the Convolutional Neural Network that controls the vehicle.

There are some great articles out there about how to [display feature maps](https://hackernoon.com/visualizing-parts-of-convolutional-neural-networks-using-keras-and-cats-5cc01b214e59) and [filters](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html), all very helpful when trying to understand and code ConvNet’s feature maps. We also saw some [cool Nvidia videos](http://www.youtube.com/watch?v=URmxzxYlmtg&t=16m51s) showing real-time ConvNet neuron activations on autonomous cars (but, how do they do that?).

So, we decided to go through it and try to replicate the experience in our prototype. To do so, we used the pre-trained [Keras ConvNet autopilot model](https://github.com/Axionable/Axionaut)we already had, and some videos taken from the car while training and racing.

With this as a good starting point, we plunged ourselves into a couple of days of finding out the answers to classic questions like ’how does the network see the world’ and ‘what the network actually pays attention to’.

The result of the experience is shown here:



<iframe data-width="854" data-height="480" width="700" height="393" data-src="/media/f9ad16f048995b03338a9e7035200902?postId=418b986adda7" data-media-id="f9ad16f048995b03338a9e7035200902" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Fi.ytimg.com%2Fvi%2FYC13O-U5MnY%2Fhqdefault.jpg&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/f9ad16f048995b03338a9e7035200902?postId=418b986adda7" style="display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 393.391px;"></iframe>

### Implementation

If you are curious about how we did this, the first thing you need to understand is how feature maps of the convolutional layers ’fire’ when detecting relevant features on its visual field. A very nice explanation of this can be found on [Harsh Pokharna’s article](https://medium.com/technologymadeeasy/the-best-explanation-of-convolutional-neural-networks-on-the-internet-fbb8b1ad5df8).

In this case, our car has become an ‘expert’ on detecting lanes!

But, how does it do that? In fact, there is no explicit programming behind it. During training we provided desired outputs (turn left, right or continue straight) and example road images, so the network automatically ’learned’ that lanes are key.

As humans would do, without taking into account any other factors (other vehicles, road signs, pedestrians or destination) lanes give us relevant information regarding the correct decision. Shall we turn left? Right? Continue straight?

Well, let’s go back to the matter. The first thing we should do is to access a convolutive layer of interest and draw a heatmap of the activations. To do so, we used a slightly modified version of this great [repository](https://github.com/philipperemy/keras-visualize-activations).

A complete reconstruction of the activations would necessarily imply taking into account the contributions of both ‘deep’ and ‘shallow’ convolutional layers as explained [here](https://arxiv.org/abs/1611.05418).

To simplify, we decided to estimate the activations from a single convolutive layer, performing a cubic interpolation upsampling instead of a deconvolution. After a visual inspection of all feature maps across the network, we selected the second convolutive layer.



<iframe width="700" height="250" data-src="/media/9ec0212179b779e4698d2afe6b531564?postId=418b986adda7" data-media-id="9ec0212179b779e4698d2afe6b531564" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars2.githubusercontent.com%2Fu%2F23075615%3Fs%3D400%26v%3D4&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/9ec0212179b779e4698d2afe6b531564?postId=418b986adda7" style="display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 763px;"></iframe>

The results are shown here:



![img](https://cdn-images-1.medium.com/max/1600/1*G2BAKsfW62P6EwBSw2-NWw.png)

Input image



![img](https://cdn-images-1.medium.com/max/1600/1*jdpXgEj9RekeT0hiWBU-Uw.png)

Feature map of the second convolutive layer

At this point it’s pretty clear the network is mainly responding to lanes. The next step is to overlap the original input image and the activations, in a way that regions with high responses are cleanly superposed without compromising the shape or colours of the original image.

OpenCV to the rescue! The first step is to create a binary mask that allows us to segment the highest activations while excluding the rest. Due to the small size of the activation map, upsampling will be also needed. After this, we will apply some bitwise operations to obtain the final merged image.

The first bitwise operation is the ‘and’ between the binary mask and the activation map. This operation can be easily implemented using OpenCV and allows to segment the highest map’s activations.



<iframe width="700" height="250" data-src="/media/32771363c9f76622fcca6e5b4b552120?postId=418b986adda7" data-media-id="32771363c9f76622fcca6e5b4b552120" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars2.githubusercontent.com%2Fu%2F23075615%3Fs%3D400%26v%3D4&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/32771363c9f76622fcca6e5b4b552120?postId=418b986adda7" style="display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 388.984px;"></iframe>



![img](https://cdn-images-1.medium.com/max/1600/1*BYYXc0l-4oJ8aoUoLWUilg.png)

Binary mask



![img](https://cdn-images-1.medium.com/max/1600/1*ouf2Be_w066hnfrVWp5I_Q.png)

Bitwise “and” operation between mask and feature map

As expected, we obtain a clean lane segmentation entirely made by the Convolutive Neural Network.

At this point I guess you can imagine the second bitwise operation needed to obtain the final image: the addition. The cold blue color appears due to differences between Matplotlib (RGB) and OpenCV (BGR) color formats. You can play with this changing Matplotlib’s color maps to get different colors!



<iframe width="700" height="250" data-src="/media/569cb0b220af25d210468c15c80c1fbd?postId=418b986adda7" data-media-id="569cb0b220af25d210468c15c80c1fbd" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars2.githubusercontent.com%2Fu%2F23075615%3Fs%3D400%26v%3D4&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/569cb0b220af25d210468c15c80c1fbd?postId=418b986adda7" style="display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 169px;"></iframe>

And voilà, we obtained the final merger between the input image and the feature map with a reasonable estimation of the network activations.



![img](https://cdn-images-1.medium.com/max/1600/1*t6mvqkN0yGS5xFGc2pwRCA.png)

Final merged image

Now, let’s render an .avi video with the results.



<iframe width="700" height="250" data-src="/media/4e3237743b8f86b59fb4536d1ed5e49a?postId=418b986adda7" data-media-id="4e3237743b8f86b59fb4536d1ed5e49a" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars2.githubusercontent.com%2Fu%2F23075615%3Fs%3D400%26v%3D4&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://towardsdatascience.com/media/4e3237743b8f86b59fb4536d1ed5e49a?postId=418b986adda7" style="display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 256.984px;"></iframe>

### Questions

If you have any questions, I’ll be happy to answer them in the comments. A link to the public repository including all code and database is available [here](https://github.com/Axionable/FeatureMaps). Don’t forget to follow us on [Twitter](https://twitter.com/AxionableData).

Thanks to [Carl Robinson](https://medium.com/@carl.robinson?source=post_page).

- [Towards Data Science](https://towardsdatascience.com/tagged/towards-data-science?source=post)
- [Self Driving Cars](https://towardsdatascience.com/tagged/self-driving-cars?source=post)
- [Deep Learning](https://towardsdatascience.com/tagged/deep-learning?source=post)
- [Artificial Intelligence](https://towardsdatascience.com/tagged/artificial-intelligence?source=post)
- [Machine Learning](https://towardsdatascience.com/tagged/machine-learning?source=post)



667 claps

3

Follow

[![Go to the profile of Nelson Fernandez](https://cdn-images-1.medium.com/fit/c/120/120/1*4xs-KRIrBSHHKhR-hEz26g.png)](https://towardsdatascience.com/@nelson.fernandez?source=footer_card)

### [Nelson Fernandez](https://towardsdatascience.com/@nelson.fernandez)

AI researcher at Renault Automotive, Paris, France

Follow

[![Towards Data Science](https://cdn-images-1.medium.com/fit/c/120/120/1*F0LADxTtsKOgmPa-_7iUEQ.jpeg)](https://towardsdatascience.com/?source=footer_card)

### [Towards Data Science](https://towardsdatascience.com/?source=footer_card)

Sharing concepts, ideas, and codes.





More from Towards Data Science

Data Scientists Are Thinkers





Conor Dewey

[May 12](https://towardsdatascience.com/data-scientists-are-thinkers-a36cc186d570?source=placement_card_footer_grid---------0-41)



1K







More from Towards Data Science

Extreme Rare Event Classification using Autoencoders in Keras

[![Go to the profile of Chitta Ranjan](https://cdn-images-1.medium.com/fit/c/36/36/1*RWPL3zHc1gcwdnl-RGgkZQ.jpeg)](https://towardsdatascience.com/@cran2367)

Chitta Ranjan

[May 4](https://towardsdatascience.com/extreme-rare-event-classification-using-autoencoders-in-keras-a565b386f098?source=placement_card_footer_grid---------1-41)



2.3K







More from Towards Data Science

Using What-If Tool to investigate Machine Learning models.





Parul Pandey

[May 3](https://towardsdatascience.com/using-what-if-tool-to-investigate-machine-learning-models-913c7d4118f?source=placement_card_footer_grid---------2-41)



1.6K





Responses

![Ws Aisha](https://cdn-images-1.medium.com/fit/c/36/36/0*qeyEdMrSCjvsk0wl.jpg)

Write a response…

Ws Aisha



Conversation with [Nelson Fernandez](https://medium.com/@nelson.fernandez).

[![Go to the profile of Mostafa Hussein](https://cdn-images-1.medium.com/fit/c/36/36/0*3xw__VKLu3m0nrIR)](https://medium.com/@mostafa.husseinsh)

Mostafa Hussein

[Oct 8, 2018](https://medium.com/@mostafa.husseinsh/hello-thanks-for-this-nice-article-but-does-this-only-works-on-your-model-c0cf40b260c?source=responses---------0---------------------)

Hello, thanks for this nice article
but does this only works on your model ? I already have a model in TensorFlow and I can extract the feature vectors from the last layer, but I want to visualize them as you did





[1 response](https://medium.com/@mostafa.husseinsh/hello-thanks-for-this-nice-article-but-does-this-only-works-on-your-model-c0cf40b260c?source=responses---------0---------------------#--responses)

[![Go to the profile of Nelson Fernandez](https://cdn-images-1.medium.com/fit/c/36/36/1*4xs-KRIrBSHHKhR-hEz26g.png)](https://medium.com/@nelson.fernandez)

Nelson Fernandez

[Oct 9, 2018](https://medium.com/@nelson.fernandez/hello-mostafa-1bc477ed19f7?source=responses---------0---------------------)

Hello Mostafa,

Yes it should work. Please check the jupyter notebook in the repository to guide you through the process.

Have fun!

Nelson