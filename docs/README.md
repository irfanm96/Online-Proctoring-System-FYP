---
layout: home
permalink: index.html

# Please update this with your repository name and title
repository-name: e15-4yp-online-proctoring-system
title:
---

[comment]: # "This is the standard layout for the project, but you can clean this and use your own template"

# Online Exam Proctoring System

#### Team

- E/15/138, Mohamed Irfan, [email](mailto:irfanmm96@gmail.com)
- E/15/021, Mohamed Aslam, [email](mailto:e15021@ce.pdn.ac.lk)

#### Supervisors

- Dr. Ziyan Maraikar, [email](mailto:ziyanm@eng.pdn.ac.lk)
- Dr. Upul Jayasinghe, [email](mailto:upuljm@eng.pdn.ac.lk)
- Mr. Mohamed Fawzan, [email](mailto:fawzanm@gmail.com)

#### Table of content

1. [Abstract](#abstract)
2. [Related works](#related-works)
3. [Methodology](#methodology)
4. [Experiment Setup and Implementation](#experiment-setup-and-implementation)
5. [Results and Analysis](#results-and-analysis)
6. [Conclusion](#conclusion)
7. [Links](#links)

---

## Abstract

The requirement of e-learning has grown rapidly but examinations still rely on the traditional approaches where test takers are required to visit testing centers, there are also concerns of stealing the proprietary content of the exams such as questions/answers.

This paper proposes an Online Proctoring System to automate invigilation which makes use of inputs from a web browser based online examination, without using any external hardware or any standalone application. As a result, any misconduct during the examination can be identified and labeled as suspicious.

The system addresses a subset of misconduct such as frequent abnormal activities of the test taker’s head pose, asking help from a nearby person, or inappropriate use of keyboard or mouse in the context.

**Keywords:** Online proctoring/invigilation, anomaly detection, unsupervised learning, behaviour fingerprinting.

## Related works

This section presents the state of art in the area of online exam proctoring. Various ways to handle the online exam proctoring are described  classed by modalities used for detecting the abnormal behaviors. Selected examples from the literature are presented to support the proctoring methods.

In the literature, the proctoring of online exams or assessments has been growing in interest.

In Terms of live human invigilation,, there are certain drawbacks to this method. An invigilator can only invigilate a particular number of candidates.

Moreover, since these are online examinations, in most of the examinations the candidate has the freedom to do the exam anytime they want and there are exams which allow the candidate to even pause the exam and do it later.

The paper [[1]](#bookmark=id.ncqiy7luj2a7) has highlighted a problem of similarity between candidates and the difficulty that an invigilator could face in differentiating between lookalikes, family members, or twins.

Moving to automated online invigilation without live human proctoring, involves computer vision and machine learning techniques, and behaviour fingerprinting approaches. Most of the literature is not dependent on single modality, because of various limitations, the systems are built based on multi-modal architecture. The following content is summarized according to each modality based on the literature.


### 2.1 Analyzing webcam video

Webcam video stream is considered as one of the crucial inputs of an online invigilation system. This is the basis for live human proctoring as well. Instead of a human invigilator, several papers propose to use computer vision techniques to analyze the video.

The paper [[2]](#bookmark=id.2nef0acyrhv9) has used the webcam video as an input for the user verification system which is one of the modalities in their proposed system. A Minimum Average Correlation Energy (MACE) filter is used for user identification. They used a special hardware called wear cam ( a camera that the candidate needs to wear which is attached to a spectacle ) to capture the vision of the user. This camera feed is used to check if the user is looking at any printed content or a mobile phone screen. Though it seems a good add on, not everyone can afford this extra hardware and there are people who already wear spectacles since they have sight issues.

The paper [[3]](#bookmark=id.m5e0l1xzl9fz) has defined a system where it summarizes the video for the use of remote invigilators. The idea is to track the abnormal head poses of the candidate and summarize the content for a human proctor who can decide based on the summary.

Another parameter that is used in the literature is the liveness detection using the webcam. The paper [[4]](#bookmark=id.7hclkzt9l5rd) has included this modality to their online invigilation system where they have more modalities like face detection and verification as well. Liveness detection is performed using LBP (Local Binary Patterns) In the method proposed in [[4]](#bookmark=id.7hclkzt9l5rd). They have marked some liveness detection methods as unsuitable for online examination.

i. A method proposed in [[8]](#bookmark=id.jgezbl4kure6) that uses eye blinking as a criterion.

ii. [[9]](#bookmark=id.hchtsp1zhh1u) have used the fact that facial parts in real faces move differently than on photographs.

iii. Liveness using properties of skin [[10]](#bookmark=id.xjs3083d4a5l).

This paper [[6]](#bookmark=id.e5hqehst5dje) also has analyzed the video using various techniques. They tracked eye gaze using the method proposed in [[5]](#bookmark=id.u6e1jmsagxyv) in their exam monitoring system. They used algorithms proposed in [[13]](#bookmark=id.xv56g0eq0bu7) to detect faces and do landmark localization. They considered face disappearances as well to detect misconducts.


### 2.2 Analyzing microphone audio

When taking an online examination a candidate may get help from others by having a conversation with them. Therefore we will record the audio during the examination and then we will analyze that.

In the method proposed in paper [[4]](#bookmark=id.7hclkzt9l5rd), initially they record the voice sample of the candidate and then allow him to do the examination. During the examination audio monitor is activated on preset intervals of time or if the candidate is moving his lips. Then a pre-recorded sample is used to ensure any cheating attempts have occurred. Here they are concerned more about speaker recognition than speech recognition. To implement this they have used a speaker model based on [[7]](#bookmark=id.pn5dufwlr5j5). If any voice mismatching has found an event will be triggered that is pre-defined by the examiner like giving warnings or ending the session.


## 2.3 Analyzing keyboard and mouse usage patterns

Another factor in deciding academic dishonesty is the biometric factors such as keyboard and mouse usage which is available from the user side. With the evolution of pattern recognition there are various methods proposed in the literature on this topic.

In this paper[[11]](https://docs.google.com/document/d/1fbOdPrKxNmR4U5He4vnS6p7eqnijnEjUsz-wqt4Ccwo/edit#bookmark=id.pxqpz698fjk) they have suggested different modalities to be used and one of them is the keystroke analysis. Apart from the regular keystroke analysis they have suggested a simple method to reduce its complexity by tracking the question type. For instance for a multiple choice question there should not be much usage of keyboard. So by tracking the keystroke with the question type is going to reduce the complexity of the analysis.

On the other hand mouse usage pattern also plays an important role when detecting the abnormal behaviour of the candidate. many practical studies argue that the required time for data collection of mouse dynamics is very long to complete, But when combining with other biometric techniques the ease of use, popularity, and the high level of transparency of the mouse interface encourages its implementation[[12]](https://docs.google.com/document/d/1fbOdPrKxNmR4U5He4vnS6p7eqnijnEjUsz-wqt4Ccwo/edit#bookmark=id.zib0rwqpb578)

As mentioned in the proposed solution, the feasibility of doing a keyboard and mouse usage analysis is less in our project where it might not be able to cater the mobile devices and the weight on the client side for data collection is also higher.


### 2.4 Analyzing screen capture

Screen capture will be a crucial input when there's a need to analyse the usage of other applications while taking the test. Native browser API allows us to capture a particular application screen or the whole window as specified in the MDN [docs](https://developer.mozilla.org/en-US/docs/Web/API/Screen_Capture_API/Using_Screen_Capture). Literature also proposes to use standalone applications to capture the screen, but with the context being restricted to a browser, we can make use of default browser APIs.

The thesis [23] proposes a CNN (Convolutional Neural network) on students’ screen recordings to determine if they show fraudulent behaviour or not during the examination. The fraudulent activities that they are proposing to detect are the communication with others (chatting or mail) , using search engines. The main drawback in this approach is to capture the complete screen recording since there are chances where the test taker can use completely isolated browser windows. Since the idea is very contextual there was not much related work found on screen capture analysis in terms of fraud detection.


### 2.4 Other methods proposed

Apart from the above modalities for detecting the abnormal behaviours of the candidates, the literature proposes some other methods. Active window detection is a method proposed in the paper[[2]](#bookmark=id.2nef0acyrhv9). They have used the Operating system APIs to track the active window. Most of the time, there should be only one active window, which is the online exam itself. Based on this assumption they track the active windows periodically and use it as another parameter for their final multi-modal for detecting the cheating activities.

In the method proposed in paper [[6]](#bookmark=id.e5hqehst5dje), they expect to capture all the active processes during the examination using a standalone application they developed. So, they can monitor all the system usage and detect any cheating that happens using other softwares.

As these methods use a standalone application developed to take the test they have the advantage of accessing operating system APIs. But normally the tests are taken through a web browser. When we restrict the application to browser limitations we can not perform this kind of monitoring.

## Methodology

## Experiment Setup and Implementation

## Results and Analysis

## Conclusion

## Links










3.
Methodology

4.
Results and Evaluation

## 4.1 Head pose

**4.1.1 head pose estimation and number of faces**

The data collected from the FSA-net is the yaw, roll,pitch angles and the number of faces in the frame. As an example, the dataset [here](https://gist.github.com/irfanm96/2da787192f785822fcff935d55d5b65f) is generated by the FSA-net from a private dataset. It had 1656 images of webcam images. As per the guidelines given by the owner of the original dataset it cannot be published since it is treated as private information of the test taker.



<p id="gdcalert7" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image7.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert8">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image7.png "image_tooltip")


Figure 5: Preview of dataset generated by FSA-net

The dataset obtained were just images without any other information. So we used the trained model of FSA-net implementation to predict the head pose. SInce we do not have the head poses labelled for the dataset given, it is not possible to evaluate the estimation of yaw, roll, and pitch angles.

But since we account for the number of faces in the image we manually labelled each image according to the number of faces in the frame. Out of 1656 images there was no image which contained zero or multiple faces. So all the images contained only one face in the frame.

FSA-net purely deals with the head pose, we can use any face detection algorithm to detect the face first then it can be fed into FSA-net to predict the head pose.

FSA-net uses 3 different methods to identify the faces in the frame, each one has its own advantages such as accuracy and fastness. The following are the methods used for face detection



*   Local Binary Patterns (LBP)

A texture operator which labels the pixels of an image by thresholding the neighborhood of each pixel and considers the result as a binary number.



*   Multi Task Cascaded Neural networks (MTCNN) [31]

A neural network which detects faces and facial landmarks on images.



*   Single Shot Detector (SSD)[32]

The output of the FSA-net on number of faces were different as below,


<table>
  <tr>
   <td>Number of faces
   </td>
   <td>SSD
   </td>
   <td>MTCNN
   </td>
   <td>LBP
   </td>
  </tr>
  <tr>
   <td> 0
   </td>
   <td>10
   </td>
   <td>45
   </td>
   <td>0
   </td>
  </tr>
  <tr>
   <td> 1
   </td>
   <td>1557
   </td>
   <td>1611
   </td>
   <td><strong>1651</strong>
   </td>
  </tr>
  <tr>
   <td> 2
   </td>
   <td>89
   </td>
   <td>0
   </td>
   <td>5
   </td>
  </tr>
</table>


Table 2: Results of number faces detected by different algorithms

So considering the results obtained we could see that there are 99 ( 89+10 ) false positives when using an SSD-face detector. With the MTCNN method there are 45 false positives. All of them detect no face where actually there's a face present in the image. But with LBP method it almost predicted all the images except 5 images being detected with multiple faces. So it falls under the acceptable tolerance ( &lt; 1% ) in terms of the accuracy with the dataset obtained. So we used LBP as the final method to detect the faces and then we feed it to head pose estimation for anomaly detection analysis.

We directly label the images as suspicious, based on the assumption that, if there are more than one face in the frame then it is suspicious. Moreover if the number of faces are zero FSA-net does not perform the head pose estimation. So the filtered out results are going to be fed into the anomaly detection algorithm from the head pose estimation results. That is, the results where it has only one face in the frame. (1651 images).

**4.1.2 Anomaly detection**

Let us consider the DBSCAN algorithm results where we took it as the baseline. The figure below shows how the DBSCAN identified the outlier points.

As preprocessing the data for the DBSCAN modal, is standardized. And following are the parameters fed in to it



*   Radius of the cluster - 0.9
*   Minimum samples in a cluster - 100
*   Number of threads - 01
*   Metric used - Euclidean



<p id="gdcalert8" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image8.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert9">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image8.png "image_tooltip")


Figure 6: Visualization of predicted anomalies and normal points with DBSCAN

The figure 7 shows how the anomaly scores are distributed on  the Isolation forest algorithm on the same dataset. It allows to set up a threshold value for labeling anomalies



<p id="gdcalert9" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image9.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert10">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image9.png "image_tooltip")


Figure 7: Histogram of anomaly score predicted by Isolation Forest

It can be seen that there is a high distribution of normal data.and less contaminated outliers.

So if the cutoff anomaly score is 0, the following figure shows how the anomalies are there among the dataset.



<p id="gdcalert10" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image10.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert11">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image10.png "image_tooltip")


Figure 8: Visualization of predicted anomalies and normal points with Isolation Forest

As we also approached the anomaly detection problem as a one class classification problem, the One class SVM is trained with part of normal data and tested with both normal and abnormal data, the figure 9 depicts the detected anomalies and the normal points. The data points are less when comparing to previous two methods since part of the data is used to train the model.

FIGURE HERE

Figure 9: Visualization of predicted anomalies and normal points with One class SVM

**4.1.3 Labelling and Evaluation**

One of the major challenges we faced is to evaluate the performance of the model based on labeled data. We can compare the model by the percentage of True positives and True negatives , i.e  if it has correctly predicted outliers as outliers while predicting normal points as normal and we can use other matrices too. But the challenge we had is to have labelled data to evaluate the system.

The evaluation is tricky in our case since we use unsupervised learning methods to predict outliers. As mentioned in paper[28], The evaluation methods can be classified into two main categories.



1. **Internal validation**

    Internal validation methods make it possible to establish the quality of the clustering structure without having access to external information (i.e. they are based on the information provided by data used as input to the clustering algorithm).So these evaluation metrics determine how close each point in one cluster is to points in the neighboring clusters. Following are most commonly used evaluation methods.

*   Silhouette coefficient: for partitional algorithms
*   cophenetic coefficient : for hierarchical algorithms



<p id="gdcalert11" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image11.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert12">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image11.png "image_tooltip")



    	Figure 10:  illustration of cohesion and separation


                (source : [28])

2. **External validation**

External validation methods can be associated with a supervised learning problem. External validation proceeds by incorporating additional information in the clustering validation process, i.e. external class labels for the training examples. Since unsupervised learning techniques are primarily used when such information is not available, external validation methods are not used on most clustering problems. However, they can still be applied when external information is available.

We want to compare the result of a clustering algorithm to a potentially different partition of data, which might represent the expert knowledge of the analyst (his experience or intuition), prior knowledge of the data in the form of class labels, the results obtained by another clustering algorithm, or simply a grouping considered to be “correct”.

In order to carry out this analysis, a contingency matrix must be built to evaluate the clusters detected by the algorithm. This contingency matrix contains four terms as shown



<p id="gdcalert12" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image12.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert13">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image12.png "image_tooltip")


So now as in a normal supervised learning problem we can use metrics such as accuracy, precision, recall, etc.

In our context we decided to choose external validation methods to choose which algorithm performs better in detecting outliers, since this particular problem is treated as an outlier detection rather than clustering. The internal validation methods do not allow us to make the final decision.

So to do the external validation we planned to label each image manually as normal and abnormal. We obtained two separate lists of labels for the images. There were conflicting labels as well. For example the 1st person can label an image as normal and the other can label it as abnormal. So to resolve the issue we used a mathematical approach to try out the labelling only for the conflicting labels.

We used the following conditions to label a head pose as abnormal, if it falls under one or more conditions we label them as abnormal



*   |Yaw| > 25°
*   |Pitch| > 5°
*   |Roll| > 10°

The above conditions are obtained by repeated tests on the head pose estimation model.

As a result, we obtained a labelled dataset with the following categorized labels,



*   Normal head poses - 1493
*   Abnormal head poses - 64



With the external validation, the final goal is to mark any abnormal head poses. Most importantly we should not mark a normal head pose as an anomaly. So to confirm that let’s consider accuracy and the false negatives.


<table>
  <tr>
   <td>
   </td>
   <td>DBSCAN
   </td>
   <td>Isolation Forest
   </td>
   <td>OSVM
   </td>
  </tr>
  <tr>
   <td>Accuracy (%)
   </td>
   <td>86.13
   </td>
   <td>89.40
   </td>
   <td>89.77
   </td>
  </tr>
  <tr>
   <td>False Negatives (%)
   </td>
   <td>13.20
   </td>
   <td>9.58
   </td>
   <td>4.35
   </td>
  </tr>
</table>


In Terms of accuracy we can shortlist the Isolation forest model and OVSM, but when comparing both of them it's clear that there are less False negatives in OSVM as a percentage. And the OVSM model accuracy and the false negatives are the average value of  a 4-fold cross validation. The cross validation process is done to ensure that the generalizability of the model.

The above results show that OVSM algorithms have less false negatives. But it can be reduced further. By fine tuning the rules in labelling the data and the parameter tuning of the model

As a conclusion for anomaly detection it can be seen that the one class classifier best suites the purpose of anomaly detection since the nature of the dataset is mostly imbalanced.


## 4.2 Audio analysis

**4.2.1 VoiceFilter**

When comparing the state of art methods for voice filter we were able to obtain the comparison metrics against standard datasets. The metric used to compare the models was a very common metric to evaluate source separation systems [31] called source to distortion ratio (SDR). It is measured using both the clean signal and the enhanced signal. It's an energy ratio and it is expressed in dB. Better models have higher SDR. Because this is an energy ratio between the energy of the target signal contained in the enhanced signal and the energy of the errors that are coming from the interfering speakers and artefacts.

Table: Source to distortion ratio on LibriSpeech for Voice Filter models with various LSTM networks  ([29])


<table>
  <tr>
   <td><strong>VoiceFilter Model</strong>
   </td>
   <td><strong>Mean SDR</strong>
   </td>
   <td><strong>Median SDR</strong>
   </td>
  </tr>
  <tr>
   <td>No VoiceFilter
   </td>
   <td>10.1
   </td>
   <td>2.5
   </td>
  </tr>
  <tr>
   <td>VoiceFilter: no LSTM
   </td>
   <td>11.9
   </td>
   <td>9.7
   </td>
  </tr>
  <tr>
   <td>VoiceFilter: LSTM
   </td>
   <td>15.6
   </td>
   <td>11.3
   </td>
  </tr>
  <tr>
   <td>VoiceFilter: bi-LSTM
   </td>
   <td><strong>17.9</strong>
   </td>
   <td><strong>12.6</strong>
   </td>
  </tr>
  <tr>
   <td>PermInv: bi-LSTM
   </td>
   <td>17.2
   </td>
   <td>11.9
   </td>
  </tr>
</table>


Since [29] uses the VoiceFilter model with bi-LSTM networks it has higher accuracy than other methods. So, we use Voice Filter model [29] with trained models obtained from [30] in our system.

**4.2.2 Voice Activity Detection (VAD)**

As mentioned above we are performing [VAD](https://arxiv.org/pdf/2003.12222.pdf) on the environment audio that is obtained with the help of voice filter model.



<p id="gdcalert13" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image13.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert14">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image13.png "image_tooltip")


_Table: Best achieved results on each respective evaluation condition. Bold marks best result for the respective dataset, while underlined marks second best._

Image source:_ _[30]

As you can see in the above table, In real-world evaluation, GPV-F largely outperforms VAD-C(CRNN based standard VAD model) in terms of frame-level evaluation metrics as well as segment-level ones. Clean refers to audio files without any noises and Syn refers to audio files with artificial noises. In our model we use the GPV-F model since this is the best in real world scenarios.

So we use [VAD](https://arxiv.org/pdf/2003.12222.pdf) along with VoiceFilter to detect cheatings that happens through verbal means.


## 4.3 Screen capture analysis

As discussed in the Methodology, we used VGG16 [33] to classify normal and abnormal screen captures. The following parameters are used in the training phase of the neural network.The parameters are decided based on the size of the dataset (which is very less in count)



*   Epochs : 10
*   Steps per epoch : 3

And with the dataset we have, we divided them into training and testing datasets as described below



*   Training -
*   Testing  -



<p id="gdcalert14" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image14.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert15">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image14.png "image_tooltip")



    	Figure 11:  Results of vgg16 binary classifier

The results of the neural network shows us the lack of training data used. And the accuracy of the model is poor too.

We could not develop this further because of time constraints as we had maximum focus on head pose anomaly detection evaluation since it had issues with detecting multiple or zero faces in the frame. The screencapture analysis will be continued further as it needs more data to train the model. Or we have to use other sampling mechanisms like synthetic data or over sampling to get a better performance of the model.



5.
Conclusion and Future directions
There are various methods proposed in the literature regarding online exam proctoring systems.The literature suggests not to depend on a single modality, rather make a decision based on several modalities. Their evaluation results implies that combining several models and creating a multi-model architecture produces better results.

Processing the webcam video is one of the main aspects when deciding the abnormal behaviour of the candidate. Head pose detection, eye gaze detection, liveness detection are several parameters that could be used for predicting the abnormal behaviors.

Meanwhile using the audio stream we can detect activities such as asking help from a person in the room ( even whispering the answer without being in the camera view ).

When considering our project scope we are constrained to develop a system which runs in the browser and without any additional hardware. Methods proposed in [[2]](#bookmark=id.2nef0acyrhv9),[[6]](#bookmark=id.e5hqehst5dje) are using a standalone application and take advantage of accessing the operating system to monitor the applications that the candidate uses. The idea of wear cam [[2]](#bookmark=id.2nef0acyrhv9) is also a way of getting the vision of the user so it adds more information about the candidate behaviour. Since these methods contradict our project scope these methods cannot be implemented in our system. But considering the other modalities they are most similar and within the scope.

Head Pose estimation and anomaly detection does not use time as a feature to detect anomalies. In future we can improve the algorithm to cater the timestamp into account (treating it as a multivariate time series anomaly detection problem) so that we could detect contextual anomalies as well. Several papers from literature [23], [24] propose Multivariate time series anomaly detection methods.

Moreover the FSA-net does not identify a live human face, in other words it detects the faces even in a picture behind the test taker. We can use face spoofing identification to overcome this issue in future.

Since an examinee may be doing the exam in noisy conditions, when doing the audio analysis we have to consider them as well. For example, if music segments are present in the recording, initially we have to remove it to get a better accuracy. Other than that we have to remove silence segments as well. Because [[21]](#bookmark=id.uodtcmqdd45j)’s diarization model is implemented assuming the audio file consists of speech segments only. Since it has separate models for music segment detection and silence removal we can use these models to overcome the above mentioned issue. Other than that, We will also focus on the possibility of using signal separation techniques such as principal component analysis(PCA) to separate speakers in the recorded audio instead of using speaker diarization.

Another addition to the system can be done with the emotion detection of the test taker based on the logs of test taker actions. It adds more strength to the existing system while using the same input. The paper [26] describes Emotions Annotation Protocol for Machine Learning (EmAP-ML). It learns emotions and behaviours based on the video records. The paper [27] also proposes an annotation methodology to tag facial expression and body movements in an educational context. There is a paradigm shift of Fraud Research Science work from pure, hardcoded feature engineering to a more model-centric approach. This newer approach focuses on modern machine learning techniques such as deep learning that expands the types of sources we can work with and helps us better capture the predictive properties of our signals.

Considering the accomplished work against the milestones proposed for the semester, we were able to achieve two of the proposed milestones completely.



1. Evaluation of head pose anomaly detection
2. Analyze audio input using signal separation techniques

And partly completed the 3rd milestone detecting abnormal fraudulent communication with screen capture analysis. The lack of data and the time consumed for other two milestones affected the 3rd milestone of the project.

In future the project will be taken forward by the screen capture analysis and finally the model to combine all the results of the subsystems will be built so that it will be ready to deliver for the client.


# References



1. Maria Apampa, K., Wills, G., & Argles, D. (n.d.). Towards Security Goals in Summative E-Assessment Security. In _ieeexplore.ieee.org_.
2. Atoum, Y., Chen, L., Liu, A. X., Hsu, S. D. H., & Liu, X. (2015). Automated Online Exam Proctoring. In _ieeexplore.ieee.org_.
3. Cote, M., Jean, F., … A. A.-2016 I. W., & 2016. Video summarization for remote invigilation of online exams. _Ieeexplore.Ieee.Org_.
4. Ahlawat, V., Pareek, V. A., & Singh, V. S. K. (2014). Online Invigilation: A Holistic Approach: Process for Automated Online Invigilation. In _International Journal of Computer Applications_ (Vol. 90, Issue 17).
5. Pan, G., Wu, Z., recognition, L. S.-R. advances in face, & 2008 Liveness detection for face recognition.
6. Reynolds, D., and, R. R.-I. transactions on speech, & 1995.Robust text-independent speaker identification using Gaussian mixture speaker models. _Ieeexplore.Ieee.Org_.
7. Prathish, S., on, K. B.-2016 I. C., & 2016, undefined. (n.d.). An intelligent system for online exam monitoring. _Ieeexplore.Ieee.Org_.
8. Cheung, Y.-M., & Peng, Q. (2015). Eye Gaze Tracking With a Web Camera in a Desktop Environment. _IEEE TRANSACTIONS ON HUMAN-MACHINE SYSTEMS_, _45_(4), 419.
9. Pan, G., Wu, Z., recognition, L. S.-R. advances in face, & 2008, undefined. (n.d.). Liveness detection for face recognition.
10. Kollreider, K., Fronthaler, H., Computing, J. B.-I. and V., & 2009. Non-intrusive liveness detection by face images. _Elsevier_.
11. Zhu, X., & Ramanan, D. (n.d.). Face Detection, Pose Estimation, and Landmark Localization in the Wild. In _ieeexplore.ieee.org_.
12. Asha, S., on, C. C.-2008 I. S., & 2008, undefined. (n.d.). Authentication of e-learners using multimodal biometric technology. _Ieeexplore.Ieee.Org_
13. Ketab, S., Clarke, N., and, P. D.-I. J. of I., & 2017. A Robust E-Invigilation System Employing Multimodal Biometric Authentication. _Ijiet.Org_.
14. Bao, W., Li, H., Li, N., on, W. J.-2009 I. C., & 2009. A liveness detection method for face recognition based on optical flow field. _Ieeexplore.Ieee.Org_.
15. Yee, K., Pedagogy, P. M.-, Policing, N., & 2009. Detecting and preventing cheating during exams. _Academia.Edu_.
16. Yang, T.-Y., Chen, Y.-T., Lin, Y.-Y., & Chuang, Y.-Y. (n.d.). FSA-Net: Learning Fine-Grained Structure Aggregation for Head Pose Estimation from a Single Image.
17. Murphy-Chutorian, E., & Trivedi, M. M. (2009). Head pose estimation in computer vision: A survey. _IEEE Transactions on Pattern Analysis and Machine Intelligence_, _31_(4), 607–626. https://doi.org/10.1109/TPAMI.2008.106
18. Yang, T.-Y., Chen, Y.-T., Lin, Y.-Y., & Chuang, Y.-Y. (n.d.). FSA-Net: Learning Fine-Grained Structure Aggregation for Head Pose Estimation from a Single Image. In _openaccess.thecvf.com_. Retrieved August 7, 2020, from http://openaccess.thecvf.com/content_CVPR_2019/html/Yang_FSA-Net_Learning_Fine-Grained_Structure_Aggregation_for_Head_Pose_Estimation_From_CVPR_2019_paper.html
19. Ruiz, N., Chong, E., & Rehg, J. M. (2017). _Fine-Grained Head Pose Estimation Without Keypoints_.
20. Bredin, H., Yin, R., Coria, J. M., Gelly, G., Korshunov, P., Lavechin, M., Fustes, D., Titeux, H., Bouaziz, W., & Gill, M.-P. (2019). _pyannote.audio: neural building blocks for speaker diarization_. http://arxiv.org/abs/1911.01255
21. Giannakopoulos, T. (2015). pyAudioAnalysis: An Open-Source Python Library for Audio Signal Analysis. _PLOS ONE_, _10_(12), e0144610. https://doi.org/10.1371/journal.pone.0144610
22. [aalto-speech/speaker-diarization: Speaker diarization scripts, based on AaltoASR](https://github.com/aalto-speech/speaker-diarization)
23. Kuin, A. (2018). _Fraud detection in video record-ings of exams using Convolu-tional Neural Networks_. https://esc.fnwi.uva.nl/thesis/centraal/files/f1774806379.pdf
24. Tsay, R. S., Peña, D., & Pankratz, A. E. (2000). Outliers in multivariate time series. _Biometrika_, _87_(4), 789–804. https://doi.org/10.1093/biomet/87.4.789
25. Wang, X. (2011). Two-phase outlier detection in multivariate time series. _Proceedings - 2011 8th International Conference on Fuzzy Systems and Knowledge Discovery, FSKD 2011_, _3_, 1555–1559. [https://doi.org/10.1109/FSKD.2011.6019794](https://doi.org/10.1109/FSKD.2011.6019794)
26. de Morais, F., Kautzmann, T. R., Bittencourt, I. I., & Jaques, P. A. (2019). EmAP-ML: A Protocol of Emotions and Behaviors Annotation for Machine Learning Labels. _Lecture Notes in Computer Science (Including Subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics)_, _11722 LNCS_, 495–509. [https://doi.org/10.1007/978-3-030-29736-7_37](https://doi.org/10.1007/978-3-030-29736-7_37)
27. Saneiro, M., Santos, O. C., Salmeron-Majadas, S., & Boticario, J. G. (2014). Towards emotion detection in educational scenarios from facial expressions and body movements through multimodal approaches. _Scientific World Journal_, _2014_. [https://doi.org/10.1155/2014/484873](https://doi.org/10.1155/2014/484873)
28. Tang, T. W., Kuo, W. H., Lan, J. H., Ding, C. F., Hsu, H., & Young, H. T. (2020). Anomaly detection neural network with dual auto-encoders GAN and its industrial inspection applications. _Sensors (Switzerland)_, _20_(12). [https://doi.org/10.3390/s20123336](https://doi.org/10.3390/s20123336)
29. Wang, Q., Muckenhirn, H., Wilson, K., Sridhar, P., Wu, Z., Hershey, J., Saurous, R. A., Weiss, R. J., Jia, Y., & Moreno, I. L. (2018). VoiceFilter: Targeted voice separation by speaker-conditioned spectrogram masking. In _arXiv[eess.AS]_.http://arxiv.org/abs/1810.04826
30. [Edresson/VoiceSplit: VoiceSplit: Targeted Voice Separation by Speaker-Conditioned Spectrogram](https://github.com/Edresson/VoiceSplit)
31. E. Vincent, R. Gribonval, and C. Févotte, “Performance measurement in blind audio source separation,” IEEE transactions on audio, speech, and language processing, vol. 14, no. 4, pp. 1462–1469, 2006.

( NOTE: EDIT THIS LINKS WITH YOUR REPO DETAILS )

- [Project Repository](https://github.com/cepdnaclk/e15-4yp-online-proctoring-system)
- [Project Page](https://cepdnaclk.github.io/e15-4yp-online-proctoring-system)
- [Department of Computer Engineering](http://www.ce.pdn.ac.lk/)
- [University of Peradeniya](https://eng.pdn.ac.lk/)

[//]: # "Please refer this to learn more about Markdown syntax"
[//]: # "https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet"
