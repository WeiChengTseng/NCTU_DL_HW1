# Deep Learning (Homework 1)
> Due date : 04/12/2019

- Any tools for automatic differentiation are forbidden in this homework, such as Tensorflow, Pytorch, Keras, MXNet, et cetera. You should implement the backpropagation by yourself.
- Submitting Homework – Please zip each of your source code and report into a single compress file and name the file using this format : HW1 StudentID StudentName.zip (rar, 7z, tar.gz, . . . etc are all not acceptable)

## Deep Neural Network for Classification

### Problem 1
![](/result/prob1.png)
#### Initialization
We initialize the weight with a normal distribution N(0, n). Also, we initialize the bias with 0.

### Problem 2
![](/result/prob2.png)
We construct a DNN with the number of neuron [6, 3, 3, 2]. To find the hypermeters, we apply grid search to get the best hyperparameters.


### Problem 3
![](/result/prob3_compared.png)
According to the figures, it is obvious that standardization on scalar feature is helpful. Model with standardization converges faster than that without standardization. We also find that we need to initialize weights with a larger variance to model with standardization. One possible reason for this phenomenon is that the value of **Fare** is much larger than other feature. If the weights are too large, gradient explosion may happen. However, since we standardize the Fare, we can initialize weights with larger variance.

### Problem 4
![](/result/prob4.png)
To find the feature that affects the performance the most, we mute one of the feature at a time. Therefore, we can figure out which feature is most important. According to the figures, we can find that **Pclass**, **Fare** and **Sex** are the most important features. This result is consistent with the history of the sinking of the RMS Titanic. 

At 00:05 on 15 April 1912, Captain Smith ordered the ship's lifeboats uncovered and the passengers mustered. The thoroughness of the muster was heavily dependent on the class of the passengers; the first-class stewards were in charge of only a few cabins, while those responsible for the second- and third-class passengers had to manage large numbers of people. The first-class stewards provided hands-on assistance, helping their charges to get dressed and bringing them out onto the deck. With far more people to deal with, the second- and third-class stewards mostly confined their efforts to throwing open doors and telling passengers to put on lifebelts and come up top. In third class, passengers were largely left to their own devices after being informed of the need to come on deck. Many passengers and crew were reluctant to comply, either refusing to believe that there was a problem or preferring the warmth of the ship's interior to the bitterly cold night air. The passengers were not told that the ship was sinking, though a few noticed that she was listing. Besides, women and children have higher priority to board the lifeboats.

### Problem 5
![](/result/prob5_compared.png)

### Problem 6
![](/result/prob6_Fare.png)
![](/result/prob6_Pclass.png)
![](/result/prob6_Sex.png)

From the figures, we can get three conclusions. First, women (`Sex` = 0) were more likely to survive. Second, the passengers who paid higher passenger fare were more likely to survive. Third, those who has better ticket class were more likely to survive.

Therefore, we create fake data with the conclusions mentioned above.


||Pclass|Sex|Age|SibSp|Parch|Fare|
|-|-|-|-|-|-|-|
|fake survivor|1|0|22|0|0|70|
|fake victim|3|1|42|5|0|5|

survivor = np.array([1, 0, 22, 0, 0, 70.0])
victim = np.array([3, 1, 42, 5, 0, 5.0])