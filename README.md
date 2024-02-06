# Memorization-and-Privacy-in-Deep-Learning

In 2015, a Kaggle competition was launched, challenging participants to create 10,000 images of dogs using a deep learning model trained on nearly 30,000 real dog images [1]. To assess the quality of the generated images, the competition organizers introduced a new evaluation metric. This metric, named the Memorization-Informed Fréchet Inception Distance (MiFID) [2], builds upon the traditional Fréchet Inception Distance by incorporating a memorization factor. This factor evaluates the degree to which the generated images resemble the training dataset, considering both intentional and unintentional memorization by the generative model. Citing the authors:

"*Specifically, we predicted that memorization of training data would be a major issue, since current generative model evaluation metrics such as IS or FID are prone to assign high scores to models that regurgitate memorized training data...*"

Indeed, traditional metrics can be manipulated by encouraging memorization. For instance, one could simply train a GAN's discriminator solely on real data and then use it to evaluate the generator's performance. This approach bypasses the typical minimax game between the generator and discriminator, allowing for an easier assessment of quality based solely on memorization of the real dataset. However, this metric can also detect unintentional memorization: citing again the authors "*unintentional memorization is a serious and common issue in popular generative models*". So what can be done? Well, memorization is also a matter of privacy, then my initial thought was, **what happens if I apply differential privacy?**

## Differential Private in Deep Learning
Differential privacy is a mathematical definition of privacy based on statistical indistinguishability [3]. The two parameters are $\varepsilon>0$ and $\delta\in[0,1]$ and the closer the are to zero the more the algorithm is private. In a few words, an algorithm (randomized) applied to a dataset $X$ is private if its output is $(\varepsilon, \delta)$-indistinguishable by any other possible output obtained from a neighboring dataset $X'$ differing in one user form $X$. So the key concept is **we can render things private by adding noise or in general randomization**.

For Deep Learning or in generale Machine Learning, the techniqued used to privately train a model is by using **Differential Private Stochastic Gradient Descent (DP-SDG)** [4]. It basically consists in adding controlled Gaussian noise to each gradient during the training. In this project I decided to train a DCGAN privately using Opacus, a library based on PyTorch that allows an easy implementation of DP-SGD.

### Notes
The DCGAN cannot be constructe using Batch-Normalization layers, as it is not possible to add noise on the gradients. I implemeneted GroupNormalization layers instead.

## Results
With the [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset, we obtained

  MiFID non private = 2460\\
  FID non private = 1.58

  MiFID private = 535\\
  FID private = 4.80

The findings indicate that the non-private model produces higher-quality images, demonstrated by a superior FID score. However, the private model exhibits reduced data memorization, as evidenced by its lower MiFID. Therefore, if minimizing memorization is the main goal, employing differential privacy is a viable strategy.

### Hyperparameters
To decrease computational demands, I generated only grayscale images within a 32-dimensional feature space. The model's learning rate is set to 0.00005, with a batch size of 128, and it undergoes training for 3 epochs. Although it's recognized that generative models typically require training over a larger number of epochs, the memory-intensive nature of DP-SGD makes it unfeasible to conduct extended training on my Apple M1 Pro machine.


# References
[1] OlgaRussakovsky,JiaDeng,HaoSu,JonathanKrause,SanjeevSatheesh,Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. 2015. Imagenet large scale visual recognition challenge. International journal of computer vision 115, 3 (2015).

[2] Bai, Ching-Yuan, et al. "On training sample memorization: Lessons from benchmarking generative modeling with a large-scale competition." Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 2021.

[3] Dwork, Cynthia, and Aaron Roth. "The algorithmic foundations of differential privacy." Foundations and Trends® in Theoretical Computer Science 9.3–4 (2014): 211-407.

[4] Abadi, Martin, et al. "Deep learning with differential privacy." Proceedings of the 2016 ACM SIGSAC conference on computer and communications security. 2016.
