# Memorization-and-Privacy-in-Deep-Learning

In 2015, a Kaggle competition was launched, challenging participants to create 10,000 images of dogs using a deep learning model trained on nearly 30,000 real dog images [1]. To assess the quality of the generated images, the competition organizers introduced a new evaluation metric. This metric, named the Memorization-Informed Fréchet Inception Distance (MiFID) [2], builds upon the traditional Fréchet Inception Distance by incorporating a memorization factor. This factor evaluates the degree to which the generated images resemble the training dataset, considering both intentional and unintentional memorization by the generative model. Citing the authors:

"*Specifically, we predicted that memorization of training data would be a major issue, since current generative model evaluation metrics such as IS or FID are prone to assign high scores to models that regurgitate memorized training data...*"

Indeed, traditional metrics can be manipulated by encouraging memorization. For instance, one could simply train a GAN's discriminator solely on real data and then use it to evaluate the generator's performance. This approach bypasses the typical minimax game between the generator and discriminator, allowing for an easier assessment of quality based solely on memorization of the real dataset. However, this metrics can also detect uninentionale memorization, and citing again the authors "*unintentional memorization is a serious and common issue in popular generative models*". So what can be done? Well, memorization is also a matter of privacy, then my initial thought was, **what happens if I apply differential privacy?**

## Differential Private in Deep Learning

dd











# References
[1] OlgaRussakovsky,JiaDeng,HaoSu,JonathanKrause,SanjeevSatheesh,Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. 2015. Imagenet large scale visual recognition challenge. International journal of computer vision 115, 3 (2015).
[2] Bai, Ching-Yuan, et al. "On training sample memorization: Lessons from benchmarking generative modeling with a large-scale competition." Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 2021.
