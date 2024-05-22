# PCL
Probabilistic Contrastive Learning for Domain Adaptation [here](https://arxiv.org/abs/2111.06021) (official Pytorch implementation). 


:bell: **News:**
* [2024-04-17] After three years of repeated revisions, we are happy to announce that PCL was accepted at **IJCAI2024**.

#### About Code

We have refactored the experimental code (UDA_GVB and SSL_Flexmatch), if you have any problems, please contact us in time.

#### Something to Say:

In order to make it easier for readers to understand our work, here I briefly outline the problems that we want to solve.


Overall, we want to answer the following three questions:

#### 1. Can we directly use contrastive learning in DA tasks and obtain considerable gains only by relying on the motivation of "contrastive learning can improve feature generalization, robustness, and compactness"?

#### 2. If not, why?

#### 3. How can the contrastive learning loss be adapted to the DA task?

We think that the core problem here is that the current FCL generally ignores the relationship between features and class weights, while PCL explicitly narrows the distance between features and class weights. Based on the above understanding, PCL can be used in all close-set learning tasks, that is, unlabeled data and labeled data belong to the same semantic space. So far, we have verified 5 tasks and we believe that PCL can bring benefits on many more tasks.
#### Therefore, we also expect researchers to verify the effectiveness of PCL in more tasks!!!!!! In addition, I sincerely hope that researchers will focus on the relationship between features and class weights when using contrastive learning loss, and design a more elegant and efficient contrastive learning loss.

#### Personal Feelings:
This article goes through a very bumpy submission process. In the previous submission process, due to writing or other reasons, readers do not understand why FCL performed poorly and why the one-hot form could describe the closeness of features and class weights. However, when everything is presented in vernacular form, readers feel that using probability for contrastive learning is as natural as breathing, and there is nothing difficult about it, let alone innovation.
I understand how you feel, but allow me to defend PCL a little bit.

I must admit that PCL is not difficult in terms of the complexity of the operation and the depth of the analysis, and it is difficult to say how valuable it is. But when I first faced the problem of FCL's poor performance in DA tasks, the uncertainty I faced made my research feel like an abyss. Every failure brings with it a painful decision: Is this path worth pursuing?  I will talk about this research process below. 


* 1 Confidence is more important than gold

In fact, at the beginning of this research, I did not think clearly why we should use contrastive learning in DA. Our idea is very simple. Driven by work such as MOCO and simclr, contrastive learning has had a huge impact on the computer vision community. Using contrastive learning loss in many downstream tasks can bring significant gains. For the purpose of catching up with hot topics, I also want to try contrastive learning in the DA field.
However, the gains obtained are very limited, which has puzzled me for a long time. After that, I also tried many commonly used techniques, such as carefully selecting positive and negative samples, but with little success. This made me suspect that contrastive learning itself might be a dispensable technology for DA tasks. This straightforward experience of using contrastive learning also made my teachers and classmates lose confidence in the role of contrastive learning in domain adaptation tasks, and suggested a different direction. I didn’t know why at the time, but I still insisted that contrastive learning would be effective, even though there was no definite basis for this insistence. After that, I embarked on a lonely journey, and every step seemed to remind me that it was time to change direction.


* 2 A sudden flash of inspiration

After I tried a series of commonly used contrastive learning improvement techniques, one day, I suddenly realized that the deviation of features and weights may be the reason for the ineffectiveness of contrastive learning in DA tasks. Maybe the readers don't agree, but for myself, realizing this is the most valuable contribution of this paper. Because it gives the way forward, otherwise, it would be difficult for me to stick to the route of contrastive learning. 

Then, a key question is how to make the features approach the class weights. More importantly, what kind of indicators should we use to measure this closeness. Fortunately, we first realized that the one-hot form can be used as a suitable metric, and thus designed the concise PCL.

Now,  when these things are clearly explained, they seem so natural and not worth mentioning. However, when I faced this problem, many things were unclear. Does contrastive learning really work? What's the reason it doesn't work? How can we shorten the distance between features and weights? These issues pose great challenges to me. For me, this process is more like looking for a faint fire in the vast darkness, and I doubt whether I am going in the right direction every step of the way. Luckily, I came out of the dark and found a solution that was so simple and so effective. From my personal experience, PCL is beautiful, and it makes me appreciate the charm of the ancient Chinese adage "The Greatest Truths is Concise". Therefore, we implore readers to forget about the analysis in this paper, set themselves up for the first time in a scenario where the standard FCL is not effective, and try to understand the hardships behind the PCL.

#### If you use this code/method or find it helpful, please cite:


```
@article{li2021semantic,
  title={Semantic-aware representation learning via probability contrastive loss},
  author={Li, Junjie and Zhang, Yixin and Wang, Zilei and Tu, Keyu},
  year={2021}
}
```


 
