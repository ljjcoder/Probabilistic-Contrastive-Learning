# PCL
Probabilistic Contrastive Learning for Domain Adaptation [here](https://arxiv.org/abs/2111.06021) (official Pytorch implementation). 

#### About code

We have refactored the experimental code (UDA_GVB and SSL_Flexmatch), if you have any problems, please contact us in time.

#### Something to Say:

The full paper is answering three questions:

#### 1. Can we directly use contrastive learning in DA tasks and obtain considerable gains only by relying on the motivation of "contrastive learning can improve feature generalization, robustness, and compactness"?

#### 2. If not, why?

#### 3. How can the contrastive learning loss be adapted to the DA task?

We think that the core problem here is that the current FCL generally ignores the relationship between features and class weights, while PCL explicitly narrows the distance between features and class weights. Based on the above understanding, PCL can be used in all close-set learning tasks, that is, unlabeled data and labeled data belong to the same semantic space. So far, we have verified 5 tasks and we believe that PCL can bring benefits on many more tasks.
#### Therefore, we also expect researchers to verify the effectiveness of PCL in more tasks!!!!!! In addition, I sincerely hope that researchers will focus on the relationship between features and class weights when using contrastive learning loss, and design a more elegant and efficient contrastive learning loss.

#### Personal Feelings:

Readers may feel that using probability for contrastive learning is as natural as breathing, and there is nothing difficult about it, let alone innovation. I understand how you feel, but allow me to defend PCL a little bit.

During my own research period, the process of migrating standard FCL to PCL was fraught with hardships. At first, based on the naive intuition that contrastive learning can improve feature generalization, like many methods in other fields, I applied it directly to the DA task. However, the gains obtained are very limited, which has puzzled me for a long time, and even suspected that contrastive learning is not suitable for domain adaptation tasks. It took me a long time before I realized that there was a problem with the matching of class weights and features. Maybe the readers don't agree, but for myself, realizing this is the most valuable contribution of this paper. Because it gives the way forward, otherwise, it would be difficult for me to stick to the route of contrastive learning. In fact, before realizing this point, my teacher and classmates could not see any hope for this route, and did not know which direction to make efforts, so they suggested me to change to a new direction.

Then, a key question is how to make the features approach the class weights. More importantly, what kind of indicators should we use to measure this closeness. Fortunately, we first realized that the one-hot form can be used as a suitable metric, and thus designed the concise PCL. How to do this is not obvious, it still requires careful thought. Because of this, the details weren't fully elaborated during my early submissions, more than one readers asked "Why are probabilities better than features?" and "Why can one-hot indicate how close features are to class weights?"

Now that the detailed explanation of PCL is completely presented in front of you, all this seems so natural that it is not worth mentioning. However, when I first faced the problem, it was not obvious what was problem with FCL for DA and how to design PCL. For me, this process is more like looking for a faint fire in the vast darkness, and I doubt whether I am going in the right direction every step of the way. Luckily, I came out of the dark and found a solution that was so simple and so effective. For me, PCL is beautiful, and it makes me appreciate the charm of the ancient Chinese adage "The Greatest Truths is Concise". Therefore, we implore readers to forget about the analysis in this paper, set themselves up for the first time in a scenario where the standard FCL is not effective, and try to understand the hardships behind the PCL.

#### If you use this code/method or find it helpful, please cite:


```
@article{li2021semantic,
  title={Semantic-aware representation learning via probability contrastive loss},
  author={Li, Junjie and Zhang, Yixin and Wang, Zilei and Tu, Keyu},
  year={2021}
}
```


 
