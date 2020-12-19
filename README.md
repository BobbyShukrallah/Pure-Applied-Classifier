# Pure-Applied-Classifier

I develop a NN that predicts titles of papers as either pure or applied.

By "applied", I mean that a title is from a field that requires an understanding of topics that are directly relevant for jobs in inudstry. For by non-academic jobs. For example, papers in statistics, computer science, and probability would be classified as applied, and differential equations, combinatorics, and algebra would be pure.

Notebooks are included to show how to build it on your own (since it requires some memory): a "Cleaning" notebook on cleaning and labeling the training data, and a "Learning" notebook where I train the NN using tensorflow.

Using the Arxiv datadump on Kaggle consisting of metadata for 1.7 million scientific articles postedd to arxiv, I first learned word2vec representations of (bigrammed) terminology used in the abstracts. Using labeled titles both from Arxiv and from Mathscinet (I show how the Arxiv training data is made, but not the mathscinet data), I embedded the titles in Euclidean space using the average of the word2vec mebeddings of the titles, then used these to a NN with one hidden layer. The final model achieves a test accuracy of around 90%, and is provided in the models folder.

