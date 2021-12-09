<h1>Essentials of Unsupervised Learning</h1>

In unsupervised learning, the training data is unlabeled. The system tries to learn without a teacher. UL  is often used for clustering, anomaly dectection and novelty detection, visualization and dimentionality reduction, and Association rule learning among other things.

<h2>Clustering</h2>

Say you have a lot of data about your blog’s visitors. You may want to run a clustering algorithm to try and detect groups of similar visitors at no point do you tell the algorithm which group a visitor belongs to: it finds those connections without your help. 

<h2>Visualization</h2>

With visualization algorithms you feed them a lot of complex and unlabeled data and they output a 2d or a 3d representation of your data that can easly be plotted, These algorithms try to preserve as much structure as they can so that you can understand how the data is organized and perhaps identitfy unsuspected patterns. 

<h2>Dimentionally reduction</h2>

The goal is to simplify the data without loosing too much information. One way to do this is to merge several correlation features into one.  For example, a car's mileage may be strongly correlated with it's age, so the dimensionality reduction algorithm will merge them into one feature that represents the car's wear and tear. This is known as feature extraction. 

<h2>Annomonly Detection</h2>

The system is shown mostly normal instances during training, so it learns to recognize them; then when it sees a new instance, it can tell wether it looks like a normal one or wether it is likely an anomoly. 
