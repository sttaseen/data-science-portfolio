# Data Science Portfolio
This repository contains some of my completed data science projects that I find interesting and commented enough to be usable as notes in the future.

_Note: Some data used in this project is fake and created only for practise purposes._

## Instructions for Running Python Notebooks Locally
1. Install dependencies using requirements.txt:
```
pip install -r requirements.txt
```
2. Run notebooks as usual by using a jupyter notebook server, vscode etc. Install jupyter notebook by using pip:
```
pip install notebook
```
To run jupyter notebook, use:
```
jupyter notebook
```

## Contents

- ### Machine Learning

	- [Decision Trees: LendingClub Loan Repayment](https://github.com/sttaseen/data-science-portfolio/blob/ad669c37a23218d422c88051d4c1087fdc9d794a/Supervised%20Machine%20Learning/Decision%20Tree%20and%20Random%20Forests/Decision%20Trees%20and%20Random%20Forest%20-%20LendingClub.ipynb): A model to predict if a loan will be repaid by a debtor using decision trees on real dataset on lendingclub.com from Kaggle.
	- [KMeansClustering: Cluster Public/Private University Students](https://github.com/sttaseen/data-science-portfolio/blob/ad669c37a23218d422c88051d4c1087fdc9d794a/Supervised%20Machine%20Learning/KNN/K%20Nearest%20Neighbors%20-%20Unlabelled%20Dataset.ipynb): Testing out K Means Clustering algorithm to build a model that accurately clusters students from private and public universities.
	- [Recommender System: Movie Recommender](https://github.com/sttaseen/data-science-portfolio/blob/ad669c37a23218d422c88051d4c1087fdc9d794a/Supervised%20Machine%20Learning/Recommender%20Systems/Movie%20Recommender%20System.ipynb): A supervised model to recommend movies based on users and their reviews.
	- [Reinforcement Learning: Frozen Lake Environment](https://github.com/sttaseen/Deep-Q-Learning): Implementing an optimized Q-Learning agent that will navigate a non-deterministic environment with a fairly high success rate by using reinforcement learning.

	_Tools: scikit-learn, Pandas, Seaborn, Matplotlib, Pytorch, Pygame_

- ### Natural Language Processing

	- [Yelp Reviews](https://github.com/sttaseen/data-science-portfolio/blob/ad669c37a23218d422c88051d4c1087fdc9d794a/Natural%20Language%20Processing/Yelp%20Reviews%20-%20NLP.ipynb): Classifying the star rating of a yelp review based on the text that the review contains.

- ### Deep Learning

	- [DNN Approach: LendingClub Loan Repayment](https://github.com/sttaseen/data-science-portfolio/blob/ad669c37a23218d422c88051d4c1087fdc9d794a/Deep%20Learning/Lending%20Club/Lending%20Club%20-%20DNN%20Approach.ipynb): Trying out a deep learning approach to model if a debtor would or would not repay their debt based on their personal information.
	- [DNN: King County House Prices](https://github.com/sttaseen/data-science-portfolio/blob/ad669c37a23218d422c88051d4c1087fdc9d794a/Deep%20Learning/Lending%20Club/House%20Prices/King%20County%20House%20Prices.ipynb): Using deep learning to predict house prices in King County based on house features like location, number of bedrooms, space etc.

	_Tools: Pandas, Seaborn, Matplotlib, scikit-learn_

- ### Data Analysis and Visualisation
	- __Python__
		- [Ecommerce Customer Time Spread Analysis](https://github.com/sttaseen/data-science-portfolio/blob/ad669c37a23218d422c88051d4c1087fdc9d794a/Supervised%20Machine%20Learning/Linear%20Regression/Ecommerce%20Company%20-%20Time%20Spent.ipynb): Analysis of time spent by users in-store, on the company's mobile app and on their website to determine needed improvements for the company.
		- [Ad-Click Based on Personal Info](https://github.com/sttaseen/data-science-portfolio/blob/ad669c37a23218d422c88051d4c1087fdc9d794a/Supervised%20Machine%20Learning/Logistic%20Regression/Ecommerce%20Logistic%20Regression.ipynb): Analysis of the likelihood of an ad being clicked based on the features of the ad and the personal info of the user.

		
	_Tools: Pandas, Seaborn and Matplotlib_

	- __R__ 
		As a part of part-II software engineering, I have done some data analysis in R [here](https://github.com/sttaseen/R-Exercises). These are uncommented and not pleasing to the eye. I will format these in the future to be readable.

- ### Micro Projects: 
	- [Benford's Law: US Elections](https://github.com/sttaseen/data-science-portfolio/blob/ad669c37a23218d422c88051d4c1087fdc9d794a/Other%20Projects/Benfords_Law_-_Elections.html): Graphing different US Presidential elections to see if Benford's Law applies to them. The file needs to be saved and run as HTML in browser.
	- [Adding Numbers Using NN](https://github.com/sttaseen/add-multiply-NN/blob/ef73029e9a570a0536dc4e89764b44b5bf22d576/Teaching%20a%20Machine%20How%20To%20Add.ipynb): Optimising the number of nodes required to add two numbers.
	- [Multiplying Numbers Using NN](https://github.com/sttaseen/add-multiply-NN/blob/ef73029e9a570a0536dc4e89764b44b5bf22d576/Teaching%20a%20Machine%20How%20to%20Multiply.ipynb): Limit testing concepts to implement a simple neural network that can multiply numbers.
	
	_Tools: Pandas, and scikit-learn_
	
- ### Computer Vision: 
	- [Object Detection: Sign Language](https://github.com/sttaseen/object-detector.git): Training a tensorflow object detection model to read sign language and translate to text in real-time (currently working on it).
	
	_Tools: Pandas, Numpy, Opencv2 and Tensorflow_
	

