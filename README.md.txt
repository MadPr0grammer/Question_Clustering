++ Information about folders:
		+ Inside the 'Codes' folder:
			- 'Data' folder has the processed dataset required for training doc2vec. 
			- 'Plots' folder has all the plots
			- 'Python_Scripts' folder has all the code files
			- 'Saved_Files' folder has the files I saved after running algorithms like K-Means, PCA, T-SNE etc. on the dataset.
			- 'Trained_Models' folder link:
					https://drive.google.com/drive/folders/1IK3vOQVv_qqDIaZPKAr73C3oSgTpcONQ?usp=sharing

		+ Inside the 'Datasets' folder:
			- There are 2 files, the description of which is explained in the Dataset section.

		+ The report is by the name NLP_Project_Report.pdf 

++ Dataset:
		+ The dataset is in the folder 'Datasets'.
		+ There are 2 files in it.
		==>The file 'Quora_Kaggle_Dataset.csv' is the dataset downloaded from Kaggle.
			Format: it's a csv file with the following header:
				"id","qid1","qid2","question1","question2","is_duplicate?"


		==>The file 'Dataset_created_by_us_using_quora_dataset.txt' is made from the kaggle dataset by us by extracting the questions from the kaggle dataset.
			Format: One question per line.

++ Implementation Details:
		+ The vectors corresponding to questions are extracted using doc2vec of the 'gensim' library
		+ There are two kinds of vectors, 50-dimensional and 300-dimensional
		+ After extracting vectors, K-Means algorithm is run on both types of vectors.
		+ We tried to find 5-clusters and 8-clusters on the dataset
		+ After getting labels of each datapoint we plotted them by reducing the dimension of features to 2 using PCA and TSNE

