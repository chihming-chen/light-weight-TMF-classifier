## A Light-weight TMF Document Classifier
Author: Chihming Chen
### Executive summary
This exploratory research aims to study the feasibility of using document titles to classify documents per Clinical Data Interchange Standards Consortium (CDISC) Trial Master Files (TMF) Reference Model that defines 249 document types that are collected during clinical trials.
### What Are Clinical Trial Master Files and the TMF Reference Model
TMFs are a crucial aspect of clinical trials, serving as the official repository for all the documents that allow for the conduct of the trial, as well as the recording of the intentions, processes, and outcomes. These files are essential for ensuring compliance with regulatory, ethics requirements and good clinical practice (GCP) standards. The CDISC TMF Reference Model is a standardized framework for organizing and managing these documents. This Reference Model serves as an industry best practice and provides a comprehensive structure to ensure that TMF documentation is complete, well-organized, and compliant with regulatory expectations.
### Rationale
The TMF Reference Model defines 249 document (a.k.a. artifact) types. It may be possible to use a Large Language Model (LLM) to summarize the documents and attempt to classify them. However, such an approach is likely to incur large costs since one either has to have powerful and expensive hardware (GPU) on-premises or pay commercial LLM service providers per token (word) for utilizing their computing power and service. The number of documents produced from a clinical trial can range from hundreds to thousands. For large, multi-center international clinical trials, the total document volume can reach hundreds of thousands of pages or tens of millions of tokens. If it’s successful, this light-weight multi-class document classifier running on a consumer-grade hardware, using only short document titles as input, should provide efficiency and financial benefits even if it can only accurately classify a portion, say 30% or 40% of the TMFs produced in a clinical trial.
### Research Question
This research tries to explore the effectiveness of using short document titles to classify documents where the *purposes* of the documents and the organization of the documents are well defined in a regulated industry, while the titles of the documents are not mandated and may vary according to an organization's standard operating procedures and its use of specific document templates.
### Data Sources
The data sources consist of the public CDISC TMF Reference Model, OpenAI GPT-4 augmented training data, and proprietary dataset containing labeled artifact titles extracted from scanned and Optical Character Recognized PDF files. Due to limited field data at present time, OpenAI’s GPT-4 is prompted to produce synthetic document titles as labeled data for model training purposes.
### Methodology
#### Exploratory Data Analysis
**Pre-training:** The research will start with the TMF Reference Model where the recommended artifact, and sub-artifact titles and their purposes are described. The model will be pre-trained on this public dataset first. Synthetic data produced by OpenAI’s GPT-4 is also incorporated into the training data. GPT-4 is prompted to produce the 10 most common titles for a given artifact (class) that fits the purpose of the artifacts within the context of TMF Reference Model and Good Clinical Practices.  The synthetic training data also alleviates the class imbalance issue where the number of sub-artifacts for an artifact range from 1 to 18. 
<img src="https://github.com/chihming-chen/light-weight-TMF-classifier/blob/main/images/class_imbalance.png"  align='center'>

Finally, the pre-trained model is tested with unseen and labeled field data.
#### Oversampling of Minority Classes (or Document Types):
Despite the distribution of document classes in the training dataset having a good central tendency around the mean and median, the class imbalance will affect some predicting models that have a bias toward the majority classes. Therefore, the training data is augmented with oversampling of minority classes to balance the distribution of document classes so that each class is represented from 15 to 18 instances in the training dataset.
#### Feature Engineering
It may appear to the contrary, machine learning algorithms do not natively work with text. The Term Frequency - Inverse Document Frequency (TF-IDF) technique is adopted and used as a numeric input to machine learning algorithms. Simply put, a TF-IDF score represents the importance and distinctiveness of a word or phrase for identifying or classifying text, the document title in our case. There are 3,551 total unigrams (single words) or bi-grams (sequences of two words) in the entire corpus of the training dataset. These n-grams are considered features for machine learning models. Their TF-IDF scores are the input to the models.
<img src="https://github.com/chihming-chen/light-weight-TMF-classifier/blob/main/images/TF-IDF_distribution.png" align='center'>
#### Model Selection and Compilation
Four base models, K-nearest Neighbors, Logistic Regression, Support Vector Machine, and Random Forest classifiers are selected and individually tuned with the 'best' hyperparameters to classify (or predict) the document types, given the document titles as input. Each model has its own way to tackle this 200-class classification problem. Common to these models, they provide confidence scores (probabilities) along with their sole-winner predictions. These four models may not always make the same prediction,  given the same input. 

Adopting the "wisdom of the crowd" principle, a final Voting Classifier aggregates the base models' predictions, specifically, the confidence scores, and makes the final prediction based on the (weighted) total probabilities. The document class that has the highest weighted total probability from all based models is the final prediction. The final prediction may be the majority vote, be totally different from any of the base models' predictions, or something in between.
### Findings
#### Training Scores
The initial results show 99.3% to 99.5% accuracy, among the four base models, on tokenized, lemmatized and stemmed sub-artifact titles defined in the public TMF Reference Model with additional synthetic training data.

#### Test Scores on Unseen Data
When tested against 439 unseen field data, the TMF Reference Model-trained model scores 52.16% accuracy - much better than I originally expected. This is a great result, considering a random guess would achieve only 0.5% accuracy.  
<pre>
                       Train score	Test score	Avg. model eval time
Classifier			
K-Nearest Neighbors	  0.995387	0.325740	0.289554
Logistic Regression	  0.995387	0.501139	109.099880
Support Vecor Machine	  0.993081	0.451025	1.681580
Random Forest	          0.995387	0.312073	27.978212
Voting Classifier	  0.995387	0.521640	19.444089
</pre>

#### Top-n Accuracy:
Many of the document types defined in the TMF Reference Model are entity specific. For example, to show a person is qualified to perform a clinical trial related task, the documentation showing the qualification of the person, such as a medical license or Curriculum Vitae is collected and become a part of the TMFs. The Reference Model defines 'Principal Investigator's Curriculum Vitae (CV)', 'Sub-investigator’s CV', 'Coordinating investigator's CV', and 'Committee Member's CV' as different document types. However, in practice, the position of the person rarely shows up on a CV. Also, for 'User Requirement Specifications' or 'Audit Certificates', they all belong to different document classes depending on what kinds of systems are involved. And, again, in real practice, the types of systems may not always show up in the title of a document. Simply put, the title of a document alone does not always be able to distinguish these entity-specific classes.

Therefore, I adopt a "Top-n Accuracy" scoring scheme to better assess the performance of the model. Instead of using the single outcome prediction, I consider the top 3 or top 5 probabilities produced by the model as the prediction of the model. In other words, if one of the top 3 or top 5 prediction candidates matches the correct classification, it is considered a correct classification by the model. The Voting Classifier model achieves an 82.2% top-3 accuracy score, a substantial increase from the 52.2% top-1 accuracy. The model's top-5 accuracy is 88.6%. The chart below shows the top-n accuracy of the base and voting models.
<img src="https://github.com/chihming-chen/light-weight-TMF-classifier/blob/main/images/top-n-accuracy.png" align='center'>

It is worth noting that the K-nearest Neighbors model jumps to be the best, among the base models, top-3 and top-5 accuracy model. It suggests the KNN model excels in assessing the runner-up candidates better than the other base models.
#### Model Performance:
<img src="https://github.com/chihming-chen/light-weight-TMF-classifier/blob/main/images/accuracy_vs_time.png" align='center'>

### Next steps
-	Expanding the scale of synthetic data 
-	Additional field data for testing, model building, and model evaluation
-	Model selection and hyperparameter tuning
### Outline of project
- Jupyter Notebook 1 [Building the Pre-trained Models](TMF%20Classifier.ipynb) (start here)
- Jupyter Notebook 2 [Testing against Unseen Field Data and Continuous Improvement](TMF%20Classifier%20Field%20Test.ipynb)

#### Contact and for Further Information
Email: chihming168.chen@gmail.com
