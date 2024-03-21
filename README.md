## A Light-weight TMF Document Classifier
Author: Chihming Chen
### Executive summary
This exploratory research aims to study the feasibility of using document titles to classify documents per Clinical Data Interchange Standards Consortium (CDISC) Trial Master Files (TMF) Reference Model. TMFs are a crucial aspect of clinical trials, serving as the official repository for all the documents that allow for the conduct of the trial, as well as the recording of the intentions, processes, and outcomes. These files are essential for ensuring compliance with regulatory, ethics requirements and good clinical practice (GCP) standards. The CDISC TMF Reference Model is a standardized framework for organizing and managing these documents. This Reference Model serves as an industry best practice and provides a comprehensive structure to ensure that TMF documentation is complete, well-organized, and compliant with regulatory expectations.
### Rationale
The TMF Reference Model defines 249 document types (a.k.a. Artifacts). It may be possible to use a Large Language Model (LLM) to summarize the documents and attempt to classify them. However, such an approach is likely to incur large costs since one either has to have powerful and expensive hardware (GPU) on-premises or pay commercial LLM service providers per token (word) for utilizing their computing power and service. The number of documents produced from a clinical trial can range from hundreds to thousands. For large, multi-center international clinical trials, the total document volume can reach hundreds of thousands of pages or tens of millions of tokens. If it’s successful, this light-weight TMF document classifier running on a consumer-grade hardware, using only short document titles as input, should provide efficiency and financial benefits even if it can only accurately classify a portion, say 30% or 40% of the TMFs produced in a clinical trial.
### Research Question
This research tries to explore the effectiveness of using short document titles to classify documents where the *purposes* of the documents and the organization of the documents are well defined in a regulated industry, while the titles of the documents are not mandated and may vary according to an organization's standard operating procedures and its use of specific document templates.
### Data Sources
The data sources consist of the public <a href=https://www.cdisc.org/tmf target='_blank'>CDISC TMF Reference Model</a>, OpenAI GPT-4 augmented training data, and proprietary dataset containing labeled titles extracted from scanned and Optical Character Recognized PDF files. 
### Methodology
#### Exploratory Data Analysis
**Training Data:** The research will start with the TMF Reference Model where the recommended artifact, and sub-artifact titles and their purposes are described. The model will be pre-trained on this public dataset first. Synthetic data produced by OpenAI’s GPT-4 is also incorporated into the training data. GPT-4 is prompted to produce the 10 most common titles for a given artifact (class) that fits the purpose of the artifacts within the context of TMF Reference Model and Good Clinical Practices.  The synthetic training data also alleviates the class imbalance issue where the numbers of sub-artifacts for individual artifacts range from 1 to 18. 

<img src="https://github.com/chihming-chen/light-weight-TMF-classifier/blob/main/images/class_imbalance.png"  align='center'>

#### Oversampling of Minority Classes (or Document Types):
Despite the distribution of document classes in the training dataset having a good central tendency around the mean and median, the class imbalance will affect some machine learning models that have a bias toward the majority classes. Therefore, the training data is augmented with oversampling of minority classes to balance the distribution of document classes so that each class is represented from 15 to 18 instances in the training dataset.
#### Feature Engineering
It may appear to the contrary, machine learning algorithms do not natively work with text. The Term Frequency - Inverse Document Frequency (TF-IDF) technique is adopted and used as a numeric input to the machine learning algorithms. Simply put, a TF-IDF score represents the importance and distinctiveness of a word or phrase for identifying or classifying text - the document titles in our case. The TF-IDF process turns the entire input corpus into 3,551 unigrams (single words) or bi-grams (sequences of two words). These n-grams are considered features for machine learning models. Their numeric TF-IDF scores are the input to the models.

<img src="https://github.com/chihming-chen/light-weight-TMF-classifier/blob/main/images/TF-IDF_distribution.png" align='center'>

The TF-IDF scores have a great central tendency - typically beneficial for machine learning models to make unbiased predictions. There are some 'outliers' exceeding a statistical upper bound. However, these 'outliers' are welcome signs indicating some of n-grams are extremely distinctive for idenfyinig docment types.

#### Model Selections
Four base models, K-nearest Neighbors, Logistic Regression, Support Vector Machine, and Random Forest classifiers are selected and individually tuned with the 'best' hyperparameters to classify (or predict) the document types, given the document titles as input. Each model has its own way to tackle this 200-plus-class classification problem - either by the proximality to other data points, the separability of data clusters, or by making successive rule-based binary decisions. Common to these models, they provide confidence scores (probabilities) along with their sole-winner predictions. 

Adopting the "wisdom of the crowd" principle, a final Voting Classifier aggregates the base models' predictions, specifically, the confidence scores, and makes the final prediction based on the (weighted) total probabilities. The document class that has the highest weighted, aggregated probability from all base models is the final prediction. The final prediction may be the same as the base models' majority vote, or it may be totally different from any of the base models' first-choice predictions. It is possible that a runner-up prediction candidate becomes the final prediction/classification of the voting classifier.
### Model Evaluation
#### Training Scores
All base models achieve 99.3% to 99.5% accuracy on the training data. It is a desirable outcome since I expect if the TMFs collected from a clinical trial that adopts the TMF Reference Model, these models should achieve almost 100% accuracy.

#### Test Scores on Unseen Data
When tested against 439 unseen field data from a real clinical trial, the TMF Reference Model-trained model scores 52.16% accuracy. It is a big drop from the accuracy score on the training data. However, it is much better than I originally set my expectations (30% to 40%). This is a great result, considering a random guess among 200+ classes would achieve not more than 0.5% accuracy.  The Logistic Regression model is the only base model that crosses the 50% mark on the unseen data.
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
Many of the document types defined in the Reference Model are entity specific. For example, to show a person is qualified to perform a clinical trial related task, the documentation showing the qualification of the person, such as a medical license or Curriculum Vitae is collected and becomes a part of the TMFs. The Reference Model defines 'Principal Investigator's Curriculum Vitae (CV)', 'Sub-investigator’s CV', 'Coordinating investigator's CV', 'Laboratory Staff's CV' and 'Committee Member's CV' as different document types. However, in practice, the position of the qualified person rarely shows up on a CV. Another example, for 'User Requirement Specifications' or 'Audit Certificate', they all belong to different document classes depending on what kinds of systems are involved. Again, in real practice, the types of systems may not always show up in the title of a document. Simply put, the title of a document alone does not always be able to distinguish these entity-specific document types.

Therefore, I adopt a "Top-n Accuracy" scoring scheme to better assess the performance of the model under these circumstances. Instead of using the single outcome prediction, I consider the top 3 or top 5 probabilities produced by a model as the prediction of the model. Specifically, if one of the top-3 or top-5 prediction candidates matches the correct classification, it is considered a correct classification by the model. (This is the main reason why I have chosen these base models.)

The final Voting Classifier model achieves an 82.2% top-3 accuracy score, a substantial increase from the 52.2% top-1 accuracy, and it's much better than the baseline top-3 class distribution of 51.9%. The model's top-5 accuracy is 88.6% outperforming the dummy classifier's top-5 accuracy of 59.9% with a great margin (A dummy classifier always predicts the top-n majority classes in a dataset). The chart below shows the top-n accuracy of the base models and the final voting classifier.
<img src="https://github.com/chihming-chen/light-weight-TMF-classifier/blob/main/images/top-n-accuracy.png" align='center'>

It is worth noting that the K-nearest Neighbors (KNN) model jumps to be the best, among the base models, top-3 and top-5 accuracy model. It suggests the KNN model excels in assessing the runner-up prediction candidates better than the other base models.
#### Model Performance:
The voting classifier, an ensemble of the base models, outperforms the individual base models in all aspects. The KNN model is the winner in top-3 and top-5 accuracy scores among the base models. Its top-3 and top-5 accuracy comes close to that of the voting classifier. However, it is the slowest model as depicted below.

<img src="https://github.com/chihming-chen/light-weight-TMF-classifier/blob/main/images/accuracy_vs_time.png" align='center'>

### Conclusions and Further Considerations
This feasibility study has answered the research question I set out to solve. It shows that using the title of a document to classify the document, among 200+ possibilities in the context of CDISC Trial Master Files, is not only feasible but achieves a great result. Over 52% of the 439 documents from a real clinical trial can be correctly classified by their titles alone. However, this has its limit and just a first step towards a more practical solution. From this experiment, I also learned:
-	The 'wisdom of the crowd' principle and approach prove to be beneficial to tackle this classification problem. The final voting classifier outperforms all individual base models, including the Random Forest model that itself is an ensemble of Decision Tree models. Additional base models, such as a Recurrent Neural Network model, can be explored and incorporated to see whether they increase the model predictive power.
-	The almost-cut-in-half drop in the accuracy scores between the training dataset and unseen data strongly suggests a mismatch between the field data and the TMF Reference Model. Knowing that the TMF Reference Model is not a comprehensive list of titles of documents produced in clinical trials, additional real-word training data is needed before putting the model in real practice.
-	Since a large portion of the TMF document types are entity specific, additional input features, tools and techniques are needed to recognize the entities involved for specific document types in order to produce an acceptable top-1 predictive model. Named Entity Recognition and external database look up are good directions to investigate.

#### Contact and for Further Information
Email: chihming168.chen@gmail.com
