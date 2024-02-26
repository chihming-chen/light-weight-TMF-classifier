## A Light-weight TMF Document Classifier
Author: Chihming Chen
### Executive summary
This exploratory research aims to study the feasibility of using document titles to classify documents per Clinical Data Interchange Standards Consortium (CDISC) Trial Master File (TMF) Reference Model that defines 249 document types that may be collected during clincal trials.
### What Are Trial Master Files and the TMF Reference Model
TMFs are a crucial aspect of clinical trials, serving as the official repository for all the documents that allow for the conduct of the trial, as well as the recording of the intentions, processes, and outcomes. These files are essential for ensuring compliance with regulatory requirements and good clinical practice (GCP) standards. The CDISC TMF Reference Model is a standardized framework for organizing and managing the documents contents that comprise a TMF in clinical trials. This model serves as an industry best practice and provides a comprehensive structure to ensure that TMF documentation is complete, well-organized, and compliant with regulatory expectations.
### Rationale
The TMF Reference Model defines 249 document/artifact types. Although it is possible to use a Large Language Model (LLM) to summarize the documents and attempt to classify them, such approach is likely to incur large bills since one either has to have powerful and expensive hardware (GPU) on-premise or pay commercial LLM service providers, such as OpenAI, per token (word) for Application Programming Interface (API) access. The number of documents produced from a clinical trial can range from hundreds to thousands. For large, multi-center international clinical trials, the total document volume can reach hundreds of thousands of pages or tens of million tokens. If it’s successful, this light-weight multi-class document classifier running on a consumer-grade hardware, using only short document titles as input, should provide efficiency and financial benefits even if it can only accurately classify a portion, say 20 or 30% of the TMFs produced in a clinical trial.
### Research Question
This research tries to explore the effectiveness of using short document titles to classify documents where the *purposes* of the documents are well defined in a regulated domain, while the titles of the documents are not mandated and may vary according to an organization's standard operating procedures.
### Data Sources
The data sources consist of the public CDISC TMF Reference Model, ChatGPT/GPT-4 augmented training data, and proprietary dataset containing labeled artifact titles manually extracted from scanned and Optical Character Recognized PDF files. Due to limited field data at present time, OpenAI’s GPT-4 is prompted to produce synthetic document titles as labeled data for model training purposes.
### Methodology
**Pre-training:** The research will start with the TMF Reference Model where the recommended artifact, and sub-artifact titles and the purposes or the contents of the documents are described. The model will be pre-trained on this public dataset first. Synthetic data produced by OpenAI’s GPT-4 is also incorporated into the training data. GPT-4 is prompted to produce five common titles for a given artifact (class) that fits the purpose of the artifacts within the context of TMF Reference Model and Good Clinical Practices.  This approach also alleviate the class imbalance issue where the number of sub-artifacts for an artifact ranges from 1 to 14. Finally, the pre-trained models are tested with unseen field data.
### Results
* The initial results show 100% accuracy on tokenized and stemmed sub-artifact titles defined in the public TMF Reference Model with a Random Forest ensemble model.
* A pre-trained model augmented with GPT-4 synthetic training data achieves 99.8% accuracy on the TMF training data augmented with the synthetic training data.
* When tested against 93 unseen field data, not enough to cover all 249 classes, the TMF-trained model scores 40.9% accuracy while the GPT-enhanced model scores 41.9% accuracy. Both better than what I was aiming for. Please note that, only the classes that have only one associated sub-artifact are boosted with 5 additional synthetic sub-artifact titles. The effect of more extensive boosting with synthetic data yet has to be studied. However, the initial results should be considered as a test of the field data’s conformance to the recommendations of the TMF Reference Model, rather than the performance of the machine learning model.
* Finally, new models are built with the base training data (from the Reference Model and GPT-4) and incrementally increasing set of filed data for training the model. Each sampling rate below indicates the portion of the field data that is added to the base training data for training new models.

The scores are testing scores against unseen data.
<pre>
Sampling/Labeling rate = 0.0, Accuracy score = 0.42
Sampling/Labeling rate = 0.1, Accuracy score = 0.54
Sampling/Labeling rate = 0.2, Accuracy score = 0.61
Sampling/Labeling rate = 0.3, Accuracy score = 0.64
Sampling/Labeling rate = 0.4, Accuracy score = 0.68
Sampling/Labeling rate = 0.5, Accuracy score = 0.68
Sampling/Labeling rate = 0.6, Accuracy score = 0.76
Sampling/Labeling rate = 0.7, Accuracy score = 0.68
Sampling/Labeling rate = 0.8, Accuracy score = 0.63
</pre>
This exercise simulates the process and the effect of sampling and labeling new data. For this particular field dataset, sampling and labeling 20% to 30% of new data yields good improvement in accuracy while keeping the cost of labeling data down. Statistical sampling formula states 26 samples are needed for a population of 93 to achieve a 80% confidence interval with 10% margin of error – roughly equivalent to the 0.3 sampling rate in this case.

### Next steps
-	Additional field data for testing and model building
-	Model selection and hyperparameter tuning
-	Evaluate the effectiveness of additional synthetic training data.
### Outline of project

- Jupyter Notebook 1 [Building the Pre-trained Models](TMF%20Classifier.ipynb) (start here)
- Jupyter Notebook 2 [Testing against Unseen Field Data and Continueous Improvement](TMF%20Classifier%20Field%20Test.ipynb)

#### Contact and for Further Information
Email: chihming168.chen@gmail.com
