## A Light-weight TMF Document Classifier
Author: Chihming Chen

### Executive summary
This experiment tries to assess the feasibility of using artifact tiltes to classify documents per CDISC TMF Reference Model.

#### Trial Master Files (TMFs)
Trial Master Files (TMFs) are a crucial aspect of clinical trials, serving as the official repository for all the documents that allow for the conduct of the trial, as well as the recording of the intentions, processes, and outcomes. These files are essential for ensuring compliance with regulatory requirements and good clinical practice (GCP) standards. The main purpose of a TMF is to provide a comprehensive documentation system that enables the conduct of a clinical trial to be fully reconstructed and audited.

#### TMF Reference Model
The CDISC (Clinical Data Interchange Standards Consortium) TMF Reference Model is a standardized framework for organizing and managing the documents and content that comprise a TMF in clinical trials. This model serves as an industry best practice and provides a comprehensive structure to ensure that TMF documentation is complete, well-organized, and compliant with regulatory expectations.

#### Rationale
The TMF Reference Model defines 249 artifact types. It is possible to use a Large Language Model (LLM) to summarize the documents and classify them.  Such approach is likely to incur large bills since one either has to have powerful and expensive hardware(GPU) on-premise or pay commercial LLM service providers, such as OpenAI, per token for API access. The number of documents produced from a clinical trial can range from hundreds to thousands. Especially for large, multi-center international trials, the total document volumn can reach hundreds of thousand of pages or tens of million tokens. A light-weight multi-class classifier running on a consumer-grade hardware, using only short document titles as input, should provide efficiency and financial benefits even if it can only accurately classify a portion of the TMF document set produced in a clinical trial.

#### Research Question
This researh tries to explor the effectiveness of using short document titles to classify document artifacts where the *purposes* of the documents are well defined in a regulated domain, such as clinical trials for the life sciences industry.

#### Data Sources
The data sources consist of the public CDISC TMF Reference Model and proprietory dataset constaing labeled artifact titles extracted from scanned adn OCR'ed PDF files where some defects in OCR are expectd.

#### Methodology
The research will start with the TMF Reference Model where the recommended artifact, and sub-artifact titles are defined. The model will be pre-trained on this public dataset first, then use the pre-trained model to classify field data.  The unseen field data will be divided into several holdout sets to simulate the continously augmented (and hopefully improving) model.

#### Results
The initial results show 99.5% accuracy on tokenized and stemmed sub-artifact tiltes defined in the public TMF Refernce Model. However, the classes are imbalanced, raning from 1 to 9 samples per class. This needs to be addressed in later phases. The performance on the field data has not been tested as the unseen data have yet to be labeled.

#### Next steps
TBD

#### Outline of project

- [Link to notebook 1]()
- [Link to notebook 2]()
- [Link to notebook 3]()


#### Contact and for Further Information
Email: chihming168.chen@gmail.com
