# CMPSC441_FinalProject
## X-Ray Brain Tumor Classification
### Multi-model Pipeline Process

Dataset used:
- Training + initial testing (test): https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset/data
- True testing (test2): https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
  

Features:
- A dual-CNN model system that allows for users to test 3 models (CNN1, CNN2, Ensemble of CNN1 and CNN2)
- Provides accuracy, recall, precision, and F1 score classification metrics to test models
- Manual testing for indivudal image predications and confidence scores
- UI, creating ease-of-use for model testing & indivudal model predictions

Data folder arrangement:
- Data --> "train", "test", and "test2" folders
- train --> "Brain_Tumor" and "Healthy" folders
- test --> "Brain_Tumor" and "Healthy" folders
- test2 --> "Brain Tumor" and "Healthy" folders (image data from second dataset combined into binary classification) 
