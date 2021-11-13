# Face Recognition Project for Artificial Vision
Training and evaluating a face recognition system able to recognize 10 reference identities with a reject option.
## Preprocessing
- Face alignment and crop
- Histogram equalization
- Bilateral filter
## Feature Extraction
- Landmarks' calculation
- Calculation of the distances between each couple of landarmks
- Applying PCA to the distances to obtain the features
## Classification
Support Vector Machine.

### Notes
It would have been possible to try out several options for preprocessing, as well as a greater focus on classification.
