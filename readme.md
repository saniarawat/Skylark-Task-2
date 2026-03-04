Project Overview\
This project implements a high-precision computer vision pipeline to
detect and localize ground markers with sub-pixel accuracy, even when
markers are partially occluded or located at the edges of the drone
frame.\
\
Key Innovation: 8-Sector Radial Partitioning\
To distinguish between full markers (Squares/Crosses) and occluded ones,
I developed a 7-vs-5 Sector Occupancy Logic.\
The area around the marker is divided into 8 angular octants (45 degrees
each).\
Square Markers: Occupy all 8 sectors (full 360-degree radial mass).\
Synthetic L-Shapes: Engineered to occupy only 5 sectors (225 degrees).\
Model Architecture\
The system utilizes a Multi-Task ResNet-18 architecture. By sharing a
pre-trained backbone, the model simultaneously optimizes for two
distinct objectives:\
Classification (Cross-Entropy Loss): Identifying the marker geometry
(Square vs. L-Shape) based on the sector logic.\
Regression (MSE Loss): Pinpointing the sub-pixel \[x, y\] coordinates of
the centroid vertex.\
Dual-Loss Weighting: To prioritize localization precision, the
regression loss is weighted by a factor of 10, ensuring the predicted
vertex remains the primary focus of the gradient descent.\
Getting Started\
Prerequisites\
Python 3.8+\
PyTorch and Torchvision\
PIL, Pandas, Scikit-learn, Tqdm\
\
Installation\
Bash\
git clone https://github.com/your-username/marker-detection-task.git\
cd marker-detection-task\
pip install -r requirements.txt\
\
Data Pipeline\
The model employs Reflective Padding to maintain texture integrity for
markers located at image boundaries. This ensures the radial mass
calculation remains accurate regardless of the marker\'s position in the
frame.\
\
Performance Metrics\
Target Pixel Error: Less than 5.0 pixels (at 512 x 512 resolution).\
Class Balancing: Implemented via WeightedRandomSampler to address the 87
percent majority-class bias observed in the initial dataset.\
\
Repository Structure\
model.py: Multi-task ResNet-18 implementation.\
train.py: Dual-loss training and validation loop.\
samples/: Visualizations of predicted versus ground-truth vertices.\
\
Conclusion\
This approach demonstrates that geometric constraints, such as the
7-vs-5 logic, can be successfully integrated into deep learning
frameworks to solve complex localization problems in transit
infrastructure projects.
