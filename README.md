Dissertation Python Script: Machine Learning Model Analysis Tool for Phishing Detection
This repository contains a Python-based application designed to analyze datasets of legitimate and phishing websites to determine the most suitable machine learning model for identifying phishing sites. The tool resets its learning state with each execution, ensuring consistent and unbiased evaluations.

The dataset used in this study is sourced from the UC Irvine Machine Learning Repository (Mohammad & McCluskey, 2012). It comprises 11,055 instances, each containing 30 features extracted from legitimate and phishing websites.

Features
1. Phishing Detection with Machine Learning
Evaluate datasets using two machine learning algorithms:
Decision Tree
Random Forest
The program calculates key performance metrics for each algorithm:
Accuracy
Precision
Recall
F1 Score
Helps determine the optimal model for detecting phishing websites.
2. Dataset Handling
Supports CSV and ARFF file formats.
Automatically splits the dataset into training and testing subsets:
Training size: First 2,000 instances
Testing size: Remaining instances.
3. Interactive Menu
Train AI: Load a phishing dataset, select a machine learning algorithm, and view performance metrics.
Analyze Website: Placeholder for future implementation (e.g., real-time phishing site analysis).
Quit: Exit the program.
Prerequisites
Ensure Python 3.x and the required libraries are installed. Install dependencies using:

bash
Copy code
pip install -r requirements.txt
If a requirements.txt file is not provided, manually install the following libraries:

pandas
scikit-learn
scipy
Dataset
The dataset for this project is not included in the repository due to its size. Download the dataset from the UC Irvine Machine Learning Repository, which includes features derived from legitimate and phishing websites. Save it in the project directory as dataset.csv or update the script to use the correct path.

Usage
Clone the Repository
bash
Copy code
git clone https://github.com/your-username/repo-name.git
cd repo-name
Run the Program
bash
Copy code
python main.py
Menu Options
Press A to load a phishing dataset:

Supported formats: .csv, .arff.
Follow on-screen prompts to evaluate the dataset with either Decision Tree or Random Forest.
Press B to analyze a website (not yet functional).

Press Q to quit the program.

File Overview
1. main.py
The primary script containing the main menu and user interaction logic.

2. tester.py
Handles the core functionality:

Dataset loading and preprocessing.
Splitting data into training and testing subsets.
Model training and evaluation using selected algorithms.
Example Output
Upon running the program and selecting an algorithm, you will receive metrics like the following:

plaintext
Copy code
Random Forest:
Accuracy: 92.34%
Precision: 89.45%
Recall: 91.67%
F1 Score: 90.54%
Limitations and Future Development
Webpage Analysis:

Currently a placeholder. Future implementation could involve real-time feature extraction from websites to assess phishing risks.
Extended Algorithm Support:

Expand the tool to include additional machine learning algorithms like SVM, k-NN, or neural networks.
Visualization:

Incorporate graphical representations of model performance metrics for easier analysis.
Disclaimer
This project is for educational purposes and focuses on phishing website detection. Ensure compliance with dataset licensing terms and use the tool responsibly.

