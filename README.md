# DecisionTree 
Decision Tree Classifier |Jupyter Notebook | Supervised Learning | Machine Learning | Python | Scikit library

## Overview
This project explores the Decision Tree Classifier, implementing it both from scratch (using Python and NumPy) and via Scikit-learn for comparison. The goal is to demonstrate how decision trees work under the hood while benchmarking performance against a well-optimized library.

### Key Features

✔ **Hands-on Implementation**: Builds a decision tree from scratch using the Gini Index as the splitting criterion.

✔ **Scikit-learn Comparison**: Evaluates the custom implementation against Scikit-learn’s DecisionTreeClassifier for validation.

✔ **Diverse Datasets**: Tested on the Iris, Wine, and Car Evaluation datasets to assess performance across different problem types.

✔ **Step-by-Step Walkthrough**: The notebook breaks down training, prediction, and evaluation, making it easy to follow along.

By the end, readers will understand:

- How decision trees split nodes and make predictions.

- Why the Gini Index is used for measuring impurity.

- The trade-offs between custom vs. library-based implementations.

This serves as a practical guide for **beginners learning ML fundamentals** and **intermediate practitioners** refining their understanding of tree-based models.

## Table of Contents
- [File Structure 📂](#file-structure-📂)
- [Requirements 📦](#requirements-📦)
- [Installation Guide 🛠](#installation-guide-🛠)
- [Dataset Information 📊](#dataset-information-📊)
- [Decision Tree Algorithm 🧠](#decision-tree-algorithm-🧠)
- [Key Findings 📈](#key-findings-📈)
- [Contributing 🚀](#contributing-🚀)
- [Contact 📬](#contact-📬)

## File Structure 📂
The repo contains the following file structure:  

```bash
📦 Decision Tree repo
│-- 📜 DecisionTree.ipynb       # Jupyter Notebook with implementation
│-- 📜 requirements.txt         # List of dependencies
│-- 📜 iris.tmls                # Iris Flower dataset
│-- 📜 wine.tmls                # Wine dataset
|-- 📜 car.tmls                 # car dataset
│-- 📜 README.md                # Project documentation

```

## Requirements 📦
- **Python Version**: 3.10 or higher
- **External Dependencies**: Managed through `requirements.txt`
- **Jupter Notebook** for the web framework
- **Numpy** 
- **Panda** 

## Installation Guide 🛠

Follow the steps below to set up and run the project:

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/adexoxo13/Naive-Bayes.git
cd Naive-Bayes
```

### 2️⃣ Create a Virtual Environment (Optional but Recommended)

```bash
conda create --name <my-env>
# When conda asks you to proceed, type y:
proceed ([y]/n)?  

#Verify that the new environment was installed correctly:
conda env list

#Activate the new environment:
conda activate myenv
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Launch Jupyter Notebook
```bash
jupyter notebook
```
Open `DecisionTree.ipynb` in Jupyter and run the cells to see the model in action.

---

## Dataset Information 📊 
- The **Iris Dataset 🌸** consists of 150 samples, with the following attributes:

| Feature        | Description                    |
|----------------|--------------------------------|
| Sepal Length   | Length of the sepal (cm)       |
| Sepal Width    | Width of the sepal (cm)        |
| Petal Length   | Length of the petal (cm)       |
| Petal Width    | Width of the petal (cm)        |
| Species        | Type of Iris Flower (Target)   |
|----------------|--------------------------------|
---


- The **Wine Dataset** consists of 178 samples, with the following attributes:

| Feature                       | Description                                            |
|-------------------------------|--------------------------------------------------------|
| Alcohol                       | Acohol in the wine (percentage)                        |
| Malic Acid                    | Measure of acidity in the wine                         |
| Ash                           | Measure of ash content in the wine                     |
| Alcalinity of ash             | Measure of alkalinity in the wine                      |
| Magnesium                     | Measure of magnesium content in the wine               | 
| Total phenols                 | Measure of total phenolic compounds in the wine        |
| Flavanoids                    | Measure of flavonoid compounds in the wine             |
| Nonflavanoid Phenols          | Measure of nonflavanoid phenolic compounds in the wine |
| Proanthocyanins               | Measure of proanthocyanin compounds in the wine        | 
| Color intensity               | Measure of color depth and richness in the wine        |
| Hue                           | Measure of color tone and variation in the wine        |
| OD280/OD315 of diluted wines  | Measure of absorbance ratios in wines                  |
| Proline                       | Measure of amino acid content in the wine              |
| class                         | Type of wine ( The target)                             |
|-------------------------------|--------------------------------------------------------|

---

- The **Car Dataset** consists of 1728 samples, with the following attributes

| Feature                       | Description                                                 |
|-------------------------------|-------------------------------------------------------------|
| Buying                        | Buying price (e.g., low, medium, high,...)                  |
| Maint (Maintenance)           | Price of the maintenance (e.g., low, medium, high)          |
| Doors                         | Number of doors in the car (e.g., two, four)                |
| Persons                       | Number of persons that can be seated in the car             |
| Lug Boot                      | The size of luggage boot (e.g., small, medium,...)          |
| Safety                        | Estimated safety of the car (e.g., low, medium, high...)    |
| Class                         | Evaulation level (unacceptable, acceptable, good, very good)|
|-------------------------------|-------------------------------------------------------------|

## Decision Tree Algorithm 🧠
The jupyter notebook contains two implementations of a Decision Tree classifier:

1. From Scratch

    - Built using a recursive algorithm

    - Splits nodes based on Gini impurity for optimal decision boundaries

2. Scikit-learn Version

    - Uses the DecisionTreeClassifier from sklearn

    - Also employs Gini impurity for splits, ensuring consistency with the manual implementation

---
## Key Findings 📈 

### The Accuracy of a Decision Tree classifier for Iris Dataset 🌸.
```python
Run 0 : 1.0
Run 1 : 0.9333333333333333
Run 2 : 0.9333333333333333
Run 3 : 0.8333333333333334
Run 4 : 1.0
```

### The Accuracy of a Decision Tree Classifier using scikit library for Iris Dataset 🌸.
Consistently achieved *100%* accuracy across all runs:
```python
Accuracy: 1.0
Accuracy: 1.0
Accuracy: 1.0
Accuracy: 1.0
Accuracy: 1.0
```

### The Decision Tree looks like the following:
```python
X[2] <= 3.727 ?  # Gini impurity: 0.271
    left: X[2] <= 1.726 ?  # Gini: 0.162
        left: Iris-setosa
        right: X[2] <= 3.0 ?  # Gini: 0.375
            left: Iris-setosa
            right: Iris-versicolor
    right: X[3] <= 1.705 ?  # Gini: 0.355
        left: X[3] <= 1.4 ?  # Gini: 0.021
            left: Iris-versicolor
            right: X[2] <= 4.738 ?  # Gini: 0.067
                left: Iris-versicolor
                right: Iris-versicolor
        right: X[1] <= 3.012 ?  # Gini: 0.002
            left: Iris-virginica
            right: X[3] <= 2.193 ?  # Gini: 0.014
                left: Iris-virginica
                right: Iris-virginica
```


### The Accuracy of a Decision Tree classifier for Wine Dataset 🍷.
```python
Run 0 : 0.8055555555555556
Run 1 : 0.8888888888888888
Run 2 : 0.9444444444444444
Run 3 : 0.8333333333333334
Run 4 : 0.8055555555555556
```

### The Accuracy of a Decision Tree Classifier from scikit library for Wine Dataset 🍷

Consistently *~94–96%* accuracy:
```python
Accuracy: 0.9629629629629629
Accuracy: 0.9444444444444444
Accuracy: 0.9629629629629629
Accuracy: 0.9444444444444444
Accuracy: 0.9444444444444444
```
#### The Decision Tree looks like the following:

```python
X[12] <= 762.0 ?  # Gini: 0.265
    left: X[9] <= 4.850 ?  # Gini: 0.374
        left: X[1] <= 1.939 ?  # Gini: 0.0017
            left: Class 2
            right: X[1] <= 3.180 ?  # Gini: 0.013
                left: Class 2
                right: Class 2
        right: X[6] <= 1.099 ?  # Gini: 0.046
            left: Class 3
            right: X[10] <= 0.759 ?  # Gini: 0.258
                left: Class 3
                right: Class 2
    right: X[1] <= 2.054 ?  # Gini: 0.045
        left: X[0] <= 13.664 ?  # Gini: 0.0135
            left: X[0] <= 13.152 ?  # Gini: 0.087
                left: Class 1
                right: Class 1
            right: Class 1
        right: X[5] <= 2.302 ?  # Gini: 0.473
            left: Class 3
            right: Class 1

```
---

#### The Accuracy of a Decision Tree classifier for car Dataset 🚙 with nominal feature value.
```python
Run 0 : 0.7745664739884393
Run 1 : 0.7514450867052023
Run 2 : 0.8121387283236994
Run 3 : 0.7890173410404624
Run 4 : 0.7543352601156069
```
#### The Accuracy of a Decision Tree Classifier from scikit library for Car Dataset 🚙

Consistently *~95–96%* accuracy:
```python
Accuracy: 0.9595375722543352
Accuracy: 0.9556840077071291
Accuracy: 0.9556840077071291
Accuracy: 0.9614643545279383
Accuracy: 0.9556840077071291
```
#### The Decision Tree looks like the following:
```python
X[3] in [0, 1, 2] ?  # Gini: 0.072
    left: "unacc"
    right: X[5] in [0, 1, 2] ?  # Gini: 0.074
        left: X[0] in [0, 1, 2, 3] ?  # Gini: 0.049
            left: X[0] in [0, 1, 2] ?  # Gini: 0.064
                left: "acc"
                right: "acc"
            right: X[1] in [0, 1, 2, 3] ?  # Gini: 0.150
                left: "acc"
                right: "unacc"
        right: X[5] in [1, 2] ?  # Gini: 0.151
            left: "unacc"
            right: X[4] in [0, 1, 2] ?  # Gini: 0.057
                left: "acc"
                right: "unacc"

```

---

## Contributing 🚀

Contributions are welcome! Feel free to fork the repository and submit a pull request. 

---

### Dataset Citation
The datasets used in this project are from the [UCI Machine Learning Repository](https://archive.ics.uci.edu):  
**Dua, D. and Graff, C.** (2019). *UCI Machine Learning Repository*. Irvine, CA: University of California, School of Information and Computer Science.  

---
## Contact 📬

Feel free to reach out or connect with me:

- 📧 **Email:** [adenabrehama@gmail.com](mailto:adenabrehama@gmail.com)
- 💼 **LinkedIn:** [linkedin.com/in/aden](https://www.linkedin.com/in/aden-alemayehu-1629aa255)
- 🎨 **CodePen:** [codepen.io/adexoxo](https://codepen.io/adexoxo)

📌 *Star this repository if you found it useful!* ⭐

