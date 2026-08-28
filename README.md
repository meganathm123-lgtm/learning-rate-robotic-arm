# Robotic Arm Learning Rate Analysis

## 1. Project Overview

This project analyzes the effect of different learning rates on the training of a neural network for a robotic-arm model.

The neural network learns the relationship between two robotic-arm joint angles and the corresponding end-effector position.

Multiple learning rates are tested using the Stochastic Gradient Descent (SGD) optimizer. The resulting training and validation learning curves are analyzed to compare convergence speed, stability, and prediction performance.

The main purpose of the project is to determine which learning rate provides the best balance between stable training and sufficient learning progress.

---

## 2. Problem Statement

A robotic-arm model is trained with several learning rates and shows oscillation at one setting and slow progress at another.

The task is to analyze the observed learning curves, compare the convergence behaviors, and infer which learning rate provides the best balance between stability and progress.

---

## 3. Objective

The objectives of this project are:

- To understand the role of learning rate in neural-network training.
- To train a neural network for robotic-arm end-effector position prediction.
- To compare different learning rates using the same model architecture.
- To analyze training and validation learning curves.
- To identify slow convergence caused by a small learning rate.
- To identify unstable or fluctuating behavior caused by a large learning rate.
- To compare the final test performance of different learning rates.
- To select the learning rate that provides the best balance between stability and training progress.

---

## 4. Concept Used

### Learning Rate

The learning rate is a hyperparameter that controls the size of the update made to the model parameters during training.

A very small learning rate can make training slow because the model takes small steps toward the minimum of the loss function.

A very large learning rate can cause unstable training because the model may take excessively large steps and move around the minimum instead of converging smoothly.

A suitable learning rate provides a balance between:

- Training speed
- Convergence
- Stability
- Final prediction performance

### Stochastic Gradient Descent

The main experiment uses the Stochastic Gradient Descent (SGD) optimizer.

During training, SGD updates the neural-network weights using the gradient of the loss function.

The simplified update rule is:

    New Weight = Old Weight - Learning Rate × Gradient

Therefore, the learning rate directly influences how much the model parameters change during each update.

---

## 5. Dataset

The project uses a simulated robotic-arm dataset containing 1000 observations.

The robotic arm contains two joints and two links.

### Input Features

- Joint_Angle_1
- Joint_Angle_2

### Target Outputs

- End_Effector_X
- End_Effector_Y

The end-effector coordinates are generated using forward-kinematics equations.

The dataset also includes a small amount of measurement noise to make the problem more realistic.

### Dataset Split

The dataset is divided into:

- Training samples: 800
- Testing samples: 200

During training, 20% of the training data is used for validation.

---

## 6. Methodology / Working Steps

The project follows these steps:

1. Generate the robotic-arm dataset.
2. Save the generated data as a CSV file.
3. Load the dataset using Pandas.
4. Separate input features and target outputs.
5. Divide the data into training and testing sets.
6. Normalize the input and output values.
7. Construct a feed-forward neural network.
8. Use the SGD optimizer.
9. Train the model using different learning rates.
10. Record training and validation losses.
11. Generate learning curves.
12. Evaluate each model using the test dataset.
13. Compare the numerical results.
14. Analyze the convergence behavior.
15. Select the most suitable learning rate.

### Experimental Learning Rates

The following learning rates are compared:

| Learning Rate | Category |
|---:|---|
| 0.001 | Low |
| 0.05 | Balanced |
| 0.5 | High |

All three experiments use the same dataset, neural-network architecture, optimizer, batch size, and number of epochs. Only the learning rate is changed.

This allows the effect of the learning rate to be studied fairly.

---

## 7. Model Architecture

The neural network used in the experiment contains:

```text
Input Layer
    2 neurons
       |
       v
Hidden Layer 1
    32 neurons
    ReLU activation
       |
       v
Hidden Layer 2
    32 neurons
    ReLU activation
       |
       v
Output Layer
    2 neurons
```

### Input

The model receives:

- Joint 1 angle
- Joint 2 angle

### Output

The model predicts:

- X coordinate of the end effector
- Y coordinate of the end effector

### Training Configuration

- Optimizer: SGD
- Loss function: Mean Squared Error (MSE)
- Metric: Mean Absolute Error (MAE)
- Epochs: 100
- Batch size: 32
- Validation split: 20%

---

## 8. Implementation

### Tools and Libraries

The project is implemented using Python and the following libraries:

- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

### Source Files

The main source files are:

```text
src/
├── generate_dataset.py
├── learning_rate_analysis.py
└── sgd_learning_rate_analysis.py
```

### Dataset Generator

`generate_dataset.py` creates the robotic-arm dataset using forward-kinematics equations and saves it as:

```text
dataset/robotic_arm_data.csv
```

### Learning Rate Analysis

`learning_rate_analysis.py` performs an initial learning-rate experiment using the Adam optimizer.

### Main SGD Experiment

`sgd_learning_rate_analysis.py` performs the main learning-rate comparison using SGD.

The main experiment compares:

```text
0.001
0.05
0.5
```

---

## 9. Experimental Results

The following results were obtained from the SGD experiment.

| Learning Rate | Final Training Loss | Final Validation Loss | Test Loss | Test MAE |
|---:|---:|---:|---:|---:|
| 0.001 | 0.178542 | 0.169160 | 0.216001 | 0.336934 |
| 0.05 | 0.003007 | 0.003335 | 0.003925 | 0.038864 |
| 0.5 | 0.005495 | 0.003959 | 0.004633 | 0.053636 |

### Learning Rate = 0.001

The low learning rate produces slow progress.

The loss decreases gradually over the training epochs, but the final error remains considerably higher than the other learning rates.

This indicates slow convergence.

### Learning Rate = 0.05

The learning rate of 0.05 produces fast and stable convergence.

The training loss decreases rapidly and reaches a low value.

It also produces the lowest test loss and lowest test MAE among the three tested learning rates.

### Learning Rate = 0.5

The high learning rate produces greater fluctuations in the learning curves.

Although the model learns quickly at the beginning, the training behavior is less stable than the balanced learning rate.

The final test performance is also slightly worse than the learning rate of 0.05.

---

## 10. Learning Curve Analysis

The learning curves provide a visual comparison of the three learning rates.

### Low Learning Rate

The 0.001 learning rate produces a gradual downward curve.

This represents slow convergence because the parameter updates are small.

### Balanced Learning Rate

The 0.05 learning rate produces a rapid decrease in loss followed by stable convergence.

This indicates that the model is able to learn efficiently without excessive fluctuations.

### High Learning Rate

The 0.5 learning rate shows greater fluctuations in the learning curves.

The larger parameter updates make the training process less stable.

Therefore, the learning curves demonstrate that an excessively large learning rate can reduce convergence stability.

---

## 11. Results Interpretation

The experimental results can be summarized as follows:

```text
Learning Rate
      |
      +--------------------+
      |                    |
     0.001               0.05
      |                    |
Slow progress       Fast + stable
      |                    |
      +--------------------+
               |
              0.5
               |
       Greater fluctuation
       / less stable
```

The comparison shows that:

- 0.001 is too small for efficient training.
- 0.05 provides fast and stable convergence.
- 0.5 produces more fluctuation and slightly poorer test performance.

The test loss values also support this observation.

The test loss for 0.05 is:

```text
0.003925
```

which is lower than:

```text
0.216001  → 0.001
0.004633  → 0.5
```

Therefore, 0.05 provides the best overall performance in this experiment.

---

## 12. Best Learning Rate

Based on the learning curves and numerical evaluation, the selected learning rate is:

# 0.05

The model achieved:

```text
Final Training Loss  : 0.003007
Final Validation Loss: 0.003335
Test Loss            : 0.003925
Test MAE             : 0.038864
```

The learning rate of 0.05 provides the best balance between:

- Training speed
- Convergence stability
- Low prediction error
- Generalization to test data

Therefore, it is selected as the most suitable learning rate for the robotic-arm model in this experiment.

---

## 13. Project Structure

```text
learning-rate-robotic-arm/
│
├── README.md
├── requirements.txt
│
├── dataset/
│   └── robotic_arm_data.csv
│
├── src/
│   ├── generate_dataset.py
│   ├── learning_rate_analysis.py
│   └── sgd_learning_rate_analysis.py
│
├── notebooks/
│
├── results/
│   ├── learning_curves.png
│   ├── validation_curves.png
│   ├── learning_rate_comparison.csv
│   ├── sgd_learning_curves.png
│   ├── sgd_validation_curves.png
│   ├── sgd_learning_rate_comparison.csv
│   └── final_results.txt
│
└── screenshots/
```

---

## 14. How to Run the Project

### Step 1: Clone the repository

After the GitHub repository is created:

```bash
git clone <repository-url>
```

Enter the project folder:

```bash
cd learning-rate-robotic-arm
```

### Step 2: Install the required packages

```bash
pip install -r requirements.txt
```

### Step 3: Generate the dataset

```bash
python src/generate_dataset.py
```

This creates:

```text
dataset/robotic_arm_data.csv
```

### Step 4: Run the initial Adam experiment

```bash
python src/learning_rate_analysis.py
```

### Step 5: Run the main SGD experiment

```bash
python src/sgd_learning_rate_analysis.py
```

The experiment generates the learning curves and comparison files inside the `results` directory.

---

## 15. Output Files

The project generates the following important outputs:

### Learning Curves

```text
results/learning_curves.png
```

Shows the training loss for the initial Adam experiment.

### Validation Curves

```text
results/validation_curves.png
```

Shows the validation loss for the initial Adam experiment.

### SGD Learning Curves

```text
results/sgd_learning_curves.png
```

Shows the training loss comparison for the three SGD learning rates.

### SGD Validation Curves

```text
results/sgd_validation_curves.png
```

Shows the validation loss comparison.

### Numerical Comparison

```text
results/sgd_learning_rate_comparison.csv
```

Contains the final training loss, validation loss, test loss, and test MAE for each learning rate.

### Final Results

```text
results/final_results.txt
```

Contains the summarized experimental findings and selected learning rate.

---

## 16. Conclusion

This project demonstrates the effect of learning rate on neural-network training using a robotic-arm prediction problem.

The experiment shows that a very small learning rate such as 0.001 results in slow progress because the model takes small parameter-update steps.

A learning rate of 0.05 provides fast and stable convergence and achieves the best test performance.

A high learning rate of 0.5 produces greater fluctuations and slightly poorer test performance, demonstrating that excessively large updates can reduce training stability.

Based on the observed learning curves and numerical results, the learning rate of 0.05 provides the best balance between stability and progress.

Therefore:

**Selected Learning Rate = 0.05**

---

## 17. References

The project uses the following documentation resources:

1. TensorFlow Documentation
2. Keras Documentation
3. NumPy Documentation
4. Pandas Documentation
5. Scikit-learn Documentation
6. Matplotlib Documentation
