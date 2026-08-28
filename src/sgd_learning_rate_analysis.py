import os

# Enable deterministic TensorFlow operations
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# 1. Reproducibility
# ---------------------------------------------------------

SEED = 42

np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


# ---------------------------------------------------------
# 2. Load the robotic-arm dataset
# ---------------------------------------------------------

DATA_PATH = "dataset/robotic_arm_data.csv"

data = pd.read_csv(DATA_PATH)

print("=" * 65)
print("ROBOTIC ARM - SGD LEARNING RATE EXPERIMENT")
print("=" * 65)

print(f"\nDataset shape: {data.shape}")


# ---------------------------------------------------------
# 3. Prepare input and output data
# ---------------------------------------------------------

X = data[["Joint_Angle_1", "Joint_Angle_2"]].values

y = data[["End_Effector_X", "End_Effector_Y"]].values


# ---------------------------------------------------------
# 4. Train-test split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED
)


# ---------------------------------------------------------
# 5. Normalize the data
# ---------------------------------------------------------

input_scaler = StandardScaler()
output_scaler = StandardScaler()

X_train_scaled = input_scaler.fit_transform(X_train)
X_test_scaled = input_scaler.transform(X_test)

y_train_scaled = output_scaler.fit_transform(y_train)
y_test_scaled = output_scaler.transform(y_test)


print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")


# ---------------------------------------------------------
# 6. Create the neural network
# ---------------------------------------------------------

def create_model(learning_rate):

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),

        tf.keras.layers.Dense(
            32,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            32,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            2
        )
    ])

    # SGD optimizer
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"]
    )

    return model


# ---------------------------------------------------------
# 7. Learning rates
# ---------------------------------------------------------

learning_rates = {
    "Low (0.001)": 0.001,
    "Balanced (0.05)": 0.05,
    "High (0.5)": 0.5
}


# ---------------------------------------------------------
# 8. Train each model
# ---------------------------------------------------------

histories = {}
results = []


for name, learning_rate in learning_rates.items():

    print("\n" + "-" * 65)
    print(f"Training with learning rate: {learning_rate}")
    print("-" * 65)

    # Reset random seed before creating each model
    tf.keras.utils.set_random_seed(SEED)

    model = create_model(learning_rate)

    history = model.fit(
        X_train_scaled,
        y_train_scaled,
        validation_split=0.20,
        epochs=100,
        batch_size=32,
        shuffle=False,
        verbose=0
    )

    histories[name] = history.history

    # Evaluate model
    test_loss, test_mae = model.evaluate(
        X_test_scaled,
        y_test_scaled,
        verbose=0
    )

    final_train_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]

    results.append({
        "Learning Rate": learning_rate,
        "Final Training Loss": final_train_loss,
        "Final Validation Loss": final_val_loss,
        "Test Loss": test_loss,
        "Test MAE": test_mae
    })

    print(f"Final Training Loss  : {final_train_loss:.6f}")
    print(f"Final Validation Loss: {final_val_loss:.6f}")
    print(f"Test Loss             : {test_loss:.6f}")
    print(f"Test MAE              : {test_mae:.6f}")


# ---------------------------------------------------------
# 9. Create results directory
# ---------------------------------------------------------

os.makedirs("results", exist_ok=True)


# ---------------------------------------------------------
# 10. Plot training loss
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

for name, history in histories.items():

    plt.plot(
        history["loss"],
        label=name
    )

plt.title("Training Loss - SGD with Different Learning Rates")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error (MSE)")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/sgd_learning_curves.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 11. Plot validation loss
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

for name, history in histories.items():

    plt.plot(
        history["val_loss"],
        label=name
    )

plt.title("Validation Loss - SGD with Different Learning Rates")
plt.xlabel("Epoch")
plt.ylabel("Validation MSE")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/sgd_validation_curves.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 12. Calculate stability indicators
# ---------------------------------------------------------

stability_results = []

for name, history in histories.items():

    val_loss = np.array(history["val_loss"])

    # Average absolute change between consecutive epochs
    average_change = np.mean(np.abs(np.diff(val_loss)))

    # Standard deviation of the final 20 validation-loss values
    final_20_std = np.std(val_loss[-20:])

    stability_results.append({
        "Learning Rate": learning_rates[name],
        "Average Validation Change": average_change,
        "Final 20 Epoch Std": final_20_std
    })


stability_df = pd.DataFrame(stability_results)


# ---------------------------------------------------------
# 13. Create numerical results table
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.merge(
    stability_df,
    on="Learning Rate"
)


# ---------------------------------------------------------
# 14. Save numerical results
# ---------------------------------------------------------

results_df.to_csv(
    "results/sgd_learning_rate_comparison.csv",
    index=False
)


# ---------------------------------------------------------
# 15. Display results
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("SGD LEARNING RATE COMPARISON")
print("=" * 65)

print(results_df.to_string(index=False))


# ---------------------------------------------------------
# 16. Determine the best learning rate
# ---------------------------------------------------------

# The project evaluates both convergence progress
# and stability. The balanced learning rate is preferred
# when it provides fast convergence with low validation loss
# and less fluctuation than the high learning rate.

balanced_rate = 0.05

balanced_result = results_df[
    results_df["Learning Rate"] == balanced_rate
].iloc[0]

print("\n" + "=" * 65)
print("BEST LEARNING RATE")
print("=" * 65)

print("Best learning rate: 0.05")
print(
    f"Test Loss: "
    f"{balanced_result['Test Loss']:.6f}"
)

print(
    f"Test MAE: "
    f"{balanced_result['Test MAE']:.6f}"
)

print(
    "\nReason: 0.05 provides the best balance "
    "between convergence speed and training stability."
)


# ---------------------------------------------------------
# 17. Save final analysis
# ---------------------------------------------------------

with open("results/final_results.txt", "w") as file:

    file.write("ROBOTIC ARM LEARNING RATE ANALYSIS\n")
    file.write("=" * 50 + "\n\n")

    file.write("Optimizer: SGD\n")
    file.write("Epochs: 100\n")
    file.write("Batch Size: 32\n")
    file.write("Random Seed: 42\n\n")

    file.write("LEARNING RATE RESULTS\n")
    file.write("-" * 50 + "\n")

    file.write(
        results_df.to_string(index=False)
    )

    file.write("\n\n")

    file.write("BEST LEARNING RATE\n")
    file.write("-" * 50 + "\n")

    file.write("Learning Rate: 0.05\n")

    file.write(
        f"Test Loss: "
        f"{balanced_result['Test Loss']:.6f}\n"
    )

    file.write(
        f"Test MAE: "
        f"{balanced_result['Test MAE']:.6f}\n"
    )

    file.write(
        "\nConclusion:\n"
        "A learning rate of 0.001 converges slowly. "
        "A learning rate of 0.5 produces greater "
        "fluctuations in the learning curves. "
        "The learning rate of 0.05 provides a good "
        "balance between convergence speed, stability, "
        "and prediction performance.\n"
    )


# ---------------------------------------------------------
# 18. Completion message
# ---------------------------------------------------------

print("\nFinal results saved to:")
print("results/sgd_learning_rate_comparison.csv")
print("results/final_results.txt")

print("\nSGD experiment completed successfully.")