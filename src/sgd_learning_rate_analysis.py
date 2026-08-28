import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# 1. Reproducibility
# ---------------------------------------------------------

np.random.seed(42)
tf.random.set_seed(42)


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
    random_state=42
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

    # Reset the random seed before creating each model
    tf.random.set_seed(42)

    model = create_model(learning_rate)

    history = model.fit(
        X_train_scaled,
        y_train_scaled,
        validation_split=0.20,
        epochs=100,
        batch_size=32,
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
# 12. Save numerical results
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/sgd_learning_rate_comparison.csv",
    index=False
)


# ---------------------------------------------------------
# 13. Display results
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("SGD LEARNING RATE COMPARISON")
print("=" * 65)

print(results_df.to_string(index=False))


# ---------------------------------------------------------
# 14. Find the best learning rate
# ---------------------------------------------------------

best_result = results_df.loc[
    results_df["Test Loss"].idxmin()
]

print("\n" + "=" * 65)
print("BEST LEARNING RATE")
print("=" * 65)

print(
    f"Best learning rate: "
    f"{best_result['Learning Rate']}"
)

print(
    f"Test Loss: "
    f"{best_result['Test Loss']:.6f}"
)

print(
    f"Test MAE: "
    f"{best_result['Test MAE']:.6f}"
)

print("\nSGD experiment completed successfully.")