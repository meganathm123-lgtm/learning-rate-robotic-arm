import numpy as np
import pandas as pd


# Make the generated dataset reproducible
np.random.seed(42)

# Number of robotic-arm observations
num_samples = 1000

# Length of the two robotic-arm links
L1 = 1.0
L2 = 0.8

# Generate random joint angles between 0 and 180 degrees
joint_angle_1 = np.random.uniform(0, 180, num_samples)
joint_angle_2 = np.random.uniform(0, 180, num_samples)

# Convert degrees to radians
theta_1 = np.radians(joint_angle_1)
theta_2 = np.radians(joint_angle_2)

# Calculate the end-effector position
end_effector_x = (
    L1 * np.cos(theta_1)
    + L2 * np.cos(theta_1 + theta_2)
)

end_effector_y = (
    L1 * np.sin(theta_1)
    + L2 * np.sin(theta_1 + theta_2)
)

# Add a small amount of measurement noise
end_effector_x += np.random.normal(0, 0.01, num_samples)
end_effector_y += np.random.normal(0, 0.01, num_samples)

# Create a DataFrame
robotic_arm_data = pd.DataFrame({
    "Joint_Angle_1": joint_angle_1,
    "Joint_Angle_2": joint_angle_2,
    "End_Effector_X": end_effector_x,
    "End_Effector_Y": end_effector_y
})

# Save the dataset
output_path = "dataset/robotic_arm_data.csv"
robotic_arm_data.to_csv(output_path, index=False)

print("Dataset created successfully!")
print(f"File: {output_path}")
print(f"Number of samples: {len(robotic_arm_data)}")

print("\nFirst 5 rows:")
print(robotic_arm_data.head())