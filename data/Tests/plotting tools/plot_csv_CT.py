import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "generator_losses_nonlinearity.csv"
df = pd.read_csv(file_path)

# Convert "N/A" and "---" to NaN for numerical processing
df.replace(["N/A", "---"], pd.NA, inplace=True)

# Ensure Seed column is numeric
df["Seed"] = pd.to_numeric(df["Seed"], errors="coerce")

# Convert columns to appropriate types (ensure all loss columns are numeric)
df["Epoch"] = pd.to_numeric(df["Epoch"], errors="coerce")
df["PT_Gen_Loss"] = pd.to_numeric(df["PT_Gen_Loss"], errors="coerce")
df["Boson_Gen_Loss"] = pd.to_numeric(df["Boson_Gen_Loss"], errors="coerce")
df["Boson_Gen_Nonlinear_Loss"] = pd.to_numeric(df["Boson_Gen_Nonlinear_Loss"], errors="coerce")
df["Gaussian"] = pd.to_numeric(df["Gaussian"], errors="coerce")

# Drop rows where Epoch is NaN (like "Pre-training_End")
df = df.dropna(subset=["Epoch"])

# Split data into Pre-training and Main Training
pre_training = df[df["Run"] == "Pre-training"]
main_training = df[df["Run"] != "Pre-training"]
# Aggregate across seeds: compute mean losses per epoch
pre_grouped = pre_training.groupby("Epoch")[["PT_Gen_Loss","Boson_Gen_Loss","Boson_Gen_Nonlinear_Loss","Gaussian"]].mean().reset_index()
main_grouped = main_training.groupby("Epoch")[["PT_Gen_Loss","Boson_Gen_Loss","Boson_Gen_Nonlinear_Loss","Gaussian"]].mean().reset_index()

# Plotting
plt.figure(figsize=(12, 8))

# Plot Pre-training data
plt.plot(pre_grouped["Epoch"], pre_grouped["PT_Gen_Loss"], label="PT Generator", color="blue")
plt.plot(pre_grouped["Epoch"], pre_grouped["Boson_Gen_Loss"], label="Boson Generator (Linear)", color="orange")
plt.plot(pre_grouped["Epoch"], pre_grouped["Boson_Gen_Nonlinear_Loss"], label="Boson Generator (Nonlinear)", color="red")
plt.plot(pre_grouped["Epoch"], pre_grouped["Gaussian"], label="Gaussian", color="green")

# Add vertical line at Epoch 250 (start of main training)
plt.axvline(250, color='black', linestyle=':')

# Plot Main training data
plt.plot(main_grouped["Epoch"]+250, main_grouped["PT_Gen_Loss"], color="blue")
plt.plot(main_grouped["Epoch"]+250, main_grouped["Boson_Gen_Loss"], color="orange")
plt.plot(main_grouped["Epoch"]+250, main_grouped["Boson_Gen_Nonlinear_Loss"], color="red")
plt.plot(main_grouped["Epoch"]+250, main_grouped["Gaussian"], color="green")

plt.text(175, 0.5, 'Main training start', fontsize = 12)

# Formatting
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
#plt.title('Losses during Pre-training and Main Training')

# Show the plot
plt.show()
