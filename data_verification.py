

import pandas as pd


import os
os.chdir('/homes/jheimann/Masterarbeit')

# Get current directory
current_dir = os.getcwd()
print("Current directory:", current_dir)

# List files and folders in current directory
files_and_dirs = os.listdir(current_dir)
print("Files and directories:", files_and_dirs)






# Load the CSV
df = pd.read_csv('results/cam_comparison_metrics2.csv')

# Define the word you want to remove
word_to_remove = 'L1BrendelBethgeAttack'

# Filter out rows where 'method_name' contains the word
df_filtered = df[~df['attack_name'].astype(str).str.contains(word_to_remove, case=False)]



print(df_filtered["attack_name"].value_counts())


# Save the cleaned file
df_filtered.to_csv('your_file_filtered.csv', index=False)


for i in range(4):

    file_name = f"results/cam_comparison_metrics{i}.csv"


    result = df.groupby('attack_name')['image_id'].nunique()
    print(result)
