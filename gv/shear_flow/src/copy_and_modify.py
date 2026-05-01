import os
import shutil

# Base directory and file names
base_folder = "a0"
parameters_file = "equil.py"
generate_file = "generate.py"

# Simulation parameters
a_start = 0.0
a_end = 320.0
a_step = 20.0

# Function to update equil.py
def update_parameters(file_path, afsi_value):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    with open(file_path, 'w') as file:
        for line in lines:
            if line.strip().startswith("afsi ="):
                file.write(f"afsi = {afsi_value}\n")
            else:
                file.write(line)

# Function to update generate.py
def update_generate(file_path, afsi_value):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    with open(file_path, 'w') as file:
        for line in lines:
            if "--job-name=" in line:
                file.write(f'#SBATCH --job-name="a{int(afsi_value)}"\n')
            else:
                file.write(line)

# Iterate over afsi values
for afsi in range(int(a_start), int(a_end + a_step), int(a_step)):
    new_folder = f"a{afsi}"
    
    # Copy the base folder to create a new simulation folder
    if os.path.exists(new_folder):
        print(f"Folder {new_folder} already exists. Skipping.")
        continue
    shutil.copytree(base_folder, new_folder)
    
    # Update equil.py
    parameters_path = os.path.join(new_folder, parameters_file)
    update_parameters(parameters_path, float(afsi))
    
    # Update generate.py
    generate_path = os.path.join(new_folder, generate_file)
    update_generate(generate_path, float(afsi))
    
    print(f"Created and updated {new_folder}")

