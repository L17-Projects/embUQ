import os
import subprocess

# Base folder containing all simulation subfolders
base_folder = os.getcwd()  # Assumes script is in the parent folder of a0, a20, ...
#print(sorted(os.listdir(base_folder)))
# Iterate over all subfolders
for folder in sorted(os.listdir(base_folder)):
    # Skip folder "g0" and ensure it's a valid "a<number>" folder
    if folder.startswith("a") and folder[1:].isdigit():
        folder_path = os.path.join(base_folder, folder)
        run_script = os.path.join(folder_path, "run_all_HPC.sh")

        # Check if the script exists
        if os.path.exists(run_script):
            print(f"Submitting job for folder: {folder}")
            try:
                # Execute the script
                subprocess.run(["bash", run_script], cwd=folder_path, check=True)
                print(f"Job submitted successfully for {folder}")
            except subprocess.CalledProcessError as e:
                print(f"Error submitting job for {folder}: {e}")
        else:
            print(f"run_all_HPC.sh not found in {folder}, skipping.")

