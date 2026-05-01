from MDAnalysis.analysis import align
import MDAnalysis as mda
import numpy as np

#aligned_u = mda.Universe('emb_0000000.xyz', "output/rmsfit.xyz", format="XYZ", dt=1)

aligned_u = mda.Universe('emb_0000000.xyz', 'output/rmsfit.xyz', format="XYZ", dt=1)

# Select the atoms for which you want to calculate the average structure
selection = aligned_u.select_atoms("all")

# Create an array to store the positions
n_frames = len(aligned_u.trajectory)
positions = np.zeros((n_frames, selection.n_atoms, 3))

# Loop over frames and collect positions
for i, ts in enumerate(aligned_u.trajectory):
    positions[i] = selection.positions

# Calculate the average structure
average_structure = positions.mean(axis=0)

# Assign the average structure to the Universe for saving or visualization
selection.positions = average_structure

# Save the average structure
selection.write("average_structure.xyz")


