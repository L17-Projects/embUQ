import os
from pathlib import Path

import numpy as np


def _write_xyz_frame(outfile, positions, frame_number):
    outfile.write(f"{len(positions)}\n")
    outfile.write(f"Frame {frame_number}\n")
    atoms = np.zeros(len(positions))
    np.savetxt(outfile, np.column_stack((atoms, positions)))


def combine_xyz_files(input_folder, output_file):
    """
    Combines multiple XYZ files in a directory into a single XYZ trajectory file.

    Args:
        input_folder (str): Path to the folder containing XYZ files.
        output_file (str): Path to the output XYZ trajectory file.
    """
    # Get all files in the input folder, sorted alphabetically
    xyz_files = sorted(f for f in os.listdir(input_folder) if f.endswith('.xyz'))

    ln = len(xyz_files)
    if ln == 0:
        raise FileNotFoundError(f"No XYZ files found in {input_folder}")

    with open(output_file, 'w') as outfile:
        frame_number = 1

        for xyz_file in xyz_files:
            file_path = os.path.join(input_folder, xyz_file)

            with open(file_path, 'r') as infile:
                lines = infile.readlines()
                # The first line is the number of atoms
                num_atoms = lines[0].strip()
                # The atomic positions start from the third line
                atomic_data = lines[2:]

                # Write to the output file
                outfile.write(f"{num_atoms}\n")
                outfile.write(f"Frame {frame_number}\n")
                outfile.writelines(atomic_data)

            frame_number += 1

            print(f'Frame number {frame_number -1}/{ln} done')

    print(f"Combined {len(xyz_files)} XYZ files into {output_file}")


def combine_restart_position_files(restart_folder, output_file):
    import h5py

    restart_path = Path(restart_folder)
    h5_files = sorted(restart_path.glob("emb.PV-*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No emb.PV restart HDF5 files found in {restart_folder}")

    with open(output_file, "w") as outfile:
        for frame_number, h5_file in enumerate(h5_files, start=1):
            with h5py.File(h5_file, "r") as handle:
                positions = np.asarray(handle["position"])
            _write_xyz_frame(outfile, positions, frame_number)
            print(f"Restart frame {frame_number}/{len(h5_files)} done: {h5_file}")

    print(f"Combined {len(h5_files)} restart frames into {output_file}")


# Example usage
input_folder = Path("../trj_eq/sim00001")
output_file = Path("output/positions.xyz")
output_file.parent.mkdir(parents=True, exist_ok=True)

if input_folder.is_dir():
    combine_xyz_files(input_folder, output_file)
else:
    combine_restart_position_files("../restart", output_file)
