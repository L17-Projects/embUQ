import os

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

# Example usage
input_folder = "../trj_eq/sim00001"  # Replace with the path to your folder containing XYZ files
output_file = "output/positions.xyz"  # Output file name
combine_xyz_files(input_folder, output_file)
