import os
import shutil
import argparse

vid_list = [7,8,9,22,23,12,41,51,52,116,117,118,119,120,]

def move_files(source_folder, destination_folder, action):
    # Check if source folder and destination folder exist
    if not os.path.exists(source_folder):
        raise FileNotFoundError("Source folder does not exist.")
    
    # Create destination folder if it does not exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Loop through all files in the source folder
    for filename in os.listdir(source_folder):
        if "video" in filename:
            continue
        vid = int(os.path.splitext(filename)[0][:3])
        if vid < 96:
            continue
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, filename)
        
        # Copy or move the file to the destination folder
        if action == 'copy':
            shutil.copy(source_path, destination_path)
        elif action == 'cut':
            shutil.move(source_path, destination_path)
        else:
            raise ValueError("Invalid action. Please use 'copy' or 'cut'.")

# Create ArgumentParser object
parser = argparse.ArgumentParser(description='Move files from source folder to destination folder.')

# Add command line arguments
parser.add_argument('action', choices=['copy', 'cut'], help='Specify the action: copy or cut.')
parser.add_argument('source_folder', type=str, help='Specify the source folder path.')
parser.add_argument('destination_folder', type=str, help='Specify the destination folder path.')

# Parse the command line arguments
args = parser.parse_args()

# Move (copy or cut) the files
move_files(args.source_folder, args.destination_folder, args.action)
