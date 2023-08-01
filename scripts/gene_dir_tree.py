import os

def generate_tree(folder_path, indent='', max_depth=None, current_depth=0):
    # Get the list of files and folders in the current directory
    items = os.listdir(folder_path)
    
    for item in items:
        # Construct the full path
        item_path = os.path.join(folder_path, item)
        
        # Check if it is a file or a folder
        if os.path.isfile(item_path):
            # Print the file name with proper indentation
            # print(indent + '|-- ' + item)
            pass
        else:
            # Print the folder name with proper indentation
            print(indent + '|-- ' + item + '/')
            
            # Recursively call the function on the subfolder, if depth limit is not reached
            if max_depth is None or current_depth < max_depth - 1:
                generate_tree(item_path, indent + '    ', max_depth, current_depth + 1)

# Provide the folder path for which you want to generate the tree
folder_path = 'dataset_v1'

# Set the maximum depth limit (change it according to your requirement)
max_depth = 3

# Call the function to generate the tree structure with the specified depth limit
generate_tree(folder_path, max_depth=max_depth)
