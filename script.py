import os
import glob

def delete_files_with_prefix(directory, prefix='._'):
    """
    Recursively deletes files with a specific prefix in the given directory and its subdirectories.
    
    Parameters:
    - directory: Directory to search for files.
    - prefix: Prefix of the files to delete.
    """
    # Create a search pattern
    pattern = os.path.join(directory, '**', f'{prefix}*')
    
    # Use glob to find all matching files
    files_to_delete = glob.glob(pattern, recursive=True)
    
    # Delete each file found
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

# Specify the directory to search
directory_to_clean = '/home/rob/Desktop/abstract/Task01_BrainTumour/labelsTr/'

# Run the cleanup function
delete_files_with_prefix(directory_to_clean)
