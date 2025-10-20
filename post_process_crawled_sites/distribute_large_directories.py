import os
import shutil
from pathlib import Path

# Define the source directory and the base target directories
source_dir = Path().home().joinpath(Path.home().cwd(), 'server/crawled/sampled11')
target_dirs = [
    Path().home().joinpath(Path.home().cwd(), 'server/crawled/sampled11_1'),
    Path().home().joinpath(Path.home().cwd(), 'server/crawled/sampled11_2'),
    Path().home().joinpath(Path.home().cwd(), 'server/crawled/sampled11_3')
]

# Ensure target directories exist
for target_dir in target_dirs:
    target_dir.mkdir(parents=True, exist_ok=True)

# Get a list of directories in the source directory
directories = [d for d in source_dir.iterdir() if d.is_dir()]

# Distribute directories
for i, directory in enumerate(directories):
    # Choose target directory based on round-robin distribution
    target_dir = target_dirs[i % len(target_dirs)]
    
    # Define new path for the directory in the target directory
    new_directory_path = target_dir / directory.name
    
    # Move the directory (use shutil.move) or copy if you want to keep the original
    shutil.move(str(directory), str(new_directory_path))
    # For copying instead of moving, use shutil.copytree(str(directory), str(new_directory_path))
    # Note: shutil.copytree requires the target directory not to exist.

print("Distribution complete.")