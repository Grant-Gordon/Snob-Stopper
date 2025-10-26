import os
import csv


def count_elements_in_subdirectories(base_path):
    subdirectory_counts = []

    with os.scandir(base_path) as entries:
        for entry in entries:
            if entry.is_dir():
                subdir_path = entry.path
                num_files = 0
                for root, dirs, files in os.walk(subdir_path):
                    num_files += len(files)
                subdirectory_counts.append((entry.name, num_files))

    return subdirectory_counts


def write_counts_to_csv(counts, output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)

        for count in counts:
            writer.writerow(count)


# Define the base path and output CSV file
base_path = '../Data/Images_Training'
output_csv = 'Training_data_count.csv'

# Get the counts and write to CSV
counts = count_elements_in_subdirectories(base_path)

write_counts_to_csv(counts, output_csv)

print(f"Counts have been written to {output_csv}")
