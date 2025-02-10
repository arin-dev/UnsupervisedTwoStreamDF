import csv
import sys

def count_folders(csv_file):
    two_id_count_0 = 0
    two_id_count_1 = 0
    one_id_count_0 = 0
    one_id_count_1 = 0
    Deepfake_DF_count = 0
    Deepfake_R_count = 0
    Real_DF_count = 0
    Real_R_count = 0

    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            # folder, cluster_label = row[0], int(row[1])
            folder, cluster_label = row[0], row[1]
            id_count = folder.count('id')
            
            if id_count == 2:
                if cluster_label == "0":
                    two_id_count_0 += 1
                elif cluster_label == "1":
                    two_id_count_1 += 1
            elif id_count == 1:
                if cluster_label == "0":
                    one_id_count_0 += 1
                elif cluster_label == "1":
                    one_id_count_1 += 1
            if id_count == 2:
                if cluster_label == "Deepfake":
                    Deepfake_DF_count += 1
                elif cluster_label == "Real":
                    Deepfake_R_count += 1
            elif id_count == 1:
                if cluster_label == "Deepfake":
                    Real_DF_count += 1
                elif cluster_label == "Real":
                    Real_R_count += 1

    print(f"Two 'id' folders with cluster_label 0: {two_id_count_0}")
    print(f"Two 'id' folders with cluster_label 1: {two_id_count_1}")
    print(f"One 'id' folders with cluster_label 0: {one_id_count_0}")
    print(f"One 'id' folders with cluster_label 1: {one_id_count_1}")
    print(f"Two 'id' folders with cluster_label 0 | Deepfake_DF_count: {Deepfake_DF_count}")
    print(f"Two 'id' folders with cluster_label 1 | Deepfake_R_count: {Deepfake_R_count}")
    print(f"One 'id' folders with cluster_label 0 | Real_DF_count: {Real_DF_count}")
    print(f"One 'id' folders with cluster_label 1 | Real_R_count: {Real_R_count}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python accuracy_calculator.py <csv_file>")
    else:
        count_folders(sys.argv[1])

