from rawMNISTDatasetCreator import rawMNISTDataset

def main():
    creator = rawMNISTDataset("data/", 64, "CP")
    creator.create_csv_files()

if __name__ == '__main__':
    main()