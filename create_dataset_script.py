from rawMNISTDatasetCreator import rawMNISTDataset

def main():
    creator = rawMNISTDataset("data/", 16, "CP")
    creator.create_csv_files(verbose=True)

if __name__ == '__main__':
    main()