from src.data.data_loader import DataLoader


def main():
    # Acess cfg file and get the path to the data file

    # Data Loading
    data_loader = DataLoader("data/processed_data/creditcard_test_light099.csv")
    data = data_loader.load_csv()


if __name__ == "__main__":
    main()
