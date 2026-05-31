from load_data import main as load_data
from preprocess import main as preprocess
from score import main as score
from save_result import main as save_result


def main():
    load_data()
    preprocess()
    score()
    save_result()


if __name__ == "__main__":
    main()
