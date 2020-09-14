from decision_tree import DecisionTree
import numpy as np
import argparse


def load_file(path):
    return np.loadtxt(path, delimiter=",", dtype="int")


def main():
    parser = argparse.ArgumentParser(description="Instructions:")
    parser.add_argument(
        "-m",
        dest="mode",
        help="impurity heuristic: 1 for entrophy, 2 for variance",
        type=int,
        choices=[1, 2],
        required=True,
    )
    parser.add_argument(
        "-c",
        dest="clause",
        help="clause count",
        type=int,
        choices=[300, 500, 1000, 1500, 1800],
        required=True,
    )
    parser.add_argument(
        "-s",
        dest="sample",
        help="sample count",
        type=int,
        choices=[100, 1000, 5000],
        required=True,
    )
    parser.add_argument(
        "-d", dest="dbp", help="depth-based pruning", action="store_true"
    )
    parser.add_argument(
        "-r", dest="rep", help="reduced error pruning", action="store_true"
    )
    parse(parser.parse_args())


def parse(args):
    id3 = DecisionTree(DecisionTree.Mode(args.mode))
    train_data = load_file(f"data/train_c{args.clause}_d{args.sample}.csv")
    valid_data = load_file(f"data/valid_c{args.clause}_d{args.sample}.csv")
    test_data = load_file(f"data/test_c{args.clause}_d{args.sample}.csv")
    id3.train(train_data)
    if args.dbp:
        print(id3.depth_based_prune(valid_data, [5, 10, 15, 20, 50, 100]))
    if args.rep:
        id3.reduced_error_prune(valid_data)
    accuracy = id3.test(test_data)
    print(accuracy)


if __name__ == "__main__":
    main()
