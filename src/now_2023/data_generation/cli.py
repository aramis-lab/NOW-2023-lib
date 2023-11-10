import argparse


if __name__ == "__main__":
    from ._toy_dataset import generate_toy_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("n_samples", help="The number of samples to generate.", type=int)
    parser.add_argument("output", help="Output file name.", type=str)
    args = parser.parse_args()
    generate_toy_dataset(n_samples=args.n_samples, output_filename=args.output)
