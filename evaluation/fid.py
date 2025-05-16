import argparse

from cleanfid import fid


def compute_fid(args):
    if not fid.test_stats_exists(args.dataset_name, mode="clean"):
        fid.make_custom_stats(args.dataset_name, args.real_images, mode="clean")
    print(
        fid.compute_fid(
            args.fake_images,
            dataset_name=args.dataset_name,
            mode="clean",
            dataset_res=args.dataset_res,
            dataset_split="custom"
        )
    )


def main(args):
    compute_fid(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="mjhq")
    parser.add_argument("--dataset_res", type=int, default=512)
    parser.add_argument("--real_images", type=str, default="/data/bingda/mjhq")
    parser.add_argument("--fake_images", type=str, default="/data/bingda/fake_images")
    args = parser.parse_args()
    main(args)