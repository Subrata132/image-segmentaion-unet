import argparse
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--train_img", default='data/train/', type=str)
    parser.add_argument("--test_img", default='data/val/', type=str)
    parser.add_argument("--test", default=False, action="store_true")
    args = parser.parse_args()
    trainer = Trainer(
        batch_size=args.batch_size,
        epochs=args.epochs,
        train_image_path=args.train_img,
        test_image_path=args.test_img
    )
    trainer.train(test=args.test)


if __name__ == '__main__':
    main()