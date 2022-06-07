import argparse
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    args = parser.parse_args()
    trainer = Trainer(batch_size=args.batch_size, epochs=args.epochs)
    trainer.train()


if __name__ == '__main__':
    main()