"""CLI entry point for AI Product Describer."""
import argparse
from pathlib import Path
from . import core

def main(argv=None):
    parser=argparse.ArgumentParser(prog="ai-product-describer",description="Train a tiny GPT and generate product descriptions.")
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("hello",help="check installation")
    sub.add_parser("data",help="write the starter coffee description corpus")
    train=sub.add_parser("train",help="train the custom GPT model"); train.add_argument("--steps",type=int,default=1200); train.add_argument("--vocab-size",type=int,default=320)
    gen=sub.add_parser("generate",help="generate a description from product details"); gen.add_argument("--name",required=True); gen.add_argument("--category",default="coffee"); gen.add_argument("--features",default=""); gen.add_argument("--tone",default="friendly"); gen.add_argument("--max-tokens",type=int,default=120); gen.add_argument("--temperature",type=float,default=.8); gen.add_argument("--top-k",type=int,default=40)
    args=parser.parse_args(argv)
    if args.command=="hello": print("AI Product Describer is ready.")
    elif args.command=="data": print(f"Created corpus: {core.collect_corpus()}")
    elif args.command=="train":
        if args.steps<1: parser.error("--steps must be positive")
        path=core.train_model(steps=args.steps,vocab_size=args.vocab_size); print(f"Training finished. Saved model: {path}")
    elif args.command=="generate":
        prompt=f"Product: {args.name} | Category: {args.category} | Features: {args.features} | Tone: {args.tone} | Description:"
        try: print(core.generate_description(prompt,max_new_tokens=args.max_tokens,temperature=args.temperature,top_k=args.top_k))
        except FileNotFoundError: parser.error("Model missing. Run `python -m app.cli train` first.")

if __name__=="__main__": main()
