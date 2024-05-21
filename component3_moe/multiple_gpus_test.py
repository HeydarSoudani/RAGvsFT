#!/usr/bin/env python3

import argparse
from accelerate import Accelerator
from accelerate.utils import gather_object


def main(args):
    accelerator = Accelerator()

    # each GPU creates a string
    message=[ f"Hello this is GPU {accelerator.process_index}" ] 

    # collect the messages from all GPUs
    messages=gather_object(message)

    # output the messages only on the main process with accelerator.print() 
    accelerator.print(messages)
    
    print(args.index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, required=True)

    
    args = parser.parse_args()
    main(args)