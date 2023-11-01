# -*- coding: utf-8 -*-
# (c) 2023 Hanshi Sun LGPL
# basic arguments


import argparse


class ParserArgs(object):
    """
    arguments to be used in the experiment
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.get_general_parser()

    def get_general_parser(self):
        # training settings
        self.parser.add_argument("--config", type=str, default="configs/init.yaml", help="config yaml file")

    def get_args(self):
        args = self.parser.parse_args()
        return args