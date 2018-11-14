import argparse
from importlib import import_module
from . import scripts

class ExtractAction(argparse.Action):
  def __init__(self, option_strings, dest, nargs=None, **kwargs):
    super(ExtractAction, self).__init__(option_strings, dest)

  def __call__(self, parser, namespace, values, option_string = None):
    print(parser, namespace, values)

class Parser:
  def __init__(self):
    self.parser = argparse.ArgumentParser(description='Distributed Deep Fake')
    self.subparsers = self.parser.add_subparsers(dest='command')
    
    # Extract parser
    self.extract_parser = self.subparsers.add_parser('extract', help='extract faces from videos\
    provided from web links or data/video/ path')
    self.extract_parser.add_argument('-t', '--test', type=str, nargs='+', dest='test')
    self.extract_parser.add_argument(type=str, nargs='+', dest='links')
    
    self.vars = vars(self.parser.parse_args())
    func = self.import_func(self.vars['command'])
    func(self.vars).process()

  def import_func(self, command):
    mod = ".".join(("","scripts", command.lower()))
    module = import_module(mod, package='lib')
    return getattr(module, command.title())

  def get_parser(self):
    return self.vars;
