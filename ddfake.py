import lib.cli as cli

if __name__ == '__main__':
  PARSER = cli.Parser()
  print(PARSER.get_parser())
