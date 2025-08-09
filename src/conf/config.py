import configparser
import ast

def load_config(path: str, section: str) -> dict:
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(path)

    if section not in config:
        raise ValueError(f"Section [{section}] not found in {path}")

    def parse_value(value: str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value

    return {key: parse_value(value) for key, value in config[section].items()}