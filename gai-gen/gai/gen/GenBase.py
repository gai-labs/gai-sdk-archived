from gai_common import generators_utils
class GenBase:

    def __init__(self,generator_name, config_path=None):
        self.generator_name = generator_name
        self.config = generators_utils.load_generators_config(config_path)[generator_name]
