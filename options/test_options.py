from options.base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes training options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # network saving and loading parameters
        parser.add_argument('--load_model_name', type=str, default='empty_path', help='path to save the trained model')

        self.isTrain = False
        return parser