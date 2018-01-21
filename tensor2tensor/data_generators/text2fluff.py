# coding=utf-8
from tensor2tensor.data_generators.text_encoder import TokenTextEncoder
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators.translate import token_generator
from tensor2tensor.utils import registry
from tensor2tensor.models import transformer
import os

@registry.register_problem()
class Text2fluff(problem.Text2TextProblem):
    """Problem spec for English word to dictionary definition."""

    @property
    def is_character_level(self):
        return False

    @property
    def vocab_name(self):
        return "vocab"

    @property
    def input_space_id(self):
        return problem.SpaceID.EN_TOK

    @property
    def targeted_vocab_size(self):
        return 2 ** 14

    @property
    def target_space_id(self):
        return problem.SpaceID.EN_TOK

    @property
    def num_shards(self):
        return 100

    def generator(self, data_dir, tmp_dir, train):
        text_encoder = TokenTextEncoder(os.path.join(data_dir, 'vocab'),
                                        replace_oov='<unk>')
        EOS=None
        if train:
            datasets = (os.path.join(data_dir, 'train.src'),
                        os.path.join(data_dir, 'train.target'))
        else:
            datasets = (os.path.join(data_dir, 'train.src'),
                        os.path.join(data_dir, 'train.target'))

        return token_generator(datasets[0], datasets[1], text_encoder,
                                   EOS)

    @property
    def use_subword_tokenizer(self):
        return False


@registry.register_hparams
def text2fluff_hparams():
    # hparams = transformer.transformer_base_single_gpu()  # Or whatever you'd like to build off.
    hparams = transformer.transformer_base()
    hparams.batch_size = 256
    return hparams
