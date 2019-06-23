import unittest
from pathlib import Path

import box
import torch.nn as nn
from torch.optim import SGD

import torch_model_state
import torch_model_state.configs


@box.register(tag='model')
class MockModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 16, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)

    @classmethod
    def factory(cls, config):
        return cls()


class MyTestCase(unittest.TestCase):
    def test_state_utils(self):
        state_path = Path(__file__).parent / 'build' / 'checkpoint.sf'

        config = {
            'type': 'MockModule',
        }
        info = {
            'top-1': 100.0
        }

        model = MockModule()
        optimizer = SGD(model.parameters(), lr=0.1)

        state = torch_model_state.to_state(model=model, config=config, optimizers=[optimizer], info=info)
        torch_model_state.save_state_file(state, state_path)
        self.assertTrue(state_path.exists())

        torch_model_state.from_state(state, model, [optimizer], device='cpu')

        model = torch_model_state.load_model_from_state(state_path, device='cpu')
        self.assertTrue(str(model).startswith('MockModule'))

    def test_argument_config(self):
        self.assertIsNotNone(torch_model_state.configs.ArgumentConfig())


if __name__ == '__main__':
    unittest.main()
