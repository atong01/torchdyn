# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torchdyn.core import NeuralODE


def test_integration():
    """Test load and save problems with adjoint sensitivity methods"""
    f = torch.nn.Sequential(
        torch.nn.Linear(1, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1)
    )
    node = NeuralODE(f)
    x = torch.zeros(5, 1)
    t_span = torch.linspace(0, 1, 11)
    _ = node(x, t_span)
