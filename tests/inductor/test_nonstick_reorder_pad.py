# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Approach 1 (pad + reorder) for non-stick strided views.

``x[:, ::3, :]`` on a 97-row dim is rejected today: ``Mod(3*c1, 97)`` for
``c1`` doesn't fit ``Term``'s accepted shapes, because 97 isn't a multiple
of the step (3). Reformulating so the sliced axis is the *outermost*
logical dim, and padding it up to the next multiple of the step (97 ->
99), lets the default layout machinery produce a coordinate the existing
Term already accepts (``3*c1``, no Mod) -- no change to Term's shape, only
to the physical layout the access is expressed against. Combined with the
two fixes in views.py (synthetic variable for the zero-offset gap term;
the off-by-one boundary-check fix), this compiles and executes correctly.

See the ``generalize-view`` skill (``.claude/skills/generalize-view/``) for
the full investigation, including the two sibling approaches (loop-based
lowering, and wiring deeptools' existing modulo arithmetic into the
address path) that this test does not cover.
"""

import unittest

import torch

from utils_inductor import compare_with_cpu


class TestNonstickReorderPad(unittest.TestCase):
    def test_strided_slice_via_pad_and_reorder(self):
        rows, step = 97, 3
        padded = -(-rows // step) * step  # next multiple of step >= rows

        cpu = torch.randn(2, rows, 256, dtype=torch.float16)
        # Row made the outermost logical dim, and padded to a clean
        # multiple of the step -- the padding rows are never addressed by
        # the resulting access pattern, so their content is immaterial.
        cpu_padded = torch.zeros(padded, 2, 256, dtype=torch.float16)
        cpu_padded[:rows] = cpu.permute(1, 0, 2)

        def f(t):
            return t[::step, :, :] + 1

        compare_with_cpu(f, cpu_padded)


if __name__ == "__main__":
    unittest.main()
