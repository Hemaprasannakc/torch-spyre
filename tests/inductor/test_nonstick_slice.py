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

"""End-to-end coverage for bounded outer non-stick steps (issue #5388)."""

import pytest
import torch
import torch_spyre  # noqa: F401


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "shape,step,start",
    [
        ((2, 97, 256), 3, 0),
        ((2, 96, 256), 3, 0),
        ((1, 97, 256), 3, 0),
        ((2, 98, 256), 5, 0),
        ((2, 97, 256), 3, 1),
        ((2, 7, 3, 128), 3, 0),
    ],
)
def test_bounded_nonstick_slice(shape, step, start, dtype):
    # Small dyadic values survive device conversion exactly; a wrong row or
    # batch must fail an exact comparison instead of hiding in a tolerance.
    gen = torch.Generator().manual_seed(5388)
    x = torch.randint(-16, 17, shape, generator=gen).to(dtype) / 8

    def fn(t):
        return t[:, start::step, ...] + 1

    got = torch.compile(fn, dynamic=False, fullgraph=True)(x.to("spyre")).cpu()
    torch.testing.assert_close(got, fn(x), atol=0, rtol=0)
