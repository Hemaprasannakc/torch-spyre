# Copyright 2025 The Torch-Spyre Authors.
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

"""Tests for automatic span-overflow coarse-tiling hints.

These tests intentionally mirror the compiler layers used by user
``spyre_hint`` coarse tiling:

1. Planner: span_overflow_hint_analysis returns a selected dim and split count.
2. Adapter: span_overflow_groups creates a synthetic DimHint/group.
3. Coarse-tile IR: coarse_tile consumes the group and stamps CoarseTileInfo.
4. Scheduler/codegen: generated source contains the expected LoopSpec count.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import sympy
import torch
from torch._inductor.ir import ComputedBuffer, FlexibleLayout, Pointwise
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code

from torch_spyre._C import SpyreTensorLayout
from torch_spyre._inductor import config
from torch_spyre._inductor.propagate_hints import DimHint
from torch_spyre._inductor.coarse_tile import (
    coarse_tile,
    span_overflow_groups,
)
from torch_spyre._inductor.ir import FixedTiledLayout
from torch_spyre._inductor.scheduler import (
    CountedLoopSchedulerNode,
    build_loop_scheduler_nodes,
)
from torch_spyre._inductor.span_overflow_hint_analysis import (
    plan_span_overflow_tile,
)
import torch_spyre._inductor.propagate_named_dims as _pnd


_LAUNCH_KERNEL = "torch_spyre.execution.kernel_runner.launch_kernel"


def _fixed_tiled_layout(shape, dtype=torch.float16):
    """Build the same kind of physical layout used by real Spyre lowering."""
    size = list(shape)
    stride = list(FlexibleLayout.contiguous_strides(size))
    stride_ints = [int(s) for s in stride]
    size_ints = [int(s) for s in size]
    within_stick_dim = len(size_ints) - 1
    dim_order = [i for i in range(len(size_ints)) if i != within_stick_dim]
    dim_order.append(within_stick_dim)
    device_layout = SpyreTensorLayout(size_ints, stride_ints, dtype, dim_order)
    return FixedTiledLayout("spyre:0", dtype, size, stride, device_layout)


def _pointwise_op(shape, name="buf0"):
    """Return a real ComputedBuffer with a lightweight Pointwise mock."""
    data = MagicMock(spec=Pointwise)
    data.ranges = list(shape)
    op = ComputedBuffer(
        name=name,
        layout=_fixed_tiled_layout(shape),
        data=data,
    )
    op.operation_name = name
    return op


def _graph(operations):
    return SimpleNamespace(operations=operations)


def _out_coords_for_bhld(_op):
    """Coordinates for shape [B, H, L, D] with B size 1 in these tests."""
    return [
        sympy.Integer(0),
        sympy.Symbol("h"),
        sympy.Symbol("l"),
        sympy.Symbol("d"),
    ]


def _run_span_overflow_groups(op):
    """Run span_overflow_groups with op_out_coords patched for one test op."""
    graph = _graph([op])

    with patch("torch_spyre._inductor.coarse_tile.op_out_coords", _out_coords_for_bhld):
        return span_overflow_groups(graph)


_E2E_SHAPE = (1, 8195, 256, 64)
_E2E_SPLIT_COUNT = 5
_E2E_TILE_SHAPE = [1, 1639, 256, 64]


def _manual_h_hint_group(op, hint_id=1, split_count=_E2E_SPLIT_COUNT):
    """Return the coarse-tile group produced by spyre_hint over dim H."""
    hint = DimHint(
        dim_names=["H"],
        split_count=split_count,
        loop_var=sympy.Symbol("h"),
        is_reduction=False,
        hint_id=hint_id,
    )
    op.dim_hints = [hint]
    return [([op], [(hint_id, sympy.Integer(split_count), False)])]


def _scheduler_node_for_op(op, name):
    """Return a minimal SchedulerNode mock wrapping one IR op."""
    scheduler = MagicMock()
    scheduler.name_to_fused_node = {}
    scheduler.removed_ops = set()

    snode = MagicMock(spec=SchedulerNode)
    snode.scheduler = scheduler
    snode.node = op
    snode.get_name.return_value = name
    snode.get_nodes.return_value = [snode]
    snode.ancestors = set()
    snode.min_order = 0
    snode.max_order = 0
    return snode


class TestSpanOverflowGroups(InductorTestCase):
    """Adapter-focused tests matching the user-hint group contract.

    These are intentionally close to the coarse-tiling draft tests: build one
    op, patch output coordinates, then inspect the generated group and DimHint.
    """

    def test_no_overflow_returns_empty(self):
        op = _pointwise_op((1, 2, 16, 64), name="small_op")

        with config.patch({"sencores": 4, "chunk_large_tensors": False}):
            groups = _run_span_overflow_groups(op)

        self.assertEqual(groups, [])

    def test_overflow_pointwise_returns_one_group(self):
        op = _pointwise_op(_E2E_SHAPE)

        with config.patch({"sencores": 4, "chunk_large_tensors": False}):
            groups = _run_span_overflow_groups(op)

        self.assertEqual(len(groups), 1)
        self.assertIs(groups[0][0][0], op)

    def test_group_structure(self):
        op = _pointwise_op(_E2E_SHAPE)

        with config.patch({"sencores": 4, "chunk_large_tensors": False}):
            groups = _run_span_overflow_groups(op)

        self.assertEqual(len(groups), 1)
        ops_list, levels = groups[0]
        self.assertEqual(ops_list, [op])
        self.assertEqual(len(levels), 1)
        hint_id, count, is_reduction_level = levels[0]
        self.assertIsInstance(count, sympy.Integer)
        self.assertEqual(count, sympy.Integer(_E2E_SPLIT_COUNT))
        self.assertFalse(is_reduction_level)
        self.assertEqual(hint_id, op.dim_hints[0].hint_id)

    def test_dim_hint_attached_to_op(self):
        from torch_spyre._inductor.propagate_hints import DimHint

        op = _pointwise_op(_E2E_SHAPE)

        with config.patch({"sencores": 4, "chunk_large_tensors": False}):
            _run_span_overflow_groups(op)

        self.assertTrue(hasattr(op, "dim_hints"))
        self.assertEqual(len(op.dim_hints), 1)
        hint = op.dim_hints[0]
        self.assertIsInstance(hint, DimHint)
        self.assertEqual(hint.dim_names, ["_span_overflow"])
        self.assertEqual(hint.split_count, _E2E_SPLIT_COUNT)
        self.assertEqual(hint.loop_var, sympy.Symbol("h"))
        self.assertFalse(hint.is_reduction)

    def test_trip_count_matches_level_and_hint(self):
        op = _pointwise_op(_E2E_SHAPE)

        with config.patch({"sencores": 4, "chunk_large_tensors": False}):
            groups = _run_span_overflow_groups(op)

        _, levels = groups[0]
        _, level_count, _ = levels[0]
        self.assertEqual(op.dim_hints[0].split_count, int(level_count))

    def test_non_fixed_tiled_layout_skipped(self):
        op = MagicMock(spec=ComputedBuffer)
        op.data = MagicMock(spec=Pointwise)
        op.data.ranges = [
            sympy.Integer(1),
            sympy.Integer(20),
            sympy.Integer(16),
            sympy.Integer(64),
        ]
        op.layout = MagicMock()
        op.get_name.return_value = "non_fixed_tiled"
        op.get_operation_name.return_value = "non_fixed_tiled"

        with config.patch({"sencores": 4, "chunk_large_tensors": False}):
            groups = span_overflow_groups(_graph([op]))

        self.assertEqual(groups, [])

    def test_chunk_large_tensors_config_suppresses_groups(self):
        op = _pointwise_op(_E2E_SHAPE)

        with config.patch({"sencores": 4, "chunk_large_tensors": True}):
            groups = _run_span_overflow_groups(op)

        self.assertEqual(groups, [])


class TestSpanOverflowPointwisePlannerAndAdapter(InductorTestCase):
    """Mock-heavy tests for the first three compiler layers."""

    def test_planner_selects_dim_and_split_count(self):
        op = _pointwise_op(_E2E_SHAPE)

        plan = plan_span_overflow_tile(op, max_cores=4)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.selected_host_dim, 1)
        self.assertEqual(plan.split_count, _E2E_SPLIT_COUNT)
        self.assertFalse(plan.is_reduction)
        self.assertEqual(plan.chunking_info.selected_device_dim_size, _E2E_SHAPE[1])

    @patch("torch_spyre._inductor.coarse_tile.op_out_coords", _out_coords_for_bhld)
    def test_adapter_creates_dim_hint_and_group(self):
        op = _pointwise_op(_E2E_SHAPE)

        with config.patch({"sencores": 4, "chunk_large_tensors": False}):
            groups = span_overflow_groups(_graph([op]))

        self.assertEqual(len(groups), 1)
        group_ops, levels = groups[0]
        self.assertEqual(group_ops, [op])
        self.assertEqual(levels[0][1], sympy.Integer(_E2E_SPLIT_COUNT))
        self.assertEqual(levels[0][2], False)
        self.assertEqual(len(op.dim_hints), 1)
        self.assertEqual(op.dim_hints[0].split_count, _E2E_SPLIT_COUNT)
        self.assertEqual(op.dim_hints[0].loop_var, sympy.Symbol("h"))

    @patch("torch_spyre._inductor.coarse_tile.insert_tiling_propagation")
    @patch("torch_spyre._inductor.coarse_tile.op_out_coords", _out_coords_for_bhld)
    def test_coarse_tile_consumes_auto_group_and_stamps_op(
        self,
        _mock_insert_tiling_propagation,
    ):
        op = _pointwise_op(_E2E_SHAPE)

        with config.patch({"sencores": 4, "chunk_large_tensors": False}):
            graph = _graph([op])
            groups = span_overflow_groups(graph)
            coarse_tile(graph, groups)

        self.assertEqual(list(op.data.ranges), _E2E_TILE_SHAPE)
        self.assertEqual(list(op.layout.size), _E2E_TILE_SHAPE)
        self.assertEqual(op.loop_info.loop_count, [sympy.Integer(_E2E_SPLIT_COUNT)])
        self.assertEqual(op.loop_info.loop_tiled_dims, [[1]])
        self.assertEqual(op.loop_info.loop_tiled_reduction_dims, [[]])


class TestSpanOverflowLargeShapeContract(InductorTestCase):
    """Unit-style coverage for the real large shape used in E2E testing."""

    def test_large_shape_planner_adapter_and_coarse_tile_match_manual_hint(self):
        auto_op = _pointwise_op(_E2E_SHAPE, name="auto_buf")
        manual_op = _pointwise_op(_E2E_SHAPE, name="manual_buf")

        # Layer 1: planner chooses the same H split observed in the E2E run.
        plan = plan_span_overflow_tile(auto_op, max_cores=4)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.selected_host_dim, 1)
        self.assertEqual(plan.split_count, _E2E_SPLIT_COUNT)
        self.assertFalse(plan.is_reduction)

        with patch(
            "torch_spyre._inductor.coarse_tile.op_out_coords",
            _out_coords_for_bhld,
        ):
            with patch("torch_spyre._inductor.coarse_tile.insert_tiling_propagation"):
                with config.patch({"sencores": 4, "chunk_large_tensors": False}):
                    # Layer 2: adapter emits the same group shape as user hints.
                    auto_graph = _graph([auto_op])
                    auto_groups = span_overflow_groups(auto_graph)
                    manual_graph = _graph([manual_op])
                    manual_groups = _manual_h_hint_group(manual_op)

                    self.assertEqual(len(auto_groups), 1)
                    self.assertEqual(len(manual_groups), 1)
                    self.assertEqual(auto_groups[0][1][0][1], sympy.Integer(5))
                    self.assertEqual(manual_groups[0][1][0][1], sympy.Integer(5))
                    self.assertFalse(auto_groups[0][1][0][2])
                    self.assertFalse(manual_groups[0][1][0][2])
                    self.assertEqual(auto_op.dim_hints[0].loop_var, sympy.Symbol("h"))
                    self.assertEqual(manual_op.dim_hints[0].loop_var, sympy.Symbol("h"))

                    # Layer 3: coarse_tile stamps identical per-tile IR shape.
                    coarse_tile(auto_graph, auto_groups)
                    coarse_tile(manual_graph, manual_groups)

        self.assertEqual(list(auto_op.data.ranges), _E2E_TILE_SHAPE)
        self.assertEqual(list(manual_op.data.ranges), _E2E_TILE_SHAPE)
        self.assertEqual(list(auto_op.layout.size), _E2E_TILE_SHAPE)
        self.assertEqual(list(manual_op.layout.size), _E2E_TILE_SHAPE)
        self.assertEqual(auto_op.loop_info.loop_count, [sympy.Integer(5)])
        self.assertEqual(manual_op.loop_info.loop_count, [sympy.Integer(5)])
        self.assertEqual(auto_op.loop_info.loop_tiled_dims, [[1]])
        self.assertEqual(manual_op.loop_info.loop_tiled_dims, [[1]])
        self.assertEqual(auto_op.loop_info.loop_tiled_reduction_dims, [[]])
        self.assertEqual(manual_op.loop_info.loop_tiled_reduction_dims, [[]])

        # Layer 4: scheduler wrapping sees the same counted loop on both paths.
        created = []

        def fake_create(snodes, loop_count):
            node = MagicMock(spec=CountedLoopSchedulerNode)
            node.snodes = snodes
            node.loop_count = loop_count
            node.get_nodes.return_value = snodes
            node.get_name.return_value = "_".join(n.get_name() for n in snodes)
            node.scheduler = snodes[0].scheduler
            created.append(node)
            return node

        auto_snode = _scheduler_node_for_op(auto_op, "auto_snode")
        manual_snode = _scheduler_node_for_op(manual_op, "manual_snode")
        with patch.object(CountedLoopSchedulerNode, "create", staticmethod(fake_create)):
            auto_wrapped = build_loop_scheduler_nodes([auto_snode])
            manual_wrapped = build_loop_scheduler_nodes([manual_snode])

        self.assertEqual(len(auto_wrapped), 1)
        self.assertEqual(len(manual_wrapped), 1)
        self.assertEqual(created[0].loop_count, sympy.Integer(5))
        self.assertEqual(created[1].loop_count, sympy.Integer(5))
        self.assertEqual(auto_wrapped[0].loop_count, manual_wrapped[0].loop_count)


class TestSpanOverflowPointwiseCodegen(InductorTestCase):
    """Small codegen test for scheduler/codegen LoopSpec emission."""

    @patch("torch_spyre._inductor.span_overflow_hint_analysis.MAX_SPAN_BYTES", 8192)
    @config.patch(
        {
            "sencores": 4,
            "chunk_large_tensors": False,
            "unroll_loops": False,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_codegen_contains_auto_span_overflow_loop_spec(self):
        x = torch.randn(1, 20, 16, 64, dtype=torch.float16).to("spyre")
        y = torch.randn(1, 20, 16, 64, dtype=torch.float16).to("spyre")

        def fn(x, y):
            return x + y

        cfn = torch.compile(fn, dynamic=False)
        with patch(_LAUNCH_KERNEL), patch("subprocess.run"):
            _, source_codes = run_and_get_code(cfn, x, y)

        self.assertTrue(source_codes)
        src = source_codes[0]
        self.assertIn("LoopSpec(", src)
        self.assertIn("sympify('5')", src)

    @patch("torch_spyre._inductor.span_overflow_hint_analysis.MAX_SPAN_BYTES", 8192)
    @config.patch(
        {
            "sencores": 4,
            "chunk_large_tensors": False,
            "unroll_loops": False,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_auto_span_overflow_matches_equivalent_spyre_hint_loop_spec(self):
        from torch_spyre._inductor import spyre_hint

        shape = (1, 20, 16, 64)
        x = torch.randn(shape, dtype=torch.float16).to("spyre")
        y = torch.randn(shape, dtype=torch.float16).to("spyre")

        def auto_fn(x, y):
            return x + y

        def manual_hint_fn(x, y):
            with spyre_hint(num_tiles_per_dim={"SO_H": 5}):
                return x + y

        _pnd.declare_tensor_dim("SO_B", shape[0])
        _pnd.declare_tensor_dim("SO_H", shape[1])
        _pnd.declare_tensor_dim("SO_L", shape[2])
        _pnd.declare_tensor_dim("SO_D", shape[3])
        _pnd.name_tensor_dims(x, ["SO_B", "SO_H", "SO_L", "SO_D"])
        _pnd.name_tensor_dims(y, ["SO_B", "SO_H", "SO_L", "SO_D"])

        with patch(_LAUNCH_KERNEL), patch("subprocess.run"):
            _, auto_sources = run_and_get_code(
                torch.compile(auto_fn, dynamic=False), x, y
            )
            _, manual_sources = run_and_get_code(
                torch.compile(manual_hint_fn, dynamic=False), x, y
            )

        auto_src = auto_sources[0]
        manual_src = manual_sources[0]

        # Automatic span-overflow tiling should lower to the same one-level
        # counted loop shape as the equivalent explicit spyre_hint.
        self.assertEqual(auto_src.count("LoopSpec("), manual_src.count("LoopSpec("))
        self.assertEqual(auto_src.count("sympify('5')"), 1)
        self.assertEqual(manual_src.count("sympify('5')"), 1)
        self.assertIn("sympify('4')", auto_src)
        self.assertIn("sympify('4')", manual_src)
        self.assertIn("op='add'", auto_src)
        self.assertIn("op='add'", manual_src)

