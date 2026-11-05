import copy
import unittest

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

from onnxconverter_common.float16 import convert_float_to_float16


def value(name, dtype=TensorProto.FLOAT, shape=None):
    return helper.make_tensor_value_info(name, dtype, [2] if shape is None else shape)


def branch(label, annotate=False, initializer=False):
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], [label + "_out"], name=label + "_id")],
        label,
        [],
        [value(label + "_out")],
        initializer=[helper.make_tensor("x", TensorProto.FLOAT, [2], [5, 6])]
        if initializer
        else [],
        value_info=[value("x")] if annotate else [],
    )
    return graph


def model(then=None, otherwise=None, direct=False):
    nodes = [
        helper.make_node(
            "If",
            ["cond"],
            ["out"],
            name="choose",
            then_branch=branch("then") if then is None else then,
            else_branch=branch("else") if otherwise is None else otherwise,
        )
    ]
    outputs = [value("out")]
    if direct:
        nodes.append(helper.make_node("Identity", ["x"], ["direct"], name="direct"))
        outputs.append(value("direct"))
    return helper.make_model(
        helper.make_graph(
            nodes,
            "capture",
            [value("cond", TensorProto.BOOL, []), value("x")],
            outputs,
        ),
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=9,
    )


class TestFloat16Captures(unittest.TestCase):
    def assert_inference(self, original, **kwargs):
        onnx.checker.check_model(original)
        converted = convert_float_to_float16(copy.deepcopy(original), **kwargs)
        onnx.checker.check_model(converted)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        baseline = ort.InferenceSession(
            original.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        session = ort.InferenceSession(
            converted.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        for condition in (False, True):
            inputs = {
                "cond": np.array(condition),
                "x": np.array([1.2, 3.4], np.float32),
            }
            expected = baseline.run(None, inputs)
            if not kwargs.get("keep_io_types", False):
                inputs["x"] = inputs["x"].astype(np.float16)
            actual = session.run(None, inputs)
            for want, got in zip(expected, actual):
                np.testing.assert_allclose(got, want, rtol=1e-3)
        return converted

    def test_capture_only(self):
        self.assert_inference(model(), keep_io_types=True)

    def test_capture_without_shape_inference(self):
        self.assert_inference(model(), keep_io_types=True, disable_shape_infer=True)

    def test_explicit_capture_value_info(self):
        self.assert_inference(model(branch("then", annotate=True)), keep_io_types=True)

    def test_direct_and_nested_capture(self):
        nested = helper.make_graph(
            [
                helper.make_node(
                    "If",
                    ["cond"],
                    ["nested_out"],
                    name="nested",
                    then_branch=branch("inner_then"),
                    else_branch=branch("inner_else"),
                )
            ],
            "nested_graph",
            [],
            [value("nested_out")],
        )
        self.assert_inference(model(nested, direct=True), keep_io_types=True)

    def test_blocked_captured_consumer(self):
        self.assert_inference(
            model(branch("then", annotate=True)),
            keep_io_types=True,
            node_block_list=["then_id"],
        )

    def test_blocked_graph_body(self):
        nested = helper.make_graph(
            [
                helper.make_node(
                    "If",
                    ["cond"],
                    ["nested_out"],
                    name="nested",
                    then_branch=branch("inner_then"),
                    else_branch=branch("inner_else"),
                ),
                helper.make_node(
                    "Identity", ["nested_out"], ["then_out"], name="after_nested"
                ),
            ],
            "nested_graph",
            [],
            [value("then_out")],
        )
        converted = self.assert_inference(
            model(nested, direct=True), keep_io_types=True, node_block_list=["nested"]
        )
        outer = next(
            a.g
            for n in converted.graph.node
            if n.name == "choose"
            for a in n.attribute
            if a.name == "then_branch"
        )
        blocked = next(n for n in outer.node if n.name == "nested")
        for attr in blocked.attribute:
            self.assertEqual(attr.g.node[0].input[0], "x")
            self.assertEqual(
                attr.g.output[0].type.tensor_type.elem_type, TensorProto.FLOAT
            )

    def test_local_initializer_shadowing(self):
        converted = self.assert_inference(
            model(branch("then", initializer=True), direct=True), keep_io_types=True
        )
        then = next(
            a.g
            for n in converted.graph.node
            if n.name == "choose"
            for a in n.attribute
            if a.name == "then_branch"
        )
        self.assertNotEqual(then.initializer[0].name, "x")
        self.assertEqual(then.node[0].input[0], then.initializer[0].name)
        self.assertEqual(then.initializer[0].data_type, TensorProto.FLOAT16)
        self.assertEqual(len(then.node), 1)

    def test_nested_initializer_shadowing(self):
        inner_then = branch("inner_then", initializer=True)
        inner_then.initializer[0].float_data[:] = [7, 8]
        nested = helper.make_graph(
            [
                helper.make_node(
                    "If",
                    ["cond"],
                    ["nested_out"],
                    name="nested",
                    then_branch=inner_then,
                    else_branch=branch("inner_else"),
                )
            ],
            "nested_graph",
            [],
            [value("nested_out")],
            initializer=[helper.make_tensor("x", TensorProto.FLOAT, [2], [5, 6])],
        )
        self.assert_inference(
            model(nested, copy.deepcopy(nested), direct=True), keep_io_types=True
        )

    def test_local_input_shadowing(self):
        body = helper.make_graph(
            [
                helper.make_node("Identity", ["condition"], ["condition_out"]),
                helper.make_node("Identity", ["x"], ["carry_out"]),
            ],
            "body",
            [
                value("iteration", TensorProto.INT64, []),
                value("condition", TensorProto.BOOL, []),
                value("x"),
            ],
            [value("condition_out", TensorProto.BOOL, []), value("carry_out")],
        )
        original = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        "Loop", ["trips", "cond", "x"], ["out"], name="loop", body=body
                    ),
                ],
                "loop_graph",
                [value("cond", TensorProto.BOOL, []), value("x")],
                [value("out")],
                initializer=[helper.make_tensor("trips", TensorProto.INT64, [], [2])],
            ),
            opset_imports=[helper.make_opsetid("", 17)],
            ir_version=9,
        )
        converted = self.assert_inference(original, keep_io_types=True)
        body = next(
            a.g for n in converted.graph.node if n.name == "loop" for a in n.attribute
        )
        self.assertNotEqual(body.input[2].name, "x")
        self.assertEqual(body.node[1].input[0], body.input[2].name)

    def test_without_retained_io(self):
        self.assert_inference(model(direct=True), keep_io_types=False)

    def test_capture_cast_name_collision(self):
        then = branch("then")
        then.initializer.append(
            helper.make_tensor("x_capture_cast", TensorProto.FLOAT, [2], [7, 8])
        )
        self.assert_inference(model(then), keep_io_types=True)


if __name__ == "__main__":
    unittest.main()
