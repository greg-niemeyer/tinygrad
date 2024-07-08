def get_new_test_name():
    if not hasattr(get_new_test_name, "counter"):
        get_new_test_name.counter = 0
    get_new_test_name.counter += 1
    return f"test_replay_{get_new_test_name.counter}"

def generate_test_script(tests, output_file):
    with open(output_file, 'w') as f:
        f.write("""
from typing import List, Tuple, Dict, Union
import numpy as np
import unittest
from dataclasses import replace
from test.external.fuzz_linearizer import compare_linearizer

from tinygrad.codegen.kernel import Opt, OptOps, KernelOptError
from tinygrad.codegen.linearizer import Linearizer, expand_node, expand_idxs, get_grouped_dims
from tinygrad.codegen.uops import UOp, UOps
from tinygrad.device import Device, Buffer
from tinygrad.ops import BinaryOps, BufferOps, MemBuffer, ConstBuffer, LazyOp, LoadOps, TernaryOps, ReduceOps, UnaryOps
from tinygrad.renderer import TensorCore
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from tinygrad.shape.symbolic import MulNode, Variable, NumNode, Node
from tinygrad.tensor import Tensor, _to_np_dtype
from tinygrad.engine.schedule import create_schedule
from tinygrad.engine.realize import run_schedule, lower_schedule, CompiledRunner
from tinygrad.engine.graph import print_tree
from tinygrad.helpers import DEBUG, prod, Context, getenv, CI
from tinygrad.dtype import DType, dtypes

class ProcessReplays(unittest.TestCase):
""")

        for ast, applied_opts in tests:
            f.write(f"""
    def {get_new_test_name()}(self):
        ast = {ast}
        k = Linearizer(*ast)
        applied_opts = {applied_opts}
        for opt in applied_opts:
            k.apply_opt(opt)
        k.linearize().uops.print()

""")

        f.write("""
if __name__ == '__main__':
    unittest.main()
""")
