from tinygrad.codegen.uops import UOp, UOps, END_FOR_UOP, type_verify
from tinygrad.dtype import dtypes, DType, PtrDType, ImageDType
from typing import Iterator, Optional, Tuple, Any, Dict, List, DefaultDict, Set, Callable, Union, cast, TYPE_CHECKING
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps, ReduceOps, exec_alu

def p_uop_tree(uop: UOp, indent: str = "", is_last: bool = True) -> None:
    # Print the current UOp
    branch = "└── " if is_last else "├── "
    print(f"{indent}{branch}{uop}")

    # Prepare the indentation for children
    child_indent = indent + ("    " if is_last else "│   ")

    # Recursively print children (src UOps)
    for i, child in enumerate(uop.src):
        p_uop_tree(child, child_indent, i == len(uop.src) - 1)

def pp_uop_graph(root: UOp) -> None:
    print("UOp Graph:")
    p_uop_tree(root)
