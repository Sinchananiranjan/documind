"""Generic safe AST-based math and formula evaluation engine for numerical and algebraic queries."""

import re
import ast
import math
import operator
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Allowed operators for AST evaluation
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Allowed math functions
SAFE_FUNCTIONS = {
    "log": math.log,
    "log10": math.log10,
    "sqrt": math.sqrt,
    "pow": math.pow,
    "abs": abs,
    "round": round,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}

def safe_eval_ast(node):
    """Recursively evaluate an AST math expression node safely."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.BinOp):
        left = safe_eval_ast(node.left)
        right = safe_eval_ast(node.right)
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            return SAFE_OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported binary operator: {op_type}")
    elif isinstance(node, ast.UnaryOp):
        operand = safe_eval_ast(node.operand)
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            return SAFE_OPERATORS[op_type](operand)
        raise ValueError(f"Unsupported unary operator: {op_type}")
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in SAFE_FUNCTIONS:
            args = [safe_eval_ast(arg) for arg in node.args]
            return SAFE_FUNCTIONS[node.func.id](*args)
        raise ValueError(f"Unsupported function call: {ast.dump(node.func)}")
    else:
        raise ValueError(f"Unsupported AST node: {type(node)}")

def safe_evaluate_math_string(expr_str: str) -> Optional[float]:
    """Parse and evaluate math string safely using AST."""
    try:
        clean_expr = expr_str.replace("^", "**").replace("×", "*").replace("÷", "/")
        parsed = ast.parse(clean_expr, mode='eval')
        res = safe_eval_ast(parsed.body)
        if isinstance(res, (int, float)) and not math.isnan(res) and not math.isinf(res):
            return float(res)
        return None
    except Exception as e:
        logger.debug(f"AST math evaluation skipped for '{expr_str}': {e}")
        return None

def extract_numbers_from_text(text: str) -> List[float]:
    """Extract standalone numbers from text."""
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    numbers = []
    for m in matches:
        try:
            numbers.append(float(m))
        except ValueError:
            pass
    return numbers

def extract_variable_assignments(text: str) -> Dict[str, float]:
    """
    Extract explicit variable assignments like:
    x1 = 5, x2 = 3
    x1: 5, x2: 3
    x = 4.5, y = 2
    a = 10, b = 20
    """
    assignments = {}
    matches = re.findall(r'\b([a-zA-Z][a-zA-Z0-9_]*)\s*[:=]\s*([-+]?\d*\.?\d+)\b', text)
    reserved = {"max", "min", "sub", "sum", "for", "if", "at", "find", "val", "let", "given", "the", "page", "table"}
    for var, val in matches:
        if var.lower() not in reserved:
            try:
                assignments[var] = float(val)
            except ValueError:
                pass
    return assignments


def extract_tuples_or_points(text: str) -> List[List[float]]:
    """
    Extract coordinate tuples like (4.5, 2) or (8, 4) or (6, 21).
    """
    tuple_matches = re.findall(r'\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)(?:\s*,\s*([-+]?\d*\.?\d+))?\s*\)', text)
    points = []
    for match in tuple_matches:
        pt = []
        for item in match:
            if item:
                try:
                    pt.append(float(item))
                except ValueError:
                    pass
        if pt:
            points.append(pt)
    return points

def insert_implicit_multiplications(expr: str) -> str:
    """
    Convert implicit multiplication syntax to explicit '*' multiplication:
    2x1 -> 2*x1
    4(4.5) -> 4*(4.5)
    (4)(2) -> (4)*(2)
    80x1 -> 80*x1
    10x1 -> 10*x1
    3x2 -> 3*x2
    4x2 -> 4*x2
    2x -> 2*x
    """
    # Number immediately followed by variable name (e.g. 2x1 -> 2*x1, 80x1 -> 80*x1, 10x1 -> 10*x1)
    expr = re.sub(r'(\d+(?:\.\d+)?)\s*([a-zA-Z][a-zA-Z0-9_]*)', r'\1*\2', expr)
    # Number immediately followed by opening parenthesis (e.g. 4(4.5) -> 4*(4.5))
    expr = re.sub(r'(\d+(?:\.\d+)?)\s*\(', r'\1*(', expr)
    # Closing parenthesis followed by opening parenthesis (e.g. (2)(3) -> (2)*(3))
    expr = re.sub(r'\)\s*\(', r')*(', expr)
    # Closing parenthesis followed by variable or number (e.g. (2)x -> (2)*x)
    expr = re.sub(r'\)\s*([a-zA-Z0-9_]+)', r')*\1', expr)
    # Variable followed by opening parenthesis if not a known math function name
    expr = re.sub(r'(?<![a-zA-Z0-9_])(?!log|sqrt|abs|sin|cos|tan)([a-zA-Z][a-zA-Z0-9_]*)\s*\(', r'\1*(', expr)
    return expr

def extract_formulas(text: str) -> List[Dict[str, Any]]:
    """
    Extract formulas and equations from text.
    Ignores plain variable constant assignments like x1 = 5 or x2 = 3.
    Matches formulas like Z = 2x1 + 3x2 or Max Z = 4x1 + 3x2.
    """
    formulas = []
    # Split text into mathematical clauses
    clauses = re.split(r'[\.,;\n]|(?:\bif\b|\bcalculate\b|\bcompute\b|\bfor\b|\bwhere\b|\bwhen\b|\band\b|\bgiven\b|\bwith\b|\bfind\b)', text, flags=re.IGNORECASE)

    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue

        eq_match = re.search(r'(?:(?:Max|Min)\s+)?([a-zA-Z][a-zA-Z0-9_]*)\s*=\s*(.+)', clause, re.IGNORECASE)
        if eq_match:
            target = eq_match.group(1).strip()
            rhs = eq_match.group(2).strip().rstrip('.,;:')

            # Skip if RHS is just a plain constant number (e.g. '5', '3', '4.5')
            if re.match(r'^[-+]?\d*\.?\d+$', rhs):
                continue

            # Convert implicit multiplication on RHS first so variable names (x1, x2) are isolated
            rhs_norm = insert_implicit_multiplications(rhs)

            # Extract variables in RHS
            vars_found = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]*\b', rhs_norm)
            reserved_math = {
                "log", "log10", "sqrt", "abs", "sin", "cos", "tan", "min", "max", "sum", "exp",
                "given", "find", "where", "with", "when", "for", "if", "calculate", "compute",
                "determine", "let"
            }
            vars_filtered = []
            for v in vars_found:
                if v.lower() not in reserved_math and not v.isdigit():
                    if v not in vars_filtered:
                        vars_filtered.append(v)

            # Must contain variables in RHS OR arithmetic operators
            if vars_filtered or any(op in rhs_norm for op in ['+', '-', '*', '/', '^']):
                formulas.append({
                    "target": target,
                    "rhs": rhs_norm,
                    "raw_rhs": rhs,
                    "variables": vars_filtered,
                    "raw": f"{target} = {rhs}"
                })

    return formulas

def evaluate_formula_with_values(formula_info: Dict[str, Any], var_values: Dict[str, float]) -> Optional[Dict[str, Any]]:
    """
    Evaluate a formula given a dictionary of variable values.
    """
    rhs = formula_info["rhs"]
    target = formula_info["target"]
    vars_in_rhs = formula_info["variables"]

    # Verify all variables in RHS have assigned values
    missing = [v for v in vars_in_rhs if v not in var_values]
    if missing:
        return None

    # Replace variables with their numeric values
    # Since rhs was already normalized with explicit '*', variables like x1, x2 are separated by '*'
    expr_with_vals = rhs
    sorted_vars = sorted(vars_in_rhs, key=len, reverse=True)
    for v in sorted_vars:
        val = var_values[v]
        val_str = str(int(val)) if val == int(val) else str(val)
        expr_with_vals = re.sub(r'\b' + re.escape(v) + r'\b', f'({val_str})', expr_with_vals)

    # Safe AST evaluate
    res = safe_evaluate_math_string(expr_with_vals)
    if res is not None:
        sub_str = formula_info.get("raw_rhs", rhs)
        for v in sorted_vars:
            val = var_values[v]
            val_str = str(int(val)) if val == int(val) else str(val)
            sub_str = re.sub(r'\b' + re.escape(v) + r'\b', val_str, sub_str)

        formatted_res = int(res) if res == int(res) else round(res, 4)
        return {
            "value": formatted_res,
            "target": target,
            "formula_rhs": rhs,
            "substitutions": var_values,
            "step_desc": f"{target} = {sub_str} = {formatted_res}"
        }
    return None

def execute_generic_calculation(question: str, context: str) -> Dict[str, Any]:
    """
    Generic algebraic and numerical calculation engine.
    Bypasses LLM generation for deterministic math evaluation.
    """
    q_lower = question.lower()
    full_text = f"{question}\n{context}"

    # 1. Extract formulas from question and context
    formulas = extract_formulas(question)
    if not formulas and context:
        formulas = extract_formulas(context)

    # 2. Extract variable assignments (x1=5, x2=3) from question & context
    var_values = extract_variable_assignments(question)
    if context:
        ctx_assignments = extract_variable_assignments(context)
        for k, v in ctx_assignments.items():
            if k not in var_values:
                var_values[k] = v

    # 3. Extract coordinate points (4.5, 2) from question & context
    points = extract_tuples_or_points(question)
    if not points and context:
        points = extract_tuples_or_points(context)

    # 4. Try formula evaluation
    for form in formulas:
        vars_needed = form["variables"]
        # Case 4a: Explicit variable assignments match formula
        if vars_needed and all(v in var_values for v in vars_needed):
            eval_res = evaluate_formula_with_values(form, var_values)
            if eval_res:
                return {
                    "calculated_value": eval_res["value"],
                    "operation_desc": eval_res["step_desc"],
                    "extracted_numbers": list(var_values.values())
                }

        # Case 4b: Tuple coordinates map to formula variables in order of appearance
        if vars_needed and points:
            for pt in points:
                if len(pt) == len(vars_needed):
                    mapped_values = dict(var_values)
                    for var_name, val in zip(vars_needed, pt):
                        mapped_values[var_name] = val
                    eval_res = evaluate_formula_with_values(form, mapped_values)
                    if eval_res:
                        return {
                            "calculated_value": eval_res["value"],
                            "operation_desc": eval_res["step_desc"],
                            "extracted_numbers": pt
                        }

    # 5. Check Parameter Math (Domain-independent parameter equations)
    k_match = re.search(r'\bk\s*=\s*(\d+)', full_text, re.I)
    r_match = re.search(r'\br\s*=\s*(\d+)', full_text, re.I)
    if k_match and r_match and ("find n" in q_lower or "n =" in q_lower or "codeword" in q_lower):
        k_val = int(k_match.group(1))
        r_val = int(r_match.group(1))
        n_val = k_val + r_val
        return {
            "calculated_value": n_val,
            "operation_desc": f"n = k + r = {k_val} + {r_val} = {n_val}",
            "extracted_numbers": [k_val, r_val]
        }

    n_match = re.search(r'\bn\s*=\s*(\d+)', full_text, re.I)
    if n_match and ("codeword" in q_lower or "possible" in q_lower or "states" in q_lower):
        n_val = int(n_match.group(1))
        ans_val = 2 ** n_val
        return {
            "calculated_value": ans_val,
            "operation_desc": f"2^n = 2^{n_val} = {ans_val} possible codewords",
            "extracted_numbers": [n_val]
        }

    dmin_match = re.search(r'\bd_?min\s*=\s*(\d+)', full_text, re.I)
    if dmin_match and ("detect" in q_lower or "error" in q_lower):
        d_val = int(dmin_match.group(1))
        s_val = d_val - 1
        return {
            "calculated_value": s_val,
            "operation_desc": f"s = dmin - 1 = {d_val} - 1 = {s_val} errors guaranteed to be detected",
            "extracted_numbers": [d_val]
        }

    # 6. Bitwise Hamming Distance calculation
    binary_matches = re.findall(r'\b[01]{2,64}\b', question)
    if len(binary_matches) >= 2 and ("hamming" in q_lower or "distance" in q_lower):
        b1, b2 = binary_matches[0], binary_matches[1]
        max_len = max(len(b1), len(b2))
        b1_pad = b1.zfill(max_len)
        b2_pad = b2.zfill(max_len)
        diff_count = sum(1 for c1, c2 in zip(b1_pad, b2_pad) if c1 != c2)
        return {
            "calculated_value": diff_count,
            "operation_desc": f"Hamming distance d({b1}, {b2}) = {diff_count} differing bit positions",
            "extracted_numbers": [diff_count]
        }

    # Direct arithmetic evaluation (e.g. "52 + 53", "125 × 8", "125 x 8", "500 ÷ 25", "100 - 35", "150 * 4")
    q_norm = question.replace("×", "*").replace("÷", "/").replace("^", "**")
    q_norm_clean = re.sub(r'(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)', r'\1 * \2', q_norm)
    arith_match = re.search(r'\(?\s*\d+(?:\.\d+)?\s*\)?(?:\s*[\+\-\*\/\%\*\*]\s*\(?\s*\d+(?:\.\d+)?\s*\)?)+', q_norm_clean)
    if arith_match and not formulas:
        expr_str = arith_match.group(0).strip()
        res = safe_evaluate_math_string(expr_str)
        if res is not None:
            formatted_res = int(res) if res == int(res) else round(res, 4)
            return {
                "calculated_value": formatted_res,
                "operation_desc": f"{expr_str} = {formatted_res}",
                "extracted_numbers": extract_numbers_from_text(expr_str)
            }

    # 8. Verbal operations (percentage increase, average, sum, difference)
    numbers_in_q = extract_numbers_from_text(question)
    numbers_in_ctx = extract_numbers_from_text(context)
    nums = numbers_in_q or numbers_in_ctx

    has_formula_or_eq = bool(formulas or re.search(r'[a-zA-Z0-9_]+\s*=\s*', question))
    if has_formula_or_eq:
        return {
            "calculated_value": None,
            "operation_desc": "",
            "extracted_numbers": nums
        }

    if ("percent" in q_lower or "%" in q_lower or "increase" in q_lower) and len(nums) >= 2:
        old_v, new_v = nums[0], nums[1]
        if old_v != 0:
            pct = round(((new_v - old_v) / abs(old_v)) * 100, 2)
            return {
                "calculated_value": pct,
                "operation_desc": f"(({new_v} - {old_v}) / {old_v}) * 100 = {pct}%",
                "extracted_numbers": [old_v, new_v]
            }

    if ("average" in q_lower or "mean" in q_lower) and nums:
        avg_val = round(sum(nums) / len(nums), 3)
        return {
            "calculated_value": avg_val,
            "operation_desc": f"sum({nums}) / {len(nums)} = {avg_val}",
            "extracted_numbers": nums
        }

    if ("sum" in q_lower or "total" in q_lower or "add" in q_lower) and len(nums) >= 2:
        sum_val = round(sum(nums), 4)
        return {
            "calculated_value": sum_val,
            "operation_desc": f"sum({nums}) = {sum_val}",
            "extracted_numbers": nums
        }

    if ("how many times" in q_lower or "ratio" in q_lower or "times as" in q_lower or "times longer" in q_lower or "times larger" in q_lower or "times greater" in q_lower) and len(nums) >= 2:
        val1, val2 = nums[0], nums[1]
        if val2 != 0:
            ratio_val = round(val1 / val2, 4)
            formatted_ratio = int(ratio_val) if ratio_val == int(ratio_val) else ratio_val
            return {
                "calculated_value": formatted_ratio,
                "operation_desc": f"{val1} / {val2} = {formatted_ratio} times",
                "extracted_numbers": [val1, val2]
            }

    if ("difference" in q_lower or "minus" in q_lower or "subtract" in q_lower) and len(nums) >= 2:
        diff_val = round(abs(nums[0] - nums[1]), 4)
        return {
            "calculated_value": diff_val,
            "operation_desc": f"|{nums[0]} - {nums[1]}| = {diff_val}",
            "extracted_numbers": nums
        }

    return {
        "calculated_value": None,
        "operation_desc": "",
        "extracted_numbers": nums
    }
