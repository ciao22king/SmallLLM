"""
Math Solver Plugin for SmallLLM
Solve mathematical problems and add calculation capabilities
"""

import re
import math
from typing import Dict, Any, List, Optional

try:
    import sympy
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

from .base import BasePlugin


class MathSolverPlugin(BasePlugin):
    """Plugin for solving mathematical problems."""

    def __init__(self, **config):
        super().__init__(**config)
        self.display_name = "Math Solver"
        self.description = "Solve math problems and perform calculations"
        self.icon = "➕"
        self.version = "1.0"
        self.author = "SmallLLM"

    def post_generate(self, response: str,
                      context: Dict[str, Any] = None) -> str:
        """Solve math problems in the response."""
        if context is None:
            context = {}

        if self.contains_math(response):
            solved = self.solve_math_in_text(response)
            if solved != response:
                return solved

        return response

    def pre_generate(self, prompt: str,
                     context: Dict[str, Any] = None) -> str:
        """Detect and prepare math problems in prompt."""
        if context is None:
            context = {}

        math_expressions = self.extract_math_expressions(prompt)
        if math_expressions:
            context["math_expressions"] = math_expressions
            return f"{prompt}\n\n[Math mode: solve expressions]"

        return prompt

    def contains_math(self, text: str) -> bool:
        """Check if text contains mathematical expressions."""
        math_patterns = [
            r'\\dfrac', r'\\frac', r'\\sqrt',
            r'\\sum', r'\\int', r'\\lim',
            r'\\log', r'\\ln', r'\\sin',
            r'\\cos', r'\\tan', r'\+',
            r'\-', r'\*', r'\/', r'\=',
            r'\^', r'\(', r'\)', r'\[',
            r'\]', r'\{', r'\}', '=',
            '+', '-', '*', '/', '^', 'sqrt',
            'log', 'ln', 'sin', 'cos', 'tan',
            'integral', 'derivative'
        ]

        return any(pattern in text for pattern in math_patterns)

    def extract_math_expressions(self, text: str) -> List[str]:
        """Extract mathematical expressions from text."""
        expressions = []

        latex_pattern = r'\$\$(.*?)\$\$|\$(.*?)\$'
        matches = re.findall(latex_pattern, text, re.DOTALL)
        for match in matches:
            for expr in match:
                if expr.strip():
                    expressions.append(expr.strip())

        arithmetic_pattern = r'(?:^|\s)(\d+\s*[+\-*/^]\s*\d+)(?:\s|$)'
        matches = re.findall(arithmetic_pattern, text)
        expressions.extend(matches)

        return expressions

    def solve_math_in_text(self, text: str) -> str:
        """Solve mathematical expressions in text."""

        def replace_latex(match):
            expr = match.group(1) or match.group(2)
            if expr:
                try:
                    result = self.solve_math(expr)
                    if result is not None:
                        return f"${result}$"
                except Exception:
                    pass
            return match.group(0)

        text = re.sub(
            r'\$\$(.*?)\$\$|\$(.*?)\$',
            replace_latex, text, flags=re.DOTALL
        )

        def replace_arithmetic(match):
            expr = match.group(1)
            try:
                result = self.solve_math(expr)
                if result is not None:
                    return str(result)
            except Exception:
                pass
            return expr

        text = re.sub(
            r'(\d+\s*[+\-*/^]\s*\d+)',
            replace_arithmetic, text
        )

        return text

    def solve_math(self, expression: str) -> Optional[str]:
        """Solve a mathematical expression."""
        try:
            expr = expression.strip()

            expr = expr.replace('\\frac', '/')
            expr = expr.replace('\\sqrt', 'sqrt')
            expr = expr.replace('\\log', 'log')
            expr = expr.replace('\\ln', 'ln')
            expr = expr.replace('\\sin', 'sin')
            expr = expr.replace('\\cos', 'cos')
            expr = expr.replace('\\tan', 'tan')
            expr = expr.replace('\\pi', 'pi')
            expr = expr.replace('\\cdot', '*')
            expr = expr.replace('\\times', '*')
            expr = expr.replace('\\div', '/')
            expr = expr.replace('\\left', '')
            expr = expr.replace('\\right', '')
            expr = expr.replace('^', '**')

            if SYMPY_AVAILABLE:
                try:
                    result = sympy.simplify(expr)
                    if result != expr:
                        return str(result)
                except Exception:
                    pass

            try:
                expr = re.sub(r'(\d+)(\w+)', r'\1*\2', expr)
                expr = re.sub(r'(\w+)(\d+)', r'\1*\2', expr)
                expr = re.sub(r'(\))(\w+)', r'\1*\2', expr)
                expr = re.sub(r'(\w+)(\()', r'\1*\2', expr)

                result = eval(expr, {'__builtins__': None}, {
                    'sqrt': math.sqrt,
                    'log': math.log10,
                    'ln': math.log,
                    'sin': math.sin,
                    'cos': math.cos,
                    'tan': math.tan,
                    'pi': math.pi,
                    'e': math.e,
                    'pow': math.pow,
                    'exp': math.exp,
                })
                return str(result)
            except Exception:
                pass

        except Exception:
            pass

        return None

    def calculate(self, expression: str) -> Dict[str, Any]:
        """Calculate a mathematical expression."""
        result = self.solve_math(expression)

        if result is not None:
            return {
                "expression": expression,
                "result": result,
                "success": True
            }
        else:
            return {
                "expression": expression,
                "result": None,
                "error": "Could not solve expression",
                "success": False
            }

    def solve_equation(self, equation: str,
                       variable: str = 'x') -> Dict[str, Any]:
        """Solve an equation for a variable."""
        try:
            eq = equation.replace('\\frac', '/')
            eq = eq.replace('\\sqrt', 'sqrt')
            eq = eq.replace('^', '**')

            expr = sympy.Eq(
                sympy.sympify(eq.split('=')[0]),
                sympy.sympify(eq.split('=')[1])
            )

            solutions = sympy.solve(expr, sympy.Symbol(variable))

            return {
                "equation": equation,
                "variable": variable,
                "solutions": [str(s) for s in solutions],
                "success": True
            }
        except Exception as e:
            return {
                "equation": equation,
                "variable": variable,
                "error": str(e),
                "success": False
            }

    def get_actions(self) -> List[Dict[str, Any]]:
        """Get available actions."""
        return [
            {
                "name": "calculate",
                "display_name": "Calculate",
                "description": "Calculate a mathematical expression",
                "icon": "➕"
            },
            {
                "name": "solve_equation",
                "display_name": "Solve Equation",
                "description": "Solve an equation for a variable",
                "icon": "≠"
            },
            {
                "name": "simplify",
                "display_name": "Simplify Expression",
                "description": "Simplify a mathematical expression",
                "icon": "⟿"
            },
            {
                "name": "plot",
                "display_name": "Plot Function",
                "description": "Plot a function",
                "icon": "📊"
            }
        ]

    def execute_action(self, action: str, **kwargs) -> Any:
        """Execute a plugin action."""
        if action == "calculate":
            expression = kwargs.get("expression", "")
            return self.calculate(expression)
        elif action == "solve_equation":
            equation = kwargs.get("equation", "")
            variable = kwargs.get("variable", "x")
            return self.solve_equation(equation, variable)
        elif action == "simplify":
            expression = kwargs.get("expression", "")
            try:
                result = sympy.simplify(expression)
                return {
                    "expression": expression,
                    "simplified": str(result),
                    "success": True
                }
            except Exception as e:
                return {
                    "expression": expression,
                    "error": str(e),
                    "success": False
                }
        elif action == "plot":
            expression = kwargs.get("expression", "")
            try:
                import matplotlib.pyplot as plt
                x = sympy.Symbol('x')
                expr = sympy.sympify(expression)
                f = sympy.lambdify(x, expr, 'numpy')

                import numpy as np
                x_vals = np.linspace(-10, 10, 100)
                y_vals = f(x_vals)

                plt.figure()
                plt.plot(x_vals, y_vals)
                plt.title(f"Plot of {expression}")
                plt.xlabel("x")
                plt.ylabel("f(x)")
                plt.grid(True)
                plt.savefig("/tmp/math_plot.png")
                plt.close()

                return {
                    "expression": expression,
                    "plot_path": "/tmp/math_plot.png",
                    "success": True
                }
            except Exception as e:
                return {
                    "expression": expression,
                    "error": str(e),
                    "success": False
                }

        return None

    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema."""
        return {
            "auto_solve": {
                "type": "bool",
                "default": True,
                "description": "Automatically solve math expressions"
            },
            "precision": {
                "type": "int",
                "default": 10,
                "min": 1,
                "max": 20,
                "description": "Number of decimal places"
            }
        }
