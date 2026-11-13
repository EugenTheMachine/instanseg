import py_compile
import sys

try:
    py_compile.compile('instanseg/utils/models/InstanSeg_UNet.py', doraise=True)
    print("SUCCESS: Syntax is valid")
    sys.exit(0)
except SyntaxError as e:
    print(f"SYNTAX ERROR: {e}")
    print(f"  File: {e.filename}")
    print(f"  Line: {e.lineno}")
    print(f"  Text: {e.text}")
    sys.exit(1)
