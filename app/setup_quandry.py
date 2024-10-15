# Import quandry
import sys
import os
# Add the src directory to the Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../quandry_package/src/'))
if src_path not in sys.path:
    sys.path.append(src_path)
        
import quandry