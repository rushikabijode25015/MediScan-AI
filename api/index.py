import os
import sys

# Add root folder to sys.path so app imports function correctly on Vercel environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
