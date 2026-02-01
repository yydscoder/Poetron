#!/usr/bin/env python3
"""
Main entry point for the Poetry Generator CLI
"""

if __name__ == '__main__':
    import sys
    from pathlib import Path
    
    # Add the src directory to the path so we can import our modules
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    
    from cli import cli
    cli()