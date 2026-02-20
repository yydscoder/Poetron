# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Poetron
Builds a standalone executable for Windows
"""

block_cipher = None

a = Analysis(
    ['interactive_haiku.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('src', 'src'),
        ('models', 'models'),
    ],
    hiddenimports=[
        'torch',
        'transformers',
        'tokenizers',
        'huggingface_hub',
        'click',
        'peft',
        'torch.distributed',
        'torch.nn',
        'torch.optim',
        'transformers.models.gpt_neo',
        'transformers.models.gpt_neo.modeling_gpt_neo',
        'transformers.models.gpt_neo.configuration_gpt_neo',
        'transformers.generation',
        'transformers.generation.utils',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'numpy.testing',
        'PIL',
        'scipy',
        'tkinter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Poetron',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if you create one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Poetron',
)
