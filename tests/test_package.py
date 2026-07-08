import cvrptw


def test_top_level_api_exports():
    for name in cvrptw.__all__:
        assert getattr(cvrptw, name, None) is not None, name


def test_top_level_import_stays_light():
    """viz/benchmark (and thus matplotlib) must not load on `import cvrptw`."""
    import subprocess
    import sys
    code = 'import sys, cvrptw; sys.exit("matplotlib" in sys.modules)'
    proc = subprocess.run([sys.executable, '-c', code])
    assert proc.returncode == 0
