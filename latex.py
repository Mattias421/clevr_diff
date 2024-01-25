from clevr_diff.sdxl_pacs import main
grid = main.dora.dir / 'grids' / 'sdxl_pacs'
for child in grid.iterdir():
    xp = main.get_xp_from_sig(child.name)
    xp.link.load()
    # look at xp.delta and xp.link.history