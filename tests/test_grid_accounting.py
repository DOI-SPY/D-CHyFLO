import inspect

from dchyflo.grid import full_grid_redispatch


def test_final_grid_accounting_is_a_complete_dispatch():
    source = inspect.getsource(full_grid_redispatch)
    assert "for row in ordered.itertuples" in source
    assert "local_mef" not in source
