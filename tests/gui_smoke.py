"""Optional GUI smoke test. Run from a desktop: python tests/gui_smoke.py."""
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import App, ROOT


def wait(app):
    deadline = time.monotonic()+30
    while app.busy and time.monotonic() < deadline:
        app.update()
        time.sleep(.02)
    app.update()
    assert not app.busy, 'Processing timed out'
    assert app.result is not None, 'Processing failed'


def main():
    app = App()
    try:
        app.update()
        app.open_path(ROOT/'images/example_02.png')
        wait(app)
        assert len(app.table.get_children()) == 6
        app.ref_choice.set('1')
        app.ref_size.set('24.26')
        app.calibrate()
        assert app.rows[0]['long'] == 24.26
        for stage in app.result.stages:
            app.stage.set(stage)
            app.refresh_image()
            app.update()
        app.stage.set('Kết quả')
        app.clear_calibration()
        assert app.rows[0]['unit'] == 'px'
        with tempfile.TemporaryDirectory() as directory:
            for suffix, save in [('png', app.save_png), ('zip', app.save_bundle)]:
                path = Path(directory)/('result.'+suffix)
                with patch('app.filedialog.asksaveasfilename', return_value=str(path)):
                    save()
                assert path.is_file() and path.stat().st_size > 0
        app.min_area.set('150')
        assert app.dirty
        app.run()
        wait(app)
        assert not app.dirty and app.calibration is None
        print('GUI smoke passed: select, worker, table, calibration, stages, export, reprocess.')
    finally:
        app.destroy()


if __name__ == '__main__':
    main()
