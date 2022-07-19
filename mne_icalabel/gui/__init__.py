from qtpy.QtWidgets import QApplication

from ._label_components import ICAComponentLabeler


def label_ica_components(inst, ica):
    app = QApplication([])
    window = ICAComponentLabeler(inst=inst, ica=ica)
    window.show()
    app.exec()
    return app
