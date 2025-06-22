import sys
from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MplCanvas(FigureCanvas):
    """Matplotlib canvas widget to embed in PySide6 GUI."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)

    def plot(self, data_x, data_y, title="", xlabel="", ylabel="", legend_label=""):
        self.axes.cla() # Clear previous plot
        self.axes.plot(data_x, data_y, label=legend_label)
        self.axes.set_title(title)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        if legend_label:
            self.axes.legend()
        self.axes.grid(True)
        self.draw()

    def plot_multiple_lines(self, x_data_list, y_data_list, labels_list, title="", xlabel="", ylabel=""):
        self.axes.cla()
        for i in range(len(x_data_list)):
            self.axes.plot(x_data_list[i], y_data_list[i], label=labels_list[i])
        self.axes.set_title(title)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        if any(labels_list):
            self.axes.legend()
        self.axes.grid(True)
        self.draw()

    def imshow(self, image_data, cmap='gray'):
        self.axes.cla()
        self.axes.imshow(image_data, cmap=cmap)
        self.axes.axis('off')
        self.draw()


class PlotWidget(QWidget):
    """A simple QWidget that contains a MplCanvas."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def get_canvas(self):
        return self.canvas

if __name__ == '__main__':
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    main_widget = PlotWidget()

    # Example usage:
    # Simple line plot
    # main_widget.get_canvas().plot([0, 1, 2, 3], [1, 3, 2, 4], title="Test Plot", xlabel="X", ylabel="Y", legend_label="Data1")

    # Example imshow
    import numpy as np
    dummy_image = np.random.rand(100,100)
    main_widget.get_canvas().imshow(dummy_image)
    main_widget.get_canvas().axes.set_title("Test Imshow")

    main_widget.show()
    sys.exit(app.exec())
