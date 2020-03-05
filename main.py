import sys

from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QTreeView, QHBoxLayout, QVBoxLayout, QLabel
from PyQt5.QtWidgets import QFileSystemModel, QMenuBar, QMenu, QMainWindow, QPushButton, QAction, qApp
from PyQt5.QtWidgets import QTextEdit, QSizePolicy, QGridLayout, QStyle, QFrame, QErrorMessage, QCheckBox
from PyQt5.QtWidgets import QListWidget, QListWidgetItem
from PyQt5.QtWidgets import QLineEdit, QSpinBox, QDoubleSpinBox, QMessageBox, QDockWidget
from PyQt5.QtGui import QIcon, QPixmap, QImage
import PyQt5.QtGui as QtGui
from PyQt5.QtCore import Qt, QSize, QEvent, QPoint, QUrl
from PyQt5.Qt import pyqtSignal, pyqtSlot, QObject
# Класс QQuickView предоставляет возможность отображать QML файлы.
from PyQt5.QtQuick import QQuickView
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtQml import QQmlApplicationEngine
import sys
from dataclasses import dataclass
import os.path
import zipfile
import tempfile
import datetime
import uuid
import shutil
# from object_detection import train
import create_tf_record

from image_set_editor import ImageSetWindow
from image_set_editor import ImageSet


# Класс типа нейронной сети
@dataclass
class TypeOfNetwork:
    name: str = ""
    settingsFolder: str = ""


# Класс "Распознающая нейросеть
@dataclass
class ConvolutionalNeuralNetwork:
    imageSet: ImageSet = ImageSet()
    filePath: str = ""
    educated: bool = False
    typeOfNetwork: TypeOfNetwork = TypeOfNetwork()

    def init_test(self):
        self.imageSet.filePath = "/home/krasnov/IRZProjects/TEST5.ims"

# Главное окно
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.imgSetEditForm = ImageSetWindow()
        self.neuralNet = ConvolutionalNeuralNetwork()
        self.EXTRACT_TEMP_FOLDER = os.path.join(tempfile.gettempdir(), "NeuroDetector")
        self.EXTRACT_TEMP_SUBFOLDER = ""
        self.modified = False
        self.init_ui()
        self.new_file()

    def init_ui(self):
        # Основное меню
        menuBar = self.menuBar()
        # Меню "Файл"
        fileMenu = menuBar.addMenu("&Файл")
        fileMenuANew = QAction("&Новый", self)
        fileMenuANew.setShortcut("Ctrl+N")
        fileMenuANew.setStatusTip("Новая нейросеть")
        fileMenuANew.triggered.connect(self.new_file)
        fileMenu.addAction(fileMenuANew)
        fileMenu.addSeparator()
        fileMenuAOpen = QAction("&Открыть...", self)
        fileMenuAOpen.setShortcut("Ctrl+O")
        fileMenuAOpen.setStatusTip("Открыть существующую нейросеть")
        fileMenuAOpen.triggered.connect(self.open_file)
        fileMenu.addAction(fileMenuAOpen)
        fileMenu.addSeparator()
        fileMenuASave = fileMenu.addAction("&Сохранить")
        fileMenuASave.setShortcut("Ctrl+S")
        fileMenuASave.setStatusTip("Сохранить изменения")
        fileMenuASave.triggered.connect(self.save_file)
        fileMenuASaveAss = fileMenu.addAction("Сохранить как...")
        fileMenuASaveAss.setShortcut("Ctrl+Shift+S")
        fileMenuASaveAss.setStatusTip("Сохранить текущую нейросеть в другом файле...")
        # fileMenuASaveAss.triggered.connect(self.saveFileAss)
        fileMenu.addSeparator()
        fileMenuAExit = QAction("&Выйти", self)
        fileMenuAExit.setShortcut("Ctrl+Q")
        fileMenuAExit.setStatusTip("Закрыть приложение")
        fileMenuAExit.triggered.connect(self.close)
        fileMenu.addAction(fileMenuAExit)
        menuBar.addMenu(fileMenu)

        # Меню "Набор картинок"
        imgSetMenu = menuBar.addMenu("Набор &картинок")
        imgSetMenuAEdit = imgSetMenu.addAction("Смотреть/&Редактировать...")
        imgSetMenuAEdit.setStatusTip("Просмотр и редактирование наборов картинок")
        imgSetMenuAEdit.triggered.connect(self.img_set_edit_form_open)
        imgSetMenuAOpen = imgSetMenu.addAction("&Выбрать набор...")
        imgSetMenuAOpen.setStatusTip("Выбрать набор картинок для обучения")
        # imgSetMenuAOpen.triggered.connect(self.imgSetOpen)
        menuBar.addMenu(imgSetMenu)

        self.setWindowTitle('NeuroDetector v0.0.0')
        # Центральные элементы, включая изображение
        mainWidget = QWidget(self)
        centralLayout = QVBoxLayout()
        mainWidget.setLayout(centralLayout)
        self.setCentralWidget(mainWidget)

        self.statusBar().setStatusTip("Ready")

        self.resize(1280, 720)
        self.move(300, 300)
        self.setMinimumSize(800, 600)

        self.show()

    def create_and_clear_temp_folder(self):
        if not os.path.exists(self.EXTRACT_TEMP_FOLDER):
            os.mkdir(self.EXTRACT_TEMP_FOLDER)
        for folder in os.listdir(self.EXTRACT_TEMP_FOLDER):
            if os.path.isdir(os.path.join(self.EXTRACT_TEMP_FOLDER, folder)):
                settFile = os.path.join(self.EXTRACT_TEMP_FOLDER, folder, "settings.xml")
                if os.path.exists(settFile):
                    createDT = datetime.datetime.fromtimestamp(os.path.getctime(settFile))
                    if (datetime.datetime.now() - createDT).total_seconds() > 3000.0:
                        shutil.rmtree(os.path.join(self.EXTRACT_TEMP_FOLDER, folder))

    def img_set_edit_form_open(self):
        self.imgSetEditForm.show()

    def new_file(self):
        self.create_and_clear_temp_folder()
        self.EXTRACT_TEMP_SUBFOLDER = os.path.join(self.EXTRACT_TEMP_FOLDER, str(uuid.uuid4()))
        os.mkdir(self.EXTRACT_TEMP_SUBFOLDER)

    def open_file(self):
        if self.neuralNet.filePath and self.modified:
            mBox = QMessageBox()
            dlgResult = mBox.question(self,
                                      "Confirm Dialog",
                                      "Есть несохраненные изменения в текущем файле. Хотите сперва их сохранить?",
                                      QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Yes)
            if dlgResult == QMessageBox.Yes:
                self.saveFile()
            elif dlgResult == QMessageBox.Cancel:
                return

        openDialog = QFileDialog()
        file = openDialog.getOpenFileName(self,
                                          "Выберите файл изображения",
                                          "",
                                          "All files (*.*);;Neural network detectors (*.nnd)",
                                          "Neural network detectors (*.nnd)",
                                          options=openDialog.options() | QFileDialog.DontUseNativeDialog)
        if file[0]:
            self.new_file()
            archive = zipfile.PyZipFile(file[0])
            archive.extractall(self.EXTRACT_TEMP_SUBFOLDER)
            sumSize = 0
            for f in os.listdir(self.EXTRACT_TEMP_SUBFOLDER):
                if os.path.isfile(os.path.join(self.EXTRACT_TEMP_SUBFOLDER, f)):
                    sumSize += os.path.getsize(os.path.join(self.EXTRACT_TEMP_SUBFOLDER, f))

    def save_file_ass(self):
        self.saveFile(True)

    def save_file(self, save_dlg=False):
        if not self.EXTRACT_TEMP_SUBFOLDER:
            return

        if save_dlg or not self.fileName:
            fileDlg = QFileDialog()
            file = fileDlg.getSaveFileName(self,
                                           "Выберите место сохранения файла",
                                           "/",
                                           "All files (*.*);;Neural network detectors (*.nnd)",
                                           "Neural network detectors (*.nnd)")
            if len(file[0]) > 0:
                ext = os.path.splitext(file[0])
                if ext[1] == ".nnd":
                    self.neuralNet.filePath = file[0]
                else:
                    self.neuralNet.filePath = ext[0] + ".nnd"
                if os.exists(self.fileName):
                    mBox = QMessageBox
                    dlgResult = mBox.question(self,
                                              "Confirm Dialog",
                                              "Файл уже существует. Хотите его перезаписать? Это удалит данные в нем",
                                              QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                    if dlgResult == QMessageBox.No:
                        return

            else:
                return

        # if self.savedData.saveToFileXML(os.path.join(self.EXTRACT_TEMP_SUBFOLDER, "settings.xml")):
        #     self.savedData.folder = self.EXTRACT_TEMP_SUBFOLDER
        # else:
        #     self.errDlg = QErrorMessage()
        #     self.errDlg.setWindowTitle("Ошибка")
        #     self.errDlg.showMessage("Произошла непредвиденная ошибка записи файла!")
        #     return

        z = zipfile.ZipFile(self.fileName, 'w')
        for root, dirs, files in os.walk(self.EXTRACT_TEMP_SUBFOLDER):
            for file in files:
                if file:
                    z.write(os.path.join(self.EXTRACT_TEMP_SUBFOLDER, file), file, compress_type=zipfile.ZIP_DEFLATED)
        self.modified = False
        self.setWindowTitle("Micros - " + self.fileName)
        dlgResult = QMessageBox.question(self, "Info Dialog", "Файл сохранен", QMessageBox.Ok, QMessageBox.Ok)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())
