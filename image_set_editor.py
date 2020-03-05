import xml.etree.ElementTree as xmlET

from typing import List
from PyQt5.QtGui import QImage, QCursor
import PyQt5.QtGui as QtGui
from lxml import etree
from PyQt5.QtWidgets import QMainWindow, QLabel, QListWidget, QMenu, QAction, QWidget, QHBoxLayout, QVBoxLayout, \
    QSizePolicy, QFileDialog, QInputDialog, QMessageBox, QLineEdit
from PyQt5.QtCore import Qt, QEvent, QPoint
import os
import glob
import numpy as np
import cv2

from dataclasses import dataclass, field


# Размеры изображения для аннотаций
@dataclass
class SizeImage:
    width: int = 0
    height: int = 0
    depth: int = 3


@dataclass
class BoundBox:
    xMin: int = 0
    yMin: int = 0
    xMax: int = 0
    yMax: int = 0


# Объект на изображении
@dataclass
class ObjectInImage:
    name: str = ""
    pose: str = "Unspecified"
    truncated: int = 0
    difficult: int = 0
    bndBox: BoundBox = BoundBox()


# Класс "Картинка с аннотацией"
@dataclass
class SingleImage:
    path: str
    filename: str = ""
    folder: str = ""
    size: SizeImage = SizeImage()
    objectsFromImage: List[ObjectInImage] = field(default_factory=list, repr=False)
    segmented: int = 0

    def __post_init__(self):
        if self.path:
            os.path.splitext(self.path)


# Класс "Набор картинок"
@dataclass
class ImageSet:
    # Путь к файлу с описанием набора картинок
    filePath: str = ""
    paths: List[str] = field(default_factory=list, repr=False)
    imgPaths: List[SingleImage] = field(default_factory=dict, repr=False)
    objects: List[str] = field(default_factory=dict, repr=False)

    def load_from_file(self, file_name):
        with open(file_name) as fileObj:
            xml = fileObj.read()
        root = etree.fromstring(xml)

        self.filePath = file_name
        for elementsXML in root.getchildren():
            if elementsXML.tag == "Paths":
                for element in elementsXML.getchildren():
                    self.paths.append(element.text)
                    # self.pathListWidget.addItem(element.text)
            elif elementsXML.tag == "Annotations":
                for element in elementsXML.getchildren():
                    imagePath = element.attrib["path"]
                    self.imgPaths[imagePath] = SingleImage(imagePath)
                    for imgObject in element.getchildren():
                        newObject = ObjectInImage()
                        for param in imgObject.getchildren():
                            if param.tag == "Name":
                                newObject.name = param.text
                            elif param.tag == "Pose":
                                newObject.pose = param.text
                            elif param.tag == "Truncated":
                                newObject.truncated = int(param.text)
                            elif param.tag == "Difficult":
                                newObject.difficult = int(param.text)
                            elif param.tag == "Bndbox":
                                newBoundBox = BoundBox()
                                for coord in param.getchildren():
                                    if coord.tag == "Xmin":
                                        newBoundBox.xmin = int(coord.text)
                                    elif coord.tag == "Ymin":
                                        newBoundBox.ymin = int(coord.text)
                                    elif coord.tag == "Xmax":
                                        newBoundBox.xmax = int(coord.text)
                                    elif coord.tag == "Ymax":
                                        newBoundBox.ymax = int(coord.text)
                                newObject.bndBox = newBoundBox
                        self.imgPaths[imagePath].objectsFromImage.append(newObject)
            elif elementsXML.tag == "Types":
                for element in elementsXML.getchildren():
                    self.objects[element.text] = len(self.objects)
                    # self.objectListWidget.addItem(element.text)


# Окно для работы с наборами изображений
class ImageSetWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.imLabel = QLabel()
        # Список источников
        self.pathListWidget = QListWidget()
        self.menuPathListWidget = QMenu()
        self.actionPathSubFolder = QAction()
        self.actionPathEdit = QAction()
        self.actionPathRemove = QAction()
        # Список картинок
        self.labelImagesListWidget = QLabel()
        self.imagesListWidget = QListWidget()
        # Список объектов
        self.objectListWidget = QListWidget()
        self.menuObjectListWidget = QMenu()
        self.actionObjectEdit = QAction()
        self.actionObjectRemove = QAction()
        # Путь к текущему открытому файлу
        self.fileName = ""
        self.imageSet = ImageSet()
        self.colors = ((0, 0, 255), (0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (125, 125, 125))

        self.init_ui()

    def init_ui(self):
        """Основное меню"""
        menuBar = self.menuBar()
        """Меню Файл"""
        fileMenu = menuBar.addMenu("&Файл")
        fileMenuANew = QAction("&Новый", self)
        fileMenuANew.setShortcut("Ctrl+N")
        fileMenuANew.setStatusTip("Новый набор картинок")
        fileMenuANew.triggered.connect(self.new_file)
        fileMenu.addAction(fileMenuANew)

        fileMenu.addSeparator()
        fileMenuAOpen = QAction("&Открыть...", self)
        fileMenuAOpen.setShortcut("Ctrl+O")
        fileMenuAOpen.setStatusTip("Открыть существующий набор картинок")
        fileMenuAOpen.triggered.connect(self.open_file)
        fileMenu.addAction(fileMenuAOpen)

        fileMenu.addSeparator()
        fileMenuAImport = QAction("&Импорт...", self)
        fileMenuAImport.setShortcut("Ctrl+I")
        fileMenuAImport.setStatusTip("Импортировать набор картинок из набора Tensor Flow")
        fileMenuAImport.triggered.connect(self.import_file)
        fileMenu.addAction(fileMenuAImport)

        fileMenu.addSeparator()
        fileMenuASave = fileMenu.addAction("&Сохранить")
        fileMenuASave.setShortcut("Ctrl+S")
        fileMenuASave.setStatusTip("Сохранить изменения")
        fileMenuASave.triggered.connect(self.save_file)

        fileMenuASaveAss = fileMenu.addAction("Сохранить как...")
        fileMenuASaveAss.setShortcut("Ctrl+Shift+S")
        fileMenuASaveAss.setStatusTip("Сохранить текущий набор картинок в другом файле...")
        fileMenuASaveAss.triggered.connect(self.save_file_ass)

        fileMenuASaveCopy = fileMenu.addAction("Сохранить копию...")
        fileMenuASaveCopy.setStatusTip("Скопировать текущий набор в отдельный файл...")
        fileMenuASaveCopy.triggered.connect(self.save_file_copy)

        fileMenu.addSeparator()
        fileMenuAExit = QAction("&Выйти", self)
        fileMenuAExit.setShortcut("Ctrl+Q")
        fileMenuAExit.setStatusTip("Закрыть приложение")
        fileMenuAExit.triggered.connect(self.close)
        fileMenu.addAction(fileMenuAExit)
        menuBar.addMenu(fileMenu)

        self.setWindowTitle('Редактор наборов изображений')

        """Центральные элементы, включая изображение"""
        mainWidget = QWidget(self)
        centralLayout = QHBoxLayout()
        mainWidget.setLayout(centralLayout)
        self.setCentralWidget(mainWidget)

        leftLayout = QVBoxLayout()
        centralLayout.addLayout(leftLayout)

        labelPathListWidget = QLabel("Источники")
        leftLayout.addWidget(labelPathListWidget)
        self.pathListWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.pathListWidget.customContextMenuRequested.connect(self.show_context_menu_path)
        self.pathListWidget.itemSelectionChanged.connect(self.path_list_widget_item_selected)

        """Контекстное меню источников"""
        actionPathAddFile = self.menuPathListWidget.addAction('Добавить файл/файлы...')
        actionPathAddFile.setShortcut("insert")
        actionPathAddFile.triggered.connect(self.action_path_add_file_click)

        actionPathAddFolder = self.menuPathListWidget.addAction('Добавить папку...')
        actionPathAddFolder.setShortcut("shift+insert")
        actionPathAddFolder.triggered.connect(self.action_path_add_folder_click)

        actionPathAddMask = self.menuPathListWidget.addAction('Добавить маску...')
        actionPathAddMask.setShortcut("ctrl+insert")
        actionPathAddMask.triggered.connect(self.action_path_add_mask_click)

        self.menuPathListWidget.addSeparator()
        self.actionPathEdit.setText("Изменить...")
        self.menuPathListWidget.addAction(self.actionPathEdit)
        self.actionPathEdit.triggered.connect(self.action_path_edit_click)

        self.actionPathSubFolder.setText("Включать подпапки")
        self.actionPathSubFolder.setCheckable(True)
        self.actionPathSubFolder.setChecked(False)
        self.menuPathListWidget.addAction(self.actionPathSubFolder)
        self.actionPathSubFolder.triggered.connect(self.action_path_sub_folder_click)

        self.menuPathListWidget.addSeparator()
        self.actionPathRemove.setText("Удалить")
        self.actionPathRemove.setShortcut("delete")
        self.menuPathListWidget.addAction(self.actionPathRemove)
        self.actionPathRemove.triggered.connect(lambda: self.pathListWidget.takeItem(self.pathListWidget.currentRow()))

        leftLayout.addWidget(self.pathListWidget)
        # for i in range(10):
        #     self.pathListWidget.addItem('Set #{}'.format(i))
        # self.pathListWidget.addItem('/home/krasnov/IRZProjects/NeuroWeb/Object-Detection-Quidditch-master/images')

        # labelImagesListWidget = QLabel("Файлы")
        self.labelImagesListWidget.setText("0 файлов")
        leftLayout.addWidget(self.labelImagesListWidget)
        self.imagesListWidget.itemSelectionChanged.connect(self.image_list_widget_item_selected)
        leftLayout.addWidget(self.imagesListWidget)
        # for i in range(10):
        #     self.imagesListWidget.addItem('File #{}'.format(i))

        labelObjectListWidget = QLabel("Объекты")
        leftLayout.addWidget(labelObjectListWidget)
        leftLayout.addWidget(self.objectListWidget)
        self.objectListWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.objectListWidget.customContextMenuRequested.connect(self.show_context_menu_object)
        # self.objectListWidget.addItem("Автомобиль")
        # self.objectListWidget.addItem("Человек")
        # self.objectListWidget.addItem("Птица")

        """Контекстное меню объектов"""
        actionObjectAdd = self.menuObjectListWidget.addAction('Добавить...')
        actionObjectAdd.setShortcut("insert")
        actionObjectAdd.triggered.connect(self.action_object_add_click)

        self.actionObjectEdit.setText("Изменить...")
        self.menuObjectListWidget.addAction(self.actionObjectEdit)
        self.actionObjectEdit.triggered.connect(self.action_object_edit_click)

        self.actionObjectRemove.setText("Удалить")
        self.actionObjectRemove.setShortcut("delete")
        self.menuObjectListWidget.addAction(self.actionObjectRemove)

        # self.actionObjectRemove.triggered.connect(
        # lambda: self.objectListWidget.takeItem(self.objectListWidget.currentRow()))

        self.actionObjectRemove.triggered.connect(self.action_object_remove_click)

        # Изображение
        self.imLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.imLabel.setStyleSheet("border: 1px solid red")
        centralLayout.addWidget(self.imLabel)
        self.imLabel.installEventFilter(self)

        self.resize(800, 600)
        self.move(300, 300)
        self.setMinimumSize(800, 600)

        # Обработчики событий формы и ее компонентов

    def eventFilter(self, obj, event):
         if obj is self.imLabel:
             if event.type() == QEvent.Resize:
                 self.image_list_widget_item_selected()
         return QMainWindow.eventFilter(self, obj, event)

    def new_file(self):
        self.imageSet = ImageSet()
        self.pathListWidget.clear()
        self.objectListWidget.clear()
        self.imagesListWidget.clear()
        self.imLabel.clear()

    def open_file(self):
        fileDialog = QFileDialog()
        file = QFileDialog.getOpenFileName(fileDialog,
                                           "Выберите место сохранения файла",
                                           "",
                                           "All files (*.*);;Наборы изображений (*.ims)",
                                           "Наборы изображений (*.ims)",
                                           options=fileDialog.options() | QFileDialog.DontUseNativeDialog)
        if not file[0]:
            return

        self.new_file()
        # Загружаем данные тз нашего файла в xml формате
        self.imageSet.load_from_file(file[0])

        for path in self.imageSet.paths:
            self.pathListWidget.addItem(path)
        for obj in self.imageSet.objects.keys():
            self.objectListWidget.addItem(obj)

        self.fileName = file[0]

    # Импорт набора картинок из набора Tensor Flow
    def import_file(self):
        self.new_file()
        imgDirectory = self.get_folder_images()
        if not imgDirectory:
            return

        """fileDialog = QFileDialog()
        file = QFileDialog.getOpenFileName(fileDialog,
                                           "Укажите файл с метками (список объектов)",
                                           "",
                                           "All files (*.*);;Файлы с метками (*.pbtxt)",
                                           "Файлы с метками (*.pbtxt)",
                                           options=fileDialog.options() | QFileDialog.DontUseNativeDialog)
        if not file[0]:
            return

        with open(file[0]) as fileObj:
            xml = fileObj.read()
            """

        self.pathListWidget.addItem(imgDirectory)
        annDirectory = self.get_folder_images("", "", caption="Выберите папку с аннотациями")
        if annDirectory:
            if os.path.exists(os.path.join(annDirectory, "xmls")):
                annDirectory = os.path.join(annDirectory, "xmls")
            # Получаем файлы в папке анотаций по маске "xml", "xmls"
            annFiles = glob.glob(os.path.join(annDirectory, "*.xml*"))
            annFiles.sort()
            for annFile in annFiles:
                with open(annFile) as fileObj:
                    xml = fileObj.read()
                root = etree.fromstring(xml)
                imagePath = ""
                for elementsXML in root.getchildren():
                    if elementsXML.tag == "filename":
                        imagePath = os.path.join(imgDirectory, elementsXML.text)
                        self.imageSet.imgPaths[imagePath] = SingleImage(imagePath)
                        break
                if imagePath:
                    for elementsXML in root.getchildren():
                        if elementsXML.tag == "folder":
                            self.imageSet.imgPaths[imagePath].folder = elementsXML.text
                        elif elementsXML.tag == "filename":
                            self.imageSet.imgPaths[imagePath].filename = elementsXML.text
                        elif elementsXML.tag == "size":
                            for sizeParamXML in elementsXML.getchildren():
                                if sizeParamXML.tag == "width":
                                    self.imageSet.imgPaths[imagePath].size.width = int(sizeParamXML.text)
                                elif sizeParamXML.tag == "height":
                                    self.imageSet.imgPaths[imagePath].size.height = int(sizeParamXML.text)
                                elif sizeParamXML.tag == "depth":
                                    self.imageSet.imgPaths[imagePath].size.depth = int(sizeParamXML.text)
                        elif elementsXML.tag == "segmented":
                            self.imageSet.imgPaths[imagePath].segmented = int(elementsXML.text)
                        elif elementsXML.tag == "object":
                            newObject = ObjectInImage()
                            for param in elementsXML.getchildren():
                                if param.tag == "name":
                                    newObject.name = param.text
                                elif param.tag == "pose":
                                    newObject.pose = param.text
                                elif param.tag == "truncated":
                                    newObject.truncated = int(param.text)
                                elif param.tag == "Difficult":
                                    newObject.difficult = int(param.text)
                                elif param.tag == "bndbox":
                                    newBoundBox = BoundBox()
                                    for coord in param.getchildren():
                                        if coord.tag == "xmin":
                                            newBoundBox.xmin = int(coord.text)
                                        elif coord.tag == "ymin":
                                            newBoundBox.ymin = int(coord.text)
                                        elif coord.tag == "xmax":
                                            newBoundBox.xmax = int(coord.text)
                                        elif coord.tag == "ymax":
                                            newBoundBox.ymax = int(coord.text)
                                    newObject.bndBox = newBoundBox
                            if self.imageSet.objects.get(newObject.name, -1) == -1: # newObject.name not in self.imageSet.objects:
                                self.imageSet.objects[newObject.name] = len(self.imageSet.objects)
                                self.objectListWidget.addItem(newObject.name)
                            self.imageSet.imgPaths[imagePath].objectsFromImage.append(newObject)

        if not annDirectory:
            return

    # Сохранение файла с набором изображений
    def save_file(self, save_dlg=True):
        if self.pathListWidget.count() == 0:
            return
        if save_dlg or not self.fileName:
            fileDialog = QFileDialog()
            file = QFileDialog.getSaveFileName(fileDialog,
                                               "Выберите место сохранения файла",
                                               "",
                                               "All files (*.*);;Наборы изображений (*.ims)",
                                               "Наборы изображений (*.ims)",
                                               options=fileDialog.options() | QFileDialog.DontUseNativeDialog)
            if file[0]:
                ext = os.path.splitext(file[0])
                if ext[1] == ".ims":
                    self.fileName = file[0]
                else:
                    self.fileName = ext[0] + ".ims"
                if os.path.exists(self.fileName):
                    messageBox = QMessageBox()
                    dlgResult = QMessageBox.question(messageBox,
                                                     "Confirm Dialog",
                                                     "Файл уже существует. Хотите его перезаписать?" +
                                                     "Это удалит данные в нем",
                                                     QMessageBox.Yes | QMessageBox.No,
                                                     QMessageBox.No)
                    if dlgResult == QMessageBox.No:
                        return

            else:
                return

        # Записываем данные в наш файл в xml формате
        if self.fileName:
            root = xmlET.Element("ImageSet")

            pathsElement = xmlET.Element("Paths")
            root.append(pathsElement)
            for i in range(self.pathListWidget.count()):
                pathElement = xmlET.SubElement(pathsElement, "Path")
                pathElement.text = self.pathListWidget.item(i).text()

            annotationsElement = xmlET.Element("Annotations")
            root.append(annotationsElement)
            for key, value in self.imageSet.imgPaths.items():
                fileElement = xmlET.SubElement(annotationsElement, "File")
                fileElement.set("path", key)
                for obj in value.objectsFromImage:
                    objElement = xmlET.SubElement(fileElement, "Object")
                    objName = xmlET.SubElement(objElement, "Name")
                    objName.text = obj.name
                    objPose = xmlET.SubElement(objElement, "Pose")
                    objPose.text = obj.pose
                    objTruncated = xmlET.SubElement(objElement, "Truncated")
                    objTruncated.text = str(obj.truncated)
                    objDifficult = xmlET.SubElement(objElement, "Difficult")
                    objDifficult.text = str(obj.difficult)
                    objBndBox = xmlET.SubElement(objElement, "Bndbox")
                    objXMin = xmlET.SubElement(objBndBox, "Xmin")
                    objXMin.text = str(obj.bndBox.xmin)
                    objYMin = xmlET.SubElement(objBndBox, "Ymin")
                    objYMin.text = str(obj.bndBox.ymin)
                    objXMax = xmlET.SubElement(objBndBox, "Xmax")
                    objXMax.text = str(obj.bndBox.xmax)
                    objYMax = xmlET.SubElement(objBndBox, "Ymax")
                    objYMax.text = str(obj.bndBox.ymax)

            pathsElement = xmlET.Element("Types")
            root.append(pathsElement)
            for obj in self.imageSet.objects.keys():
                pathElement = xmlET.SubElement(pathsElement, "Type")
                pathElement.text = obj

            tree = xmlET.ElementTree(root)
            with open(self.fileName, "w"):
                tree.write(self.fileName)

    def save_file_ass(self):
        self.save_file(True)
        return

    def save_file_copy(self):
        currentFileName = self.fileName
        self.save_file(True)
        self.fileName = currentFileName
        return

    # Отображение контекстного меню путей
    def show_context_menu_path(self, point):
        if self.pathListWidget.currentItem():
            self.actionPathEdit.setEnabled(True)
            self.actionPathRemove.setEnabled(True)
            path = self.pathListWidget.currentItem().text()
            withSubFolders = False
            if str.endswith(path, "**"):
                path = path[0:-2]
                withSubFolders = True

            if os.path.exists(path) and os.path.isdir(path):
                self.actionPathSubFolder.setEnabled(True)
                self.actionPathSubFolder.setChecked(withSubFolders)
            else:
                self.actionPathSubFolder.setEnabled(False)
                self.actionPathSubFolder.setChecked(False)
        else:
            self.actionPathSubFolder.setEnabled(False)
            self.actionPathSubFolder.setChecked(False)
            self.actionPathEdit.setEnabled(False)
            self.actionPathRemove.setEnabled(False)
        # self.menuPathListWidget.exec(self.mapToGlobal(point))
        self.menuPathListWidget.exec(QCursor.pos())

    def get_files_images(self, init_dir="", multi_select=True):
        openDialog = QFileDialog()
        getFile = openDialog.getOpenFileNames
        if not multi_select:
            getFile = openDialog.getOpenFileName
        files = getFile(self,
                        "Выберите файлы изображения",
                        init_dir,
                        "JPEG (*.jpg);;All files (*)",
                        "JPEG (*.jpg)",
                        options=openDialog.options() | QFileDialog.DontUseNativeDialog)
        return files[0]

    def action_path_add_file_click(self):
        for file in self.get_files_images("", True):
            if file:
                self.pathListWidget.addItem(file)

    def get_folder_images(self, init_dir="", check_file_mask="*.jpg", caption="Выберите папку с изображениями",
                          answer="В указанной папке не обнаружено файлов изображений. Все равно добавить ее?"):
        openDialog = QFileDialog()
        directory = openDialog.getExistingDirectory(self,
                                                    caption=caption,
                                                    directory=init_dir,
                                                    options=openDialog.options() | QFileDialog.DontUseNativeDialog)
        if directory and os.path.isdir(directory) and check_file_mask:
            if not glob.glob(os.path.join(directory, check_file_mask)) \
                    and not glob.glob(os.path.join(directory, "**", check_file_mask), recursive=True):
                mBox = QMessageBox()
                dlgResult = mBox.question(self,
                                          "Диалог подтверждения",
                                          answer,
                                          QMessageBox.Yes | QMessageBox.No,
                                          QMessageBox.No)
                if dlgResult == QMessageBox.No:
                    return ""
        return directory

    def action_path_add_folder_click(self):
        directory = self.get_folder_images()
        if directory:
            self.pathListWidget.addItem(directory)

    def get_mask_images(self, init_path):
        inputDialog = QInputDialog()
        if not str.endswith(init_path, "*.jpg"):
            init_path = os.path.join(init_path, "*.jpg")

        while True:
            mask, ok = inputDialog.getText(self,
                                           "Введите маску для файлов директории",
                                           "Enter your name:",
                                           QLineEdit.Normal,
                                           init_path)

            if ok:
                if not glob.glob(mask, recursive=True):
                    dlgText = "По указанной маске не обнаружено файлов изображений. Все равно добавить маску?"
                    mBox = QMessageBox()
                    dlgResult = mBox.question(self,
                                              "Диалог подтверждения",
                                              dlgText,
                                              QMessageBox.Yes | QMessageBox.No,
                                              QMessageBox.No)
                    if dlgResult == QMessageBox.No:
                        continue
                return mask
            else:
                return ""

    def action_path_add_mask_click(self):
        directory = self.get_folder_images()
        if directory:
            mask = self.get_mask_images(directory)
            if mask:
                self.pathListWidget.addItem(mask)

    def action_path_edit_click(self):
        if self.pathListWidget.currentItem():
            path = self.pathListWidget.currentItem().text()
            withSubFolders = False
            if str.endswith(path, "**"):
                path = path[0:-2]
                withSubFolders = True
            newStr = ""
            if os.path.exists(path):
                # Папка
                if os.path.isdir(path):
                    newStr = self.get_folder_images(path)
                    if withSubFolders:
                        newStr += "**"
                # Файл
                else:
                    newStr = self.get_files_images(path, False)
            # Маска
            else:
                newStr = self.get_mask_images(path)
            if newStr:
                self.pathListWidget.currentItem().setText(newStr)
                self.path_list_widget_item_selected()

    def action_path_sub_folder_click(self):
        if self.pathListWidget.currentItem():
            if self.actionPathSubFolder.isChecked():
                if not str.endswith(self.pathListWidget.currentItem().text(), "**"):
                    self.pathListWidget.currentItem().setText(self.pathListWidget.currentItem().text() + "**")
            elif str.endswith(self.pathListWidget.currentItem().text(), "**"):
                self.pathListWidget.currentItem().setText(self.pathListWidget.currentItem().text()[0:-2])
            self.path_list_widget_item_selected()

    def path_list_widget_item_selected(self):
        self.imagesListWidget.clear()
        if self.pathListWidget.currentItem():
            path = self.pathListWidget.currentItem().text()
            withSubFolders = False
            if str.endswith(path, "**"):
                path = path[0:-2]
                withSubFolders = True
            if os.path.split(path)[1] and not str.endswith(os.path.split(path)[1], ".jpg"):
                path = os.path.join(path, "*.jpg")
            files = glob.glob(path, recursive=True)
            if withSubFolders:
                files += glob.glob(os.path.join(os.path.split(path)[0], "**", "*.jpg"), recursive=True)
            files.sort()
            for file in files:
                self.imagesListWidget.addItem(file)
        endWord = get_end_of_word(self.imagesListWidget.count(), ("", "а", "ов"))
        self.labelImagesListWidget.setText(f"{self.imagesListWidget.count()} файл{endWord}")

    def image_list_widget_item_selected(self):
        if self.imagesListWidget.currentItem():
            loadImg = cv2.imread(self.imagesListWidget.currentItem().text(), cv2.IMREAD_COLOR)[:, :, ::-1]
            mainImg = np.copy(loadImg)
            image = self.imageSet.imgPaths.get(self.imagesListWidget.currentItem().text())
            if image:
                for obj in image.objectsFromImage:
                    lineWidth = 4
                    cv2.rectangle(mainImg,
                                  (int(obj.bndBox.xmin), int(obj.bndBox.ymin)),
                                  (int(obj.bndBox.xmax), int(obj.bndBox.ymax)),
                                  (255, 0, 0), lineWidth)
                    textX = int(obj.bndBox.xmin)
                    textY = int(obj.bndBox.ymin) - lineWidth
                    cv2.putText(mainImg, obj.name, (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                self.colors[self.imageSet.objects[obj.name]], 1)
            # resizeCoefficient = min(self.imLabel.width() / mainImg.shape[1], self.imLabel.height() / mainImg.shape[0])
            resizeCoefficient = min(self.width() / 2 / mainImg.shape[1], self.height() / mainImg.shape[0])
            resizedImage = cv2.resize(mainImg.copy(),
                                      (int(resizeCoefficient * mainImg.shape[1]),
                                       int(resizeCoefficient * mainImg.shape[0])),
                                      interpolation=cv2.INTER_AREA)
            qImg = numpy_to_image(resizedImage)
            pixmap = QtGui.QPixmap.fromImage(qImg)
            self.imLabel.setPixmap(pixmap)

    # Отображение контекстного меню объектов
    def show_context_menu_object(self, point):
        if self.objectListWidget.currentItem():
            self.actionObjectEdit.setEnabled(True)
            self.actionObjectRemove.setEnabled(True)
        else:
            self.actionObjectEdit.setEnabled(False)
            self.actionObjectRemove.setEnabled(False)
        self.menuObjectListWidget.exec(QCursor.pos())

    def get_text_from_dialog(self, default_name=""):
        inputDialog = QInputDialog()
        text, ok = inputDialog.getText(self,
                                       "Введите название объекта распознавания",
                                       "Name:",
                                       QLineEdit.Normal,
                                       default_name)

        if ok:
            return text
        else:
            return ""

    def action_object_add_click(self):
        newName = self.get_text_from_dialog()
        if newName:
            self.imageSet.objects[newName] = len(self.imageSet.objects)
            self.objectListWidget.addItem(newName)

    def action_object_edit_click(self):
        newName = self.get_text_from_dialog(self.objectListWidget.currentItem().text())
        if newName:
            value = self.imageSet.objects.pop(self.objectListWidget.currentItem().text(), -1)
            self.imageSet.objects[newName] = value
            self.objectListWidget.currentItem().setText(newName)

    def action_object_remove_click(self):
        curValue = self.imageSet.objects.pop(self.objectListWidget.currentItem().text(), -1)
        for key in self.imageSet.objects.keys():
            if self.imageSet.objects[key] > curValue:
                self.imageSet.objects[key] = self.imageSet.objects[key] - 1
        self.objectListWidget.takeItem(self.objectListWidget.currentRow())


def get_end_of_word(count, ends):
    mod = count % 100
    if 5 <= mod <= 20:
        return ends[2]
    else:
        mod = (count - 1) % 10
        if mod == 0:
            return ends[0]
        elif mod < 4:
            return ends[1]
        else:
            return ends[2]


def numpy_to_image(image):
    if image.dtype == np.uint8:
        if len(image.shape) == 2:
            channels = 1
            height, width = image.shape
            bytesPerLine = channels * width
            qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_Indexed8)
            qImg.setColorTable([QtGui.qRgb(i, i, i) for i in range(256)])
            return qImg
        elif len(image.shape) == 3:
            if image.shape[2] == 3:
                height, width, channels = image.shape
                bytesPerLine = channels * width
                return QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
            elif image.shape[2] == 4:
                height, width, channels = image.shape
                bytesPerLine = channels * width
                fmt = QImage.Format_ARGB32
                return QImage(image.data, width, height, bytesPerLine, QImage.Format_ARGB32)
