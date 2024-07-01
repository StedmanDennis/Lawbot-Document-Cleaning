import json
import pymupdf
from enum import Enum
from typing import List, Union, ClassVar #remove this when upgraded to python 9.10+ and use | instead
from utils import display_progress
from pathlib import Path
from zipfile import ZipFile
import shutil
import re
from document_intelligence import document_intelligence_extract
from azure.ai.formrecognizer import AnalyzeResult
import xml.etree.ElementTree as ET

def get_all_pdf_file_paths_from_directory(directoryPath: str):
    pathObj = Path(directoryPath)
    globList = list(pathObj.glob('*.pdf'))
    filePaths = [x.as_posix() for x in globList]
    return filePaths

class DocumentVersion(Enum):
    Base = 1
    Clean = 2

class LabelingSource(Enum):
    labelimg_xml = 1
    via_csv = 2
    via_json = 3
    via_coco_json = 4

class LawBotDocument:
    default_workspace_path: ClassVar[Path] = Path('./lawbot_workspace')
    cleaned_name_tail: ClassVar[str] = '_cleaned'
    analyze_result_name_tail: ClassVar[str] = '_analyze_result'
    page_img_suffix: ClassVar[str] = '.png'
    
    base_doc_path: Path
    cleaned_doc_path: Path
    base_analyze_result_path: Path
    cleaned_analyze_result_path: Path
    base_analyze_result: Union[AnalyzeResult, None]
    cleaned_analyze_result: Union[AnalyzeResult, None]

    def __init__(self, filePath: str):
        print(f'init: {filePath}')
        self.base_doc_path = Path(filePath)
        if not self.base_doc_path.is_relative_to(self.default_workspace_path):
            raise Exception(f'File must be within {self.default_workspace_path.as_posix()}')
        pdf = pymupdf.open(self.base_doc_path)
        if pdf.is_repaired:
            raise Exception(f'File is corrupted')
        self.cleaned_doc_path = self.base_doc_path.with_stem(f'{self.base_doc_path.stem}{self.cleaned_name_tail}')
        self.base_analyze_result_path = self.base_doc_path.with_stem(f'{self.base_doc_path.stem}{self.analyze_result_name_tail}').with_suffix('.json')
        self.cleaned_analyze_result_path = self.base_analyze_result_path.with_stem(f'{self.base_analyze_result_path.stem}{self.cleaned_name_tail}')
        self.analyze_result = None
        self.cleaned_analyze_result = None

    @classmethod
    def prep_workspace_from_zip(cls, zipPath: str):
        lawbot_workspace_folder = cls.default_workspace_path
        lawbot_workspace_documents_folder = lawbot_workspace_folder.joinpath('documents')
        if not lawbot_workspace_folder.exists():
            print('Creating lawbot workspace')
            lawbot_workspace_folder.mkdir()
            lawbot_workspace_documents_folder.mkdir()
            zip_file = ZipFile(zipPath, 'r')
            zip_documents = zip_file.filelist
            zip_documents_count = len(zip_documents)
            print('Copying documents to workspace')
            added_file_names = []
            for i, file_info in enumerate(zip_documents):
                lawbot_file_source = zip_file.open(file_info, 'r')
                file_info_path_object = Path(file_info.filename)
                #https://www.mtu.edu/umc/services/websites/writing/characters-avoid/
                #replace a sequence of 2 or more '.' with just one
                #strip because some names ended with white space
                new_stem = re.sub(r'\.{2,}', '.', file_info_path_object.stem)
                #replace if ends with number between brackets, indicating duplcate
                new_stem = re.sub(r'\(\d+\)$', '', new_stem).strip()
                if new_stem in added_file_names:
                    print(f'Duplicate file name, skipping')
                    continue
                file_info_path_object = file_info_path_object.with_stem(f'{file_info_path_object.parent}_{new_stem}')
                lawbot_file_workspace_folder = lawbot_workspace_documents_folder.joinpath(file_info_path_object.stem)
                lawbot_file_workspace_folder.mkdir()
                lawbot_file_workspace_folder.joinpath('page_images').mkdir()
                lawbot_file_target = lawbot_file_workspace_folder.joinpath(file_info_path_object.name)
                shutil.copyfileobj(lawbot_file_source, lawbot_file_target.open('wb'))
                try:
                    LawBotDocument(lawbot_file_target.as_posix()).__create_page_images()
                except:
                    pass
                added_file_names.append(new_stem)
                display_progress(i+1, zip_documents_count)
        else:
            print('Lawbot workspace already exists, skipping workspace creation')

    @classmethod
    def load_from_workspace(cls):
        lawbot_workspace_folder = cls.default_workspace_path
        lawbot_workspace_documents_folder = lawbot_workspace_folder.joinpath('documents')
        if not lawbot_workspace_folder.exists():
            raise Exception("No workspace found. Please run LawbotDocument.prep_workspace_from_zip to create one")
        #https://stackoverflow.com/questions/20638040/glob-exclude-pattern
        #https://docs.python.org/3/library/fnmatch.html#module-fnmatch
        globList = list(lawbot_workspace_documents_folder.rglob(f'*[!{cls.cleaned_name_tail}].pdf'))
        filePaths = [x.as_posix() for x in globList]
        lawbot_docs: List[LawBotDocument] = []
        skipped_doc_paths: List[str] = [] 
        doc_count = len(filePaths)
        for i, doc_path in enumerate(filePaths):
            try:
                doc = LawBotDocument(doc_path)
                lawbot_docs.append(doc)
            except:
                skipped_doc_paths.append(doc_path)
                print(f'skipped {doc_path} due to error')
            display_progress(i+1, doc_count)
        #one of these could be calculated through substraction from doc count instead of using the len function on both
        loaded_doc_count = len(lawbot_docs)
        skipped_doc_count = len(skipped_doc_paths)
        print(f'{loaded_doc_count} of {doc_count} documents loaded')
        print(f'Documents skipped: {skipped_doc_count}')
        if skipped_doc_count > 0:
            print('Skipped document paths: ')
            print(skipped_doc_paths)
        return lawbot_docs

    def __get_doc_path(self, version: DocumentVersion):
        if version == DocumentVersion.Base:
            return self.base_doc_path
        elif version == DocumentVersion.Clean:
            return self.cleaned_doc_path
        
    def __get_analyse_result_path(self, version: DocumentVersion):
        if version == DocumentVersion.Base:
            return self.base_analyze_result_path
        elif version == DocumentVersion.Clean:
            return self.cleaned_analyze_result_path
        
    def __get_analyse_result(self, version: DocumentVersion):
        if version == DocumentVersion.Base:
            return self.base_analyze_result
        elif version == DocumentVersion.Clean:
            return self.cleaned_analyze_result
        
    def __set_analyse_result(self, value: Union[AnalyzeResult, None], version: DocumentVersion):
        if version == DocumentVersion.Base:
            self.base_analyze_result = value
        elif version == DocumentVersion.Clean:
            self.cleaned_analyze_result = value

    def __get_page_img_path(self, pageNumber: int):
        doc_path = self.__get_doc_path(DocumentVersion.Base).parent.joinpath('page_images')
        return doc_path.joinpath(f'page_{pageNumber}').with_suffix(f'{self.page_img_suffix}')
    
    def __get_page_labeling_xml_path(self, pageNumber: int):
        doc_path = self.__get_doc_path(DocumentVersion.Base).parent.joinpath('page_images')
        return doc_path.joinpath(f'page_{pageNumber}.xml')

    def load_analyze_result(self, version: DocumentVersion):
        analyze_result_path = self.__get_analyse_result_path(version)
        if analyze_result_path.exists():
            print('Analyze result JSON found. Loading from file.')
            json_data = json.load(analyze_result_path.open('r'))
            self.analyze_result = AnalyzeResult.from_dict(json_data)
            print('Analyze result JSON loading complete.')
        else:
            print('No analyze result JSON found.') 
            self.analyze_result = self.__analyze_document(version)

    def __analyze_document(self, version: DocumentVersion):
        print('Analyzing document using Azure Document Intelligence.')
        doc_path = self.__get_doc_path(version)
        try:
            analyze_result = document_intelligence_extract(doc_path)
            print('Analysis complete.')
        except Exception as err:
            print("Failed to analyze document: ")
            print(err)
            analyze_result = None
        self.__set_analyse_result(analyze_result, version)
        self.__save_analyze_result(version)

    def __save_analyze_result(self, version: DocumentVersion):
        analyze_result = self.__get_analyse_result(version)
        if analyze_result is not None:
            analyze_result_path = self.__get_analyse_result_path(version)
            print('Saving analysis result.')
            json_data = analyze_result_path.open('w')
            json.dump(analyze_result.to_dict(), json_data, indent=4)
            print('Analysis result saving complete.')
    
    def __load_pdf(self, version: DocumentVersion):
        if version == DocumentVersion.Base:
            return pymupdf.open(self.base_doc_path)
        elif version == DocumentVersion.Clean:
            return pymupdf.open(self.cleaned_doc_path)

    #https://github.com/HumanSignal/labelImg/issues/11
    #https://github.com/amueller/ImageNet-parsing-Python/blob/25ebdb6d0bb14dc29eec732d62068697f84649bd/imagenet_analysis.py#L112
    def __parse_page_labelimg_xml(self, pageNum):
        """Get bouning box coordinates for image with id ``imageid``
        in synset given by ``classidx``."""

        labeling_path = self.__get_page_labeling_xml_path(pageNum)
        result = []
        if labeling_path.exists():
            xmltree = ET.parse(labeling_path)
            objects = xmltree.findall('object')
            for object_iter in objects:
                bndbox = object_iter.find('bndbox')
                if bndbox is not None:
                    bndbox
                    result.append([int(it.text) for it in bndbox])
                #[xmin, ymin, xmax, ymax] = [it.text for it in bndbox]
        return result

    def create_cleaned_document(self):
        print('Creating cleaned document.')
        pdf_doc = self.__load_pdf(DocumentVersion.Base)
        page_count = pdf_doc.page_count
        for page_index in range(page_count):
            page_num = page_index + 1
            page_cleaning_boxes = self.__parse_page_labelimg_xml(page_num)
            page = pdf_doc.load_page(page_index)
            for box in page_cleaning_boxes:
                page.draw_rect(pymupdf.Rect(box[0], box[1], box[2], box[3]), color=(1, 1, 1), fill=(1, 1, 1))
            pdf_doc.save(self.__get_doc_path(DocumentVersion.Clean).with_stem(f'{self.base_doc_path.stem}{self.cleaned_name_tail}'))
            display_progress(page_num, page_count)

    def __create_page_images(self):
        print('Creating page images.')
        pdf_doc = self.__load_pdf(DocumentVersion.Base)
        page_count = pdf_doc.page_count
        for page_index in range(page_count):
            page_num = page_index + 1
            page_image_path = self.__get_page_img_path(page_num)
            if not page_image_path.exists():
                page = pdf_doc.load_page(page_index)
                pix_map: pymupdf.Pixmap = page.get_pixmap()
                pix_map.save(page_image_path)
                print(f'Saving page {page_num}')
            else:
                print(f'Page {page_num} image already exists, skipping.')
            display_progress(page_num, page_count)