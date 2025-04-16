import os
import numpy as np
import pymupdf
from pymupdf.utils import get_pixmap
from typing import Literal
from utils import print_progress_bar
from pathlib import Path
from zipfile import ZipFile
import re
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from inference import get_model
import supervision as sv
from supervision.detection.core import Detections
from supervision.geometry.core import Rect
from supervision.draw.utils import draw_filled_rectangle
from supervision.draw.color import Color
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

class LawbotWorkspace:
    workspace_path: Path

    def __init__(self, filePath: Path = Path('./lawbot_workspace')):
        self.workspace_path = filePath

    def prep_workspace_from_zip(self, zipPath: Path):
        if not self.workspace_path.exists():
            print(f'Creating workspace: {self.workspace_path.absolute()}')
            self.workspace_path.mkdir()
            lawbot_workspace_documents_folder = self.workspace_path.joinpath('documents')
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
                new_stem = re.sub(r'\.{2,}', '.', file_info_path_object.stem)
                #replace if ends with number between brackets, indicating duplicate
                new_stem = re.sub(r'\(\d+\)$', '', new_stem)
                #strip because some names ended with white space
                new_stem = new_stem.strip()
                if new_stem in added_file_names:
                    print(f'Duplicate file name: {new_stem}, skipping')
                    continue
                file_info_path_object = file_info_path_object.with_stem(f'{file_info_path_object.parent}_{new_stem}')
                lawbot_file_workspace_folder = lawbot_workspace_documents_folder.joinpath(file_info_path_object.stem)
                lawbot_file_workspace_folder.mkdir()
                lawbot_file_target_path = lawbot_file_workspace_folder.joinpath(file_info_path_object.name)
                file = lawbot_file_target_path.open('wb')
                file.write(lawbot_file_source.read())
                image_folder_path = lawbot_file_workspace_folder.joinpath('page_images')
                image_folder_path.mkdir()
                create_document_page_images(lawbot_file_target_path, image_folder_path)
                added_file_names.append(new_stem)
                print_progress_bar(i+1, zip_documents_count)
        else:
            print('Lawbot workspace already exists, skipping workspace creation')

def create_document_page_images(documentPath: Path, outputFolderPath: Path):
    print(f'Creating page images for document: {documentPath.stem}')
    pdf_doc = pymupdf.open(documentPath)
    page_count = pdf_doc.page_count
    for page_index in range(page_count):
        page_num = page_index + 1
        page_image_path = outputFolderPath.joinpath(f'page_{page_num}').with_suffix('.png')
        if not page_image_path.exists():
            page = pdf_doc.load_page(page_index)
            get_pixmap(page).save(page_image_path)
        else:
            print(f'Page {page_num} image already exists, skipping.')
        print_progress_bar(page_num, page_count)

def clean_page(imagePath: Path, confidence: float = 0.08):
    modelId = os.getenv('ROBOFLOW_MODEL_ID')
    apiKey = os.getenv('ROBOFLOW_API_KEY')
    model = get_model(model_id=modelId, api_key=apiKey)
    image = Image.open(imagePath)
    imagedata = np.array(image, copy=True)
    results = model.infer(image,confidence=confidence)[0]
    detections = Detections.from_inference(results)
    for detection in detections.xyxy:
        # annotate the image with the detections
        rect=Rect.from_xyxy(detection)
        cleanedImageData=draw_filled_rectangle(scene=imagedata,rect=rect,color=Color.from_hex("#FFFFFF")) 
        cleanedImage = Image.fromarray(cleanedImageData)
    return cleanedImage

def document_intelligence_extract(filePath: Path, outputFormat: Literal["text", "markdown"]):
    endpoint_env_var_name = 'DOCUMENT_INTELLIGENCE_ENDPOINT'
    key_env_var_name = 'DOCUMENT_INTELLIGENCE_KEY'
    endpoint = os.getenv(endpoint_env_var_name)
    apiKey = os.getenv(key_env_var_name)

    if apiKey == None:
        raise Exception(f'Document Intelligence Key missing.\nAdd your Azure Document Intelligence Key as an environment variable named {key_env_var_name}')
    if endpoint == None:
        raise Exception(f'Document Intelligence Endpoint missing.\nAdd your Azure Document Intelligence Endpoint as an environment variable named {endpoint_env_var_name}')

    client = DocumentIntelligenceClient(
        endpoint=endpoint, credential=AzureKeyCredential(apiKey)
    )

    fileBinary = filePath.open("rb")

    if outputFormat == "text":
        model = "prebuilt-read"
    else:
        model = "prebuilt-layout"

    poller = client.begin_analyze_document(
        model, fileBinary, output_content_format=outputFormat,
    )

    return poller.result()    

def get_all_pdf_file_paths_from_directory(directoryPath: str):
    pathObj = Path(directoryPath)
    globList = list(pathObj.glob('*.pdf'))
    filePaths = [x.as_posix() for x in globList]
    return filePaths