import os
import shutil
import json
import numpy as np
import pymupdf
from pymupdf.utils import get_pixmap
from typing import Literal, TypedDict
from utils import print_progress_bar
from pathlib import Path
from zipfile import ZipFile
import re
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from inference import get_model
from supervision.detection.core import Detections
from supervision.geometry.core import Rect
from supervision.draw.utils import draw_filled_rectangle
from supervision.draw.color import Color
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

class LawbotDocumentMetadata(TypedDict):
    category: str
    originalName: str
    simpleName: str

class LawbotWorkspace:
    """
    Attributes:
        workspacePath: Folder path where actions on documents will occur within (some operations delete this folder, so avoid changing if possible)
        workspaceMetadataPath: File path where metadata.json file is expected to be
        documentSource: Location to where documents are (Currently expected to be a .zip file)
        workspaceMetadata: Metadata of the documents
    """
    _documentSource: Path
    _workspacePath: Path = Path('./lawbot_workspace')
    _workspaceMetadataPath: Path = _workspacePath.joinpath('metadata.json')
    _workspaceMetadata: list[LawbotDocumentMetadata]

    def __init__(self, source: Path):
        self._documentSource = source

    def prep_workspace(self):
        if self._workspacePath.exists():
            shutil.rmtree(self._workspacePath)
        self._workspacePath.mkdir()

        with ZipFile(self._documentSource, 'r') as zipFileData:
            zipDocuments = zipFileData.filelist
            zipDocumentsCount = len(zipDocuments)
            extractedFiles: list[str] = []
            extractedMetadata: list[LawbotDocumentMetadata] = []
            duplicateFilesCountDictionary: dict[str, int] = {}
            corruptedFiles: list[str] = []
            pageCount = 0
            for i, zipFileInfo in enumerate(zipDocuments):
                zipFilePath = Path(zipFileInfo.filename)
                #https://www.mtu.edu/umc/services/websites/writing/characters-avoid/
                #replace a sequence of 2 or more '.' with just one
                originalZipfileName = zipFilePath.stem
                normalizedName = re.sub(r'\.{2,}', '.', originalZipfileName)
                #replace if ends with number between brackets, indicating duplicate
                normalizedName = re.sub(r'\(\d+\)$', '', normalizedName)
                #strip because some names ended with white space
                normalizedName = normalizedName.strip()
                if normalizedName in extractedFiles:
                    if normalizedName in duplicateFilesCountDictionary:
                        fileDupCount = duplicateFilesCountDictionary[normalizedName]
                    else:
                        fileDupCount = 0
                    duplicateFilesCountDictionary[normalizedName] = fileDupCount + 1
                    continue
                with pymupdf.open(stream=zipFileData.read(zipFileInfo)) as extractedDocument:
                    if extractedDocument.is_repaired:
                        corruptedFiles.append(normalizedName)
                        continue
                    pageCount = pageCount + extractedDocument.page_count

                #with the presumption that all the document filenames are in the format of 'folder_name/file_name'
                #where folder_name is the document's category
                metaData: LawbotDocumentMetadata = {
                    'category': zipFilePath.parent.name,
                    'originalName': originalZipfileName,
                    'simpleName': str(i+1),
                }

                extractedMetadata.append(metaData)
                extractedFiles.append(normalizedName)
                print_progress_bar(i+1, zipDocumentsCount)

        extractedFilesCount = len(extractedFiles)
        duplicateFilesCount = sum(duplicateFilesCountDictionary.values())
        corruptedFilesCount = len(corruptedFiles)
        rejectedFilesCount = duplicateFilesCount + corruptedFilesCount

        json.dump(extractedMetadata, self._workspaceMetadataPath.open("w"), indent=4)
        self._workspaceMetadata = extractedMetadata
        
        print(f"Created a metadata file at {self._workspaceMetadataPath}.")
        print("Please give each entry of that file a unique simpleName.")
        print("For convenience, each entry has a unique number in string form\n")

        print("Summary:")
        print(f"Total files in zip: {zipDocumentsCount:,}")
        print(f"Total files rejected: {rejectedFilesCount:,}")
        print(f"\tTotal duplicate files: {duplicateFilesCount:,}")
        print(f"\tTotal corrupted files: {corruptedFilesCount:,}")
        print(f"Total files extracted: {extractedFilesCount:,}")
        print(f"\tTotal pages: {pageCount:,}")

        #https://azure.microsoft.com/en-us/pricing/details/ai-document-intelligence/
        docIntelPagePriceRate = 1000 #as in 'is priced per x'
        docIntelReadRate_millAndUnder = 1.5 / docIntelPagePriceRate
        docIntelReadRate_overMill = 0.6 / docIntelPagePriceRate
        docIntelLayoutRate = 10 / docIntelPagePriceRate
        
        if pageCount <= 1_000_000:
            docIntelReadPrice = pageCount * docIntelReadRate_millAndUnder
        else:
            docIntelReadPrice_mill = 1_000_000 * docIntelReadRate_millAndUnder
            pageCount_overMill = pageCount - 1_000_000
            docIntelReadPrice_overMill = pageCount_overMill * docIntelReadRate_overMill
            docIntelReadPrice = docIntelReadPrice_mill + docIntelReadPrice_overMill

        docIntelLayoutPrice = pageCount * docIntelLayoutRate

        print(f"\nApproximate Azure Document Intelligence costs:")
        print(f"\tRead: ${docIntelReadPrice:{',.2f'}}")
        print(f"\tLayout (Markdown): ${docIntelLayoutPrice:{',.2f'}}")

    """
    Creates in-memory representation of metadata.json
    """
    def load_metadata(self):
        metadata: list[LawbotDocumentMetadata] = json.load(self._workspaceMetadataPath.open('r'))
        if (validate_lawbot_metadata(metadata)):
            self._workspaceMetadata = metadata

    """
    Creates folder for document within workspace
    Args:
        simpleName: The document to load, uses the simpleName property of LawbotDocumentMetadata from metadata.json
    """
    def load_doc_folder(self, simpleName: str):
        with ZipFile(self._documentSource, 'r') as zipFileData:
            metadata = next((metadata for metadata in self._workspaceMetadata if metadata['simpleName'] == simpleName), None)
            if metadata:
                zipfilePath = f'{metadata["category"]}/{metadata["originalName"]}.pdf'
                zipInfo = next((info for info in zipFileData.filelist if zipfilePath == info.filename), None)
                if zipInfo:
                    folder = self._workspacePath.joinpath(simpleName)
                    if not folder.exists():
                        folder.mkdir()
                        fileBytes = zipFileData.read(zipInfo)
                        with pymupdf.open(stream=fileBytes) as document:
                            document.save(folder.joinpath(simpleName).with_suffix('.pdf'))
                            imageFolder = folder.joinpath("page_images")
                            imageFolder.mkdir()
                            for pageIndex in range(document.page_count):
                                pageNum = pageIndex + 1
                                pageImagePath = imageFolder.joinpath(f'page_{pageNum}').with_suffix('.png')
                                page = document.load_page(pageIndex)
                                get_pixmap(page).save(pageImagePath)
                    else:
                        print("Folder already exists")
                else:
                    print("Document does not exit in zip file")
            else:
                print(f"Could not find metadata with simpleName: {simpleName}")


def validate_lawbot_metadata(lawBotMetadata: list[LawbotDocumentMetadata]):
    simpleNameCountDictionary: dict[str, int] = {}
    
    for _, docMetaData in enumerate(lawBotMetadata):
        simpleName = docMetaData['simpleName']
        if simpleName in simpleNameCountDictionary:
            nameCount = simpleNameCountDictionary[simpleName] + 1
        else:
            nameCount = 1
        simpleNameCountDictionary[simpleName] = nameCount
    
    hasEmptySimpleName = "" in simpleNameCountDictionary
    nonUniqueSimpleNames = [key for (key, count) in simpleNameCountDictionary.items() if key != "" and count > 1]
    hasNonUniqueSimpleName = len(nonUniqueSimpleNames) >= 1

    isInvalid = hasEmptySimpleName or hasNonUniqueSimpleName
    
    if isInvalid:
        print('There are invalid values in document metadata:')
        if hasEmptySimpleName:
            missingSimpleNameDocs = [docMetadata["originalName"] for docMetadata in lawBotMetadata if docMetadata["simpleName"] == ""]
            print("No simple name:")
            for missing in missingSimpleNameDocs:
                print(f"\t{missing}")
        if hasNonUniqueSimpleName:
            nonUniqueSimpleNameDocs = [docMetadata["originalName"] for docMetadata in lawBotMetadata if docMetadata["simpleName"] in nonUniqueSimpleNames]
            print("Non-unique simple name:")
            for name in nonUniqueSimpleNameDocs:
                print(f"\t{name}")
    
    return not isInvalid

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