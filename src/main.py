import json
from pathlib import Path
from lawbot_document_handler import LawbotWorkspace, clean_page, document_intelligence_extract

workspacePath = Path('./lawbot_workspace')
zipPath = Path('./non_release/acts of parliment.zip')

workspace = LawbotWorkspace(zipPath)
workspace.prep_workspace()

#document simpleName can be updated by changing the simpleName property of the metadata.json created by workspace.prep_workspace()
documentSimpleName = '18'

workspace.load_doc_folder(documentSimpleName)

imgPath = Path(f'./lawbot_workspace/{documentSimpleName}/page_images/page_1.png')
cleanedImgPath = imgPath.with_stem(imgPath.stem+'_cleaned')

cleanedImage = clean_page(
    imgPath, 
)

cleanedImage.save(cleanedImgPath)

extractionResultFolder = cleanedImgPath.parents[1].joinpath('extraction_results')

if not extractionResultFolder.exists():
    extractionResultFolder.mkdir()

textExtractResult = document_intelligence_extract(
    cleanedImgPath,
    "text"
)

textExtractionResultPath = extractionResultFolder.joinpath(cleanedImgPath.stem+'_text_extraction') 

json.dump(textExtractResult.as_dict(), textExtractionResultPath.with_suffix('.json').open('w'), indent=4)
textExtractionResultPath.with_suffix('.txt').open('w').write(textExtractResult.content)

markdownExtractResult = document_intelligence_extract(
    cleanedImgPath,
    "markdown"
)

markdownExtractionFilename = extractionResultFolder.joinpath(cleanedImgPath.stem+'_markdown_extraction')

json.dump(markdownExtractResult.as_dict(), markdownExtractionFilename.with_suffix('.json').open('w'), indent=4)
markdownExtractionFilename.with_suffix('.md').open('w').write(markdownExtractResult.content)