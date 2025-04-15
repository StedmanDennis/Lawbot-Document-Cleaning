from pathlib import Path
import time
from lawbot_document_handler import LawbotWorkspace, clean_page, extract_text

workspace = LawbotWorkspace()
workspace.prep_workspace_from_zip('./acts of parliment.zip')

clean_page(
    Path('./lawbot_workspace/documents/revision_Water Supply Act/page_images/page_1.png'), 
    Path('./lawbot_workspace/documents/revision_Water Supply Act/page_images/page_1_cleaned.png')
)

extract_text(
    Path('./lawbot_workspace/documents/revision_Water Supply Act/page_images/page_1_cleaned.png'),
    Path('./lawbot_workspace/documents/revision_Water Supply Act/page_images/page_1_cleaned_text.json')
)
