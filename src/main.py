from lawbot_document_handler import LawBotDocument, DocumentVersion

try: 
    LawBotDocument.prep_workspace_from_zip('./non_release/acts of parliment.zip')
    #docs = LawBotDocument.load_from_workspace()
except:
    pass