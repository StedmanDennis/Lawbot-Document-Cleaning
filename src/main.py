from lawbot_document_handler import LawBotDocument, DocumentVersion

LawBotDocument.prep_workspace_from_zip('./non_release/acts of parliment.zip')
try:
    docs = LawBotDocument.load_from_workspace()
except:
    pass
#doc.load_analyze_result(DocumentVersion.Base)
