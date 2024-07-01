from lawbot_document_handler import LawBotDocument, DocumentVersion

LawBotDocument.prep_workspace_from_zip('./non_release/acts of parliment.zip')
docs = LawBotDocument.load_from_workspace()
#doc.load_analyze_result(DocumentVersion.Base)
