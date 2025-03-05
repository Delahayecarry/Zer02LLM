from modelscope.hub.api import HubApi

YOUR_ACCESS_TOKEN = 'a6b0d8ce-464a-4b5d-96d7-70ee06e91e1d'
api = HubApi()
api.login(YOUR_ACCESS_TOKEN)

api.upload_folder(
    repo_id=f"Cao121121/Zer02LLM-v1-pretrain-0.02B",
    folder_path='./Zero2LLM-v1-0.02B-pretrained',
    commit_message='20250304-pretrain-0.02B',
)