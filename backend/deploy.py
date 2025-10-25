import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DEPLOY_DIR = os.path.join(BASE_DIR, 'deployed')

os.makedirs(DEPLOY_DIR, exist_ok=True)


def deploy_model(model_name, deployed_name=None):
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError('model not found')
    target_name = deployed_name or model_name
    target_path = os.path.join(DEPLOY_DIR, target_name)
    shutil.copy(model_path, target_path)
    return target_path


if __name__ == '__main__':
    print('deploy module')
