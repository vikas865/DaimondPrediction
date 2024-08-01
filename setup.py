
from setuptools import find_packages,setup
from typing import List

HyPEN_E_DOT ='-e .'

def get_requiremnts(file_path:str)->List[str]:
    requiremnts=[]
    with open(file_path) as file_obj:
        requiremnts= file_obj.readlines()
        requiremnts=[req.replace("\n","") for req in requiremnts]

    if HyPEN_E_DOT in requiremnts:
        requiremnts.remove(HyPEN_E_DOT)
    return requiremnts 



setup(name='DiamondPricePrediction',
      version='0.0.0.1',
      description='Daimond Price Pridction',
      author='vikas',
      author_email='vikas865@gmail.com',
      install_requires=get_requiremnts('requirements.txt'),
      packages=find_packages()
     )