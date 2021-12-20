from distutils.core import setup

# this grabs the requirements from requirements.txt
REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]

setup(
  name = 'py_neuromodulation',
  packages = ['py_neuromodulation'],
  version = '0.01',
  license='MIT',
  description = 'Real-time analysis of intracranial neurophysiology recordings. ',
  author = 'Timon Merk',
  author_email = 'timon.merk95@gmail.com',
  url = 'https://github.com/neuromodulation/py_neuromodulation',
  download_url = 'https://github.com/neuromodulation/py_neuromodulation/archive/refs/tags/v0.01.tar.gz',
  keywords = [
      'python',
      'machine-learning',
      'real-time',
      'electrocorticography ',
      'dbs',
      'ecog ',
      'deep-brain-stimulation'],
  install_requires=REQUIREMENTS,
  classifiers=[  # Optional
    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    'Development Status :: 3 - Alpha',

    # Indicate who your project is intended for
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',

    # Pick your license as you wish
    'License :: OSI Approved :: MIT License',

    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    'Programming Language :: Python :: 3'
  ],
)