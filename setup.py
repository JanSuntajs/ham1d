from setuptools import setup, find_packages


setup(name='ham1d',
      version='1.2.5',
      description='A module for calculations with 1D quantum hamiltonians',
      url='https://github.com/JanSuntajs/ham1d',
      author='Jan Suntajs',
      author_email='Jan.Suntajs@ijs.si',
      license='MIT',
      packages=find_packages(),
      install_requires=[('spectral_stats @ git+https://github.com/JanSuntajs/'
                         'spectral_statistics_tools@v1.1.1#egg=spectral_stats')], #/tarball/master/'
                         #'#egg=spectral_stats-1.1.1'), ],
      # dependency_links=[('https://github.com/JanSuntajs/'
      #                   'spectral_statistics_tools/tarball/'
      #                   'master/#egg=spectral_stats-1.1.1'),
      #                 ],
      zip_safe=False)
