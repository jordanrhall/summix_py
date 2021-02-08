from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(
      name='summix_py',
      version='0.0.1',
      description='efficiently solves ancestral deconvolution problems',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
      ],
      keywords='statgen deconvolution optmization',
      url='https://github.com/jordanrhall/summix_py',
      author='Jordan R. Hall',
      author_email='jordanroberthall@gmail.com',
      license='MIT',
      packages=['summix'],
      install_requires=[
          'numpy',
          'scipy >= 0.15.0',
          'pandas'
      ],
      #test_suite='nose.collector',
      #tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)
