# Django Learning

Django Learning is a content analysis and machine learning framework designed to make it easy to develop a codebook, 
validate it, collect training data with crowdsourcing, and apply it at scale with machine learning. At the moment, you 
won’t find bleeding edge deep learning here; instead, this library is intended to provide a rigorous and reliable 
framework for developing ML models with tried-and-true methods and validating them to ensure that they achieve 
performance comparable to humans.

## Installation

To install, you can use `pip`:

    pip install django_learning

Or you can install from source:

    git clone https://github.com/pewresearch/django_learning.git
    cd django_learning
    pip install -e .

### Installation Troubleshooting

#### Using 64-bit Python

Some of our libraries require the use of 64-bit Python. If you encounter errors during installation \
that are related to missing libraries, you may be using 32-bit Python. We recommend that you uninstall \
this version and switch to a 64-bit version instead. On Windows, these will be marked with `x86-64`; you \
can find the latest 64-bit versions of Python [here](https://www.python.org/downloads).

#### Installing ssdeep

ssdeep is an optional dependency that can be used by the `get_hash` function in Pewtils. \
Installation instructions for various Linux distributions can be found in the library's \
[documentation](https://python-ssdeep.readthedocs.io/en/latest/installation.html). The ssdeep \
Python library is not currently compatible with Windows. \
Installing ssdeep on Mac OS may involve a few additional steps, detailed below:

1. Install Homebrew
2. Install xcode
    ```
    xcode-select --install
    ```
3. Install system dependencies
    ```
    brew install pkg-config libffi libtool automake
    ln -s /usr/local/bin/glibtoolize /usr/local/bin/libtoolize
    ```
4. Install ssdeep with an additional flag to build the required libraries
    ```
    BUILD_LIB=1 pip install ssdeep
    ```
5. If step 4 fails, you may need to redirect your system to the new libraries by setting the following flags:
    ```
    export LIBTOOL=`which glibtool`
    export LIBTOOLIZE=`which glibtoolize`
    ```
   Do this and try step 4 again.
6. Now you should be able to run the main installation process detailed above.

## Documentation

Please refer to the [official documentation](https://pewresearch.github.io/django_learning/) for information on how to use this package.

## Use Policy

In addition to the [license](https://github.com/pewresearch/django_learning/blob/master/LICENSE), Users must abide by the following conditions:

- User may not use the Center's logo
- User may not use the Center's name in any advertising, marketing or promotional materials.
- User may not use the licensed materials in any manner that implies, suggests, or could otherwise be perceived as attributing a particular policy or lobbying objective or opinion to the Center, or as a Center endorsement of a cause, candidate, issue, party, product, business, organization, religion or viewpoint.

## Recommended Package Citation

Pew Research Center, 2021, "django_learning" Available at: github.com/pewresearch/django_learning

## Acknowledgements

The following authors contributed to this repository:

- Patrick van Kessel

## About Pew Research Center

Pew Research Center is a nonpartisan fact tank that informs the public about the issues, attitudes and trends shaping the world. It does not take policy positions. The Center conducts public opinion polling, demographic research, content analysis and other data-driven social science research. It studies U.S. politics and policy; journalism and media; internet, science and technology; religion and public life; Hispanic trends; global attitudes and trends; and U.S. social and demographic trends. All of the Center's reports are available at [www.pewresearch.org](http://www.pewresearch.org). Pew Research Center is a subsidiary of The Pew Charitable Trusts, its primary funder.

## Contact

For all inquiries, please email info@pewresearch.org. Please be sure to specify your deadline, and we will get back to you as soon as possible. This email account is monitored regularly by Pew Research Center Communications staff.
