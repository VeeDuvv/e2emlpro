

## how i created the virtual environment?
### Since i dont have conda installed, and brew does not help create a virtual environment, I used the following steps to create and activate the virtual environment.
    $ pip install virtualenv
    $ virtualenv myenv
    $ source myenv/bin/activate
    $ pip install package_name
    $ deactivate


### "-e ." in requirements.txt will ensure that setup.py is automatically invoked when requirements is run