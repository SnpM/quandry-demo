# build python module from ./quandry_package
python -m build quandry_package

# copy .whl file to .
cp ./quandry_package/dist/quandry*.whl .

# install the module
pip install quandry*.whl --force-reinstall --no-deps