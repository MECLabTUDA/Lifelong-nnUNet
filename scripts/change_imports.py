import os

def change_imports(codeDirectory, importTexts, importReplacemts):
    r"""This function goes through each Python Code file (.py) in the directory and changes the imports   
        from importTexts to importReplacemts. The length of theses lists need to be the same and
        corresponding regarding their indices.
        NOTE: This method loads the whole content of Python files into memory, so consider this when
        large Python files are present!
    """
    # -- Check that input lists are corresponding and have same length -- #
    assert len(importTexts) == len(importReplacemts), 'The input lists do not have the same length!'

    # -- Walk through codeDirectory and change imports -- #
    for dname, dirs, files in os.walk(codeDirectory):
        print('Walk trough directory \'{}\' and change imports..'.format(dname))
        for num, fname in enumerate(files):
            msg = 'Changing imports in ' + str(fname) + ' file.\n'
            msg += str(num + 1) + ' of ' + str(len(files)) + ' file(s).'
            print (msg, end = '\r')
            # -- Check if file is a Python file and exclude binary files -- #
            if '.py' in fname and '._' not in fname:
                fpath = os.path.join(dname, fname)
                with open(fpath) as f:
                    s = f.read()
                for idx, text in enumerate(importTexts):
                    # -- Only replace if text in s -- #
                    if text in s:
                        s = s.replace(text, importReplacemts[idx])
                with open(fpath, "w") as f:
                    f.write(s)

if __name__ == '__main__':
    codeDirectory = 'path_to_dir_with_py_files'
    importTexts = ['nnunet.paths', 'nnunet.run.default_configuration']  # Change accordingly
    importReplacemts = ['nnunet_ext.paths', 'nnunet_ext.run.default_configuration']  # Change accordingly
    # -- 'import os' will be changed in each file to 'import JIP.osp' -- #
    change_imports(codeDirectory, importTexts, importReplacemts)