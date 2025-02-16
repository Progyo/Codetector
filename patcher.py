# Automatically patch typing.py to support generic typedefs.
# !! Run this command at your own discretion as it modifies base Python packages and may break your Python install.


from pathlib import Path
import re
import sys



def extractNewTypeClass(text: str) -> str:
    pattern = r'(?ms)^class\s+NewType\b(?:\s*\(.*?\))?:.*?(?=^\S|\Z)'
    match = re.search(pattern, text)
    if match:
        return match.group(0)
    return None

def getTypingFile(pythonLibPath:str) -> str:
    path = f'{pythonLibPath}/typing.py'
    if Path(path).exists():
        with open(path,'r') as file:
            return file.read()
    else:
        raise Exception(f'Couldn\'t find typing.py at {path}!')
    
def replaceTypingFile(pythonLibPath:str,newText:str) -> str:
    path = f'{pythonLibPath}/typing.py'
    if Path(path).exists():
        with open(path,'w+') as file:
            file.write(newText)
    else:
        raise Exception(f'Couldn\'t replace typing.py at {path}!')


#Thanks to @eltoder for the patch
#Original link: https://gist.github.com/eltoder/4035faa041112a988dcf3ab101fb3db1
PATCH = r"""class NewType(_Immutable):
    def __init__(self, name, tp, *, _tvars=()):
        self.__qualname__ = name
        if '.' in name:
            name = name.rpartition('.')[-1]
        self.__name__ = name
        self.__supertype__ = tp
        self.__parameters__ = _tvars
        def_mod = _caller()
        if def_mod != 'typing':
            self.__module__ = def_mod

    @_tp_cache
    def __class_getitem__(cls, params):
        # copied from Generic.__class_getitem__
        if not isinstance(params, tuple):
            params = (params,)
        if not params:
            raise TypeError(
                f"Parameter list to {cls.__qualname__}[...] cannot be empty")
        params = tuple(_type_convert(p) for p in params)
        if not all(isinstance(p, (TypeVar, ParamSpec)) for p in params):
            raise TypeError(
                f"Parameters to {cls.__name__}[...] must all be type variables "
                f"or parameter specification variables.")
        if len(set(params)) != len(params):
            raise TypeError(
                f"Parameters to {cls.__name__}[...] must all be unique")
        return functools.partial(cls, _tvars=params)

    @_tp_cache
    def __getitem__(self, params):
        # copied from Generic.__class_getitem__
        if not isinstance(params, tuple):
            params = (params,)
        params = tuple(_type_convert(p) for p in params)
        if any(isinstance(t, ParamSpec) for t in self.__parameters__):
            params = _prepare_paramspec_params(self, params)
        else:
            _check_generic(self, params, len(self.__parameters__))
        return _GenericAlias(self, params,
                             _typevar_types=(TypeVar, ParamSpec),
                             _paramspec_tvars=True)

    def __repr__(self):
        return f'{self.__module__}.{self.__qualname__}'

    def __call__(self, x):
        return x

    def __reduce__(self):
        return self.__qualname__

    def __or__(self, other):
        return Union[self, other]

    def __ror__(self, other):
        return Union[other, self]

"""

if __name__ == "__main__":

    # pythonVer = '.'.join(input('Please input your exact python version: ').lower().split('.')[:-1])

    version = sys.version_info
    pythonVer = f'python{version.major}.{version.minor}'

    pythonLibPath = str(Path(sys.executable).parent.parent.joinpath(f'lib/{pythonVer}'))

    print(f'Detected Python lib folder: {pythonLibPath}')

    fileStr = getTypingFile(pythonLibPath)

    classStr = extractNewTypeClass(fileStr)

    if classStr == None:
        print('Couldn\'t locate class.')
        exit()

    fileStr = fileStr.replace(classStr,PATCH)

    replaceTypingFile(pythonLibPath,fileStr)

    print('Successfully patched typing.py')
    
