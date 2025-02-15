# #Don't expose these classes above this scope
# #Only use the toSample factory to create instances
# # from .code_sample_model import CodeSampleModel
# from codetector.src.features.shared.domain.entities.samples.code_sample import CodeSample
# from .c import CSample
# from .cpp import CPPSample
# from .csharp import CSharpSample
# from .generic import GenericSample
# from .go import GoSample
# from .java import JavaSample
# from .javascript import JavascriptSample
# from .python import PythonSample
# from .rust import RustSample


# def toCodeSample(programmingLanguage:str) -> type[CodeSample]:
#     """
#     Convert a string representation to a class instance of `CodeSample`.
#     """
#     languages = {
#         'python': PythonSample,
#         'python3': PythonSample,
#         'python2': PythonSample,
#         'py': PythonSample,
#         'javascript': JavascriptSample,
#         'js': JavascriptSample,
#         'c': CSample,
#         'cpp': CPPSample,
#         'c++': CPPSample,
#         'java': JavaSample,
#         'rust': RustSample,
#         'go': GoSample,
#         'c#': CSharpSample,
#         'csharp': CSharpSample,
#     }

#     if isinstance(programmingLanguage, str) and programmingLanguage in languages:
#         return languages[programmingLanguage]
#     else:
#         return GenericSample
    