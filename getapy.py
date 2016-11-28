# This file was automatically generated by SWIG (http://www.swig.org).
# Version 2.0.10
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.



from sys import version_info
if version_info >= (2,6,0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_getapy', [dirname(__file__)])
        except ImportError:
            import _getapy
            return _getapy
        if fp is not None:
            try:
                _mod = imp.load_module('_getapy', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _getapy = swig_import_helper()
    del swig_import_helper
else:
    import _getapy
del version_info
try:
    _swig_property = property
except NameError:
    pass # Python < 2.2 doesn't have 'property'.
def _swig_setattr_nondynamic(self,class_type,name,value,static=1):
    if (name == "thisown"): return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name,None)
    if method: return method(self,value)
    if (not static):
        self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)

def _swig_setattr(self,class_type,name,value):
    return _swig_setattr_nondynamic(self,class_type,name,value,0)

def _swig_getattr(self,class_type,name):
    if (name == "thisown"): return self.this.own()
    method = class_type.__swig_getmethods__.get(name,None)
    if method: return method(self)
    raise AttributeError(name)

def _swig_repr(self):
    try: strthis = "proxy of " + self.this.__repr__()
    except: strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except AttributeError:
    class _object : pass
    _newclass = 0



def cdata(*args):
  return _getapy.cdata(*args)
cdata = _getapy.cdata

def memmove(*args):
  return _getapy.memmove(*args)
memmove = _getapy.memmove
class intp(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, intp, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, intp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _getapy.new_intp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _getapy.delete_intp
    __del__ = lambda self : None;
    def __getitem__(self, *args): return _getapy.intp___getitem__(self, *args)
    def __setitem__(self, *args): return _getapy.intp___setitem__(self, *args)
    def cast(self): return _getapy.intp_cast(self)
    __swig_getmethods__["frompointer"] = lambda x: _getapy.intp_frompointer
    if _newclass:frompointer = staticmethod(_getapy.intp_frompointer)
intp_swigregister = _getapy.intp_swigregister
intp_swigregister(intp)

def intp_frompointer(*args):
  return _getapy.intp_frompointer(*args)
intp_frompointer = _getapy.intp_frompointer


def fin(*args):
  return _getapy.fin(*args)
fin = _getapy.fin

def init(*args):
  return _getapy.init(*args)
init = _getapy.init

def search(*args):
  return _getapy.search(*args)
search = _getapy.search
# This file is compatible with both classic and new-style classes.

