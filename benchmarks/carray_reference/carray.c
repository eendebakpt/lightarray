/* Minimal C-API array: owned double buffer, same semantics as lightarray.PyArray.
   Used only to isolate PyO3 binding overhead from "compiled code vs NumPy". */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    PyObject_HEAD
    double *data;
    Py_ssize_t n;
} CArray;

static PyTypeObject CArrayType;

static CArray *carray_alloc(Py_ssize_t n) {
    CArray *a = PyObject_New(CArray, &CArrayType);
    if (!a) return NULL;
    a->n = n;
    a->data = (double *)malloc(n ? n * sizeof(double) : 1);
    if (!a->data) { Py_DECREF(a); return PyErr_NoMemory(), NULL; }
    return a;
}

static void carray_dealloc(CArray *self) {
    free(self->data);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *carray_add(PyObject *x, PyObject *y) {
    if (!PyObject_TypeCheck(x, &CArrayType) || !PyObject_TypeCheck(y, &CArrayType))
        Py_RETURN_NOTIMPLEMENTED;
    CArray *a = (CArray *)x, *b = (CArray *)y;
    if (a->n != b->n) { PyErr_SetString(PyExc_ValueError, "shape mismatch"); return NULL; }
    CArray *r = carray_alloc(a->n);
    if (!r) return NULL;
    for (Py_ssize_t i = 0; i < a->n; i++) r->data[i] = a->data[i] + b->data[i];
    return (PyObject *)r;
}

static PyObject *carray_sum(CArray *self, PyObject *Py_UNUSED(ignored)) {
    double s = 0.0;
    for (Py_ssize_t i = 0; i < self->n; i++) s += self->data[i];
    return PyFloat_FromDouble(s);
}

static PyObject *carray_noop(CArray *self, PyObject *Py_UNUSED(ignored)) { Py_RETURN_NONE; }

static PyObject *carray_get_size(CArray *self, void *Py_UNUSED(c)) { return PyLong_FromSsize_t(self->n); }

static PyObject *carray_getitem(CArray *self, Py_ssize_t i) {
    if (i < 0 || i >= self->n) { PyErr_SetString(PyExc_IndexError, "index out of bounds"); return NULL; }
    return PyFloat_FromDouble(self->data[i]);
}

static PyNumberMethods carray_as_number = { .nb_add = carray_add };
static PySequenceMethods carray_as_sequence = { .sq_item = (ssizeargfunc)carray_getitem };
static PyMethodDef carray_methods[] = {
    {"sum", (PyCFunction)carray_sum, METH_NOARGS, NULL},
    {"_noop", (PyCFunction)carray_noop, METH_NOARGS, NULL},
    {NULL}
};
static PyGetSetDef carray_getset[] = {
    {"size", (getter)carray_get_size, NULL, NULL, NULL},
    {NULL}
};

static PyTypeObject CArrayType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "carray.CArray",
    .tp_basicsize = sizeof(CArray),
    .tp_dealloc = (destructor)carray_dealloc,
    .tp_as_number = &carray_as_number,
    .tp_as_sequence = &carray_as_sequence,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = carray_methods,
    .tp_getset = carray_getset,
};

static PyObject *mod_make_array(PyObject *Py_UNUSED(m), PyObject *seq) {
    PyObject *fast = PySequence_Fast(seq, "expected a sequence");
    if (!fast) return NULL;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    CArray *r = carray_alloc(n);
    if (!r) { Py_DECREF(fast); return NULL; }
    PyObject **items = PySequence_Fast_ITEMS(fast);
    for (Py_ssize_t i = 0; i < n; i++) {
        r->data[i] = PyFloat_AsDouble(items[i]);
        if (r->data[i] == -1.0 && PyErr_Occurred()) { Py_DECREF(fast); Py_DECREF(r); return NULL; }
    }
    Py_DECREF(fast);
    return (PyObject *)r;
}

static PyObject *mod_noop(PyObject *Py_UNUSED(m), PyObject *Py_UNUSED(a)) { Py_RETURN_NONE; }

static PyMethodDef mod_methods[] = {
    {"make_array", mod_make_array, METH_O, NULL},
    {"_noop", mod_noop, METH_NOARGS, NULL},
    {NULL}
};

static struct PyModuleDef moddef = { PyModuleDef_HEAD_INIT, "carray", NULL, -1, mod_methods };

PyMODINIT_FUNC PyInit_carray(void) {
    if (PyType_Ready(&CArrayType) < 0) return NULL;
    PyObject *m = PyModule_Create(&moddef);
    if (!m) return NULL;
    Py_INCREF(&CArrayType);
    if (PyModule_AddObject(m, "CArray", (PyObject *)&CArrayType) < 0) { Py_DECREF(&CArrayType); Py_DECREF(m); return NULL; }
    return m;
}
