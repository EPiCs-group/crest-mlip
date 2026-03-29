/*
 * This file is part of crest.
 * SPDX-License-Identifier: LGPL-3.0-or-later
 *
 * Modifications for MLIP support:
 * Copyright (C) 2024-2026 Alexander Kolganov
 *
 * crest is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * crest is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with crest.  If not, see <https://www.gnu.org/licenses/>.
 */

/*
 * pymlip_bridge.c — Embedded CPython bridge for MLIP inference.
 *
 * Embeds the Python interpreter and calls UMA/MACE calculators directly,
 * bypassing the TCP socket server. The GIL is acquired per-call via
 * PyGILState_Ensure/Release, making this thread-safe for OpenMP.
 *
 * The Python setup script is embedded as a string constant to avoid
 * needing external .py files at runtime.
 *
 * Compile with: -DWITH_PYMLIP and link against Python3 (-lpython3.x)
 */

#include "pymlip_bridge.h"

#ifdef WITH_PYMLIP

/* Must define PY_SSIZE_T_CLEAN before Python.h */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Internal context holding the Python calculator and cached objects */
typedef struct {
    PyObject* calculator;   /* Python calculator object (has .get_energy_and_forces()) */
    PyObject* calc_func;    /* Bound method: calculator.get_energy_and_forces */
    PyObject* numpy_mod;    /* numpy module (for array creation) */
    PyObject* np_frombuffer; /* cached numpy.frombuffer function */
    int charge;             /* molecular charge */
    int spin;               /* number of unpaired electrons (UHF) */
    int initialized;
} PyMLIPContext;

/* Track whether Python interpreter has been initialized */
static int python_initialized = 0;

/* Python setup module — loaded once, provides create_calculator() */
static PyObject* setup_module = NULL;

/* The embedded Python code that creates calculators */
static const char* SETUP_CODE =
    "import os\n"
    "os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'\n"
    "os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')\n"
    "import sys\n"
    "import numpy as np\n"
    "\n"
    "# Unit conversion constants\n"
    "BOHR_TO_ANG = 0.529177210903\n"
    "EV_TO_HARTREE = 1.0 / 27.211386245988\n"
    "\n"
    "class MLIPWrapper:\n"
    "    \"\"\"Wraps a UMA or MACE calculator for direct energy/gradient calls.\"\"\"\n"
    "\n"
    "    def __init__(self, model_type, model_path, device, task, atom_refs, charge, spin,\n"
    "                 compile_mode='', dtype='float64', turbo=0):\n"
    "        import torch\n"
    "        self.model_type = model_type.lower()\n"
    "        self.device = device\n"
    "        self.task = task\n"
    "        self.charge = charge  # molecular charge\n"
    "        self.spin = spin      # number of unpaired electrons (UHF)\n"
    "        self._atoms = None  # Cached ASE Atoms object\n"
    "        self._calc = None   # ASE calculator\n"
    "        self._compile_mode = compile_mode if compile_mode else None\n"
    "        self._dtype = dtype if dtype else 'float64'\n"
    "        self._turbo = bool(turbo)\n"
    "\n"
    "        # On GPU, limit PyTorch CPU threads to prevent oversubscription\n"
    "        if 'cuda' in device or 'mps' in device:\n"
    "            torch.set_num_threads(1)\n"
    "            try:\n"
    "                torch.set_num_interop_threads(1)\n"
    "            except RuntimeError:\n"
    "                pass  # already set, ignore\n"
    "            print(f'[pymlip] GPU mode: torch threads set to 1')\n"
    "            if 'cuda' in device:\n"
    "                torch.backends.cudnn.benchmark = True\n"
    "\n"
    "        if self.model_type == 'uma':\n"
    "            self._init_uma(model_path, device, task, atom_refs)\n"
    "        elif self.model_type == 'mace':\n"
    "            self._init_mace(model_path, device)\n"
    "        else:\n"
    "            raise ValueError(f'Unknown model type: {model_type}')\n"
    "\n"
    "    def _init_uma(self, model_path, device, task, atom_refs):\n"
    "        from fairchem.core import FAIRChemCalculator\n"
    "        import os\n"
    "\n"
    "        is_local = os.path.isfile(model_path)\n"
    "\n"
    "        if is_local:\n"
    "            # Local checkpoint: load via predict_unit for full control\n"
    "            from fairchem.core import pretrained_mlip\n"
    "            import yaml\n"
    "            atom_ref_dict = None\n"
    "            if atom_refs and len(atom_refs) > 0:\n"
    "                with open(atom_refs, 'r') as f:\n"
    "                    atom_ref_dict = yaml.safe_load(f)\n"
    "            load_kwargs = dict(\n"
    "                path=model_path,\n"
    "                device=device,\n"
    "                atom_refs=atom_ref_dict,\n"
    "            )\n"
    "            if self._turbo:\n"
    "                load_kwargs['inference_settings'] = 'turbo'\n"
    "                print(f'[pymlip] UMA turbo mode: tf32+compile+merge_mole')\n"
    "            pu = pretrained_mlip.load_predict_unit(**load_kwargs)\n"
    "            self._calc = FAIRChemCalculator(\n"
    "                predict_unit=pu,\n"
    "                task_name=task,\n"
    "            )\n"
    "            print(f'[pymlip] UMA loaded from local file: {model_path}')\n"
    "        else:\n"
    "            # Registry name: auto-download from HuggingFace\n"
    "            print(f'[pymlip] Downloading UMA model: {model_path}')\n"
    "            self._calc = FAIRChemCalculator.from_model_checkpoint(\n"
    "                name_or_path=model_path,\n"
    "                task_name=task,\n"
    "                device=device,\n"
    "            )\n"
    "            print(f'[pymlip] UMA model ready: {model_path}')\n"
    "\n"
    "        self._atom_refs = None\n"
    "\n"
    "    def _init_mace(self, model_path, device):\n"
    "        import os\n"
    "\n"
    "        is_local = os.path.isfile(model_path)\n"
    "\n"
    "        if is_local:\n"
    "            # Local checkpoint — do NOT pass compile_mode to MACECalculator.\n"
    "            # MACECalculator+compile_mode uses prepare(extract_model) which\n"
    "            # reconstructs the model from __init__. Newer mace-torch versions\n"
    "            # register extra buffers (weights_*_zeroed) that old checkpoints\n"
    "            # lack, causing load_state_dict to fail.  Instead, we load eagerly\n"
    "            # (plain torch.load/pickle) and apply torch.compile post-hoc.\n"
    "            from mace.calculators import MACECalculator\n"
    "            kwargs = dict(\n"
    "                model_paths=[model_path],\n"
    "                device=device,\n"
    "                default_dtype=self._dtype,\n"
    "            )\n"
    "            if self.task:\n"
    "                kwargs['head'] = self.task\n"
    "            self._calc = MACECalculator(**kwargs)\n"
    "            print(f'[pymlip] MACE loaded from local file: {model_path}')\n"
    "            # Apply torch.compile post-hoc (wraps the loaded model as-is)\n"
    "            if self._compile_mode:\n"
    "                try:\n"
    "                    import torch\n"
    "                    for i, m in enumerate(self._calc.models):\n"
    "                        self._calc.models[i] = torch.compile(m, mode=self._compile_mode)\n"
    "                    self._calc.use_compile = True\n"
    "                    print(f'[pymlip] MACE torch.compile({self._compile_mode}) enabled')\n"
    "                except Exception as e:\n"
    "                    print(f'[pymlip] WARNING: torch.compile failed: {e}, using eager mode')\n"
    "        elif model_path.startswith('off/'):\n"
    "            from mace.calculators import mace_off\n"
    "            size = model_path[4:]\n"
    "            print(f'[pymlip] Downloading MACE-OFF model: {size}')\n"
    "            self._calc = mace_off(model=size, device=device,\n"
    "                                  default_dtype=self._dtype)\n"
    "        elif model_path.startswith('omol/'):\n"
    "            from mace.calculators import mace_omol\n"
    "            size = model_path[5:]\n"
    "            print(f'[pymlip] Downloading MACE-OMOL model: {size}')\n"
    "            self._calc = mace_omol(model=size, device=device,\n"
    "                                   default_dtype=self._dtype)\n"
    "        else:\n"
    "            # Default: mace-mp foundation model\n"
    "            from mace.calculators import mace_mp\n"
    "            print(f'[pymlip] Downloading MACE-MP model: {model_path}')\n"
    "            self._calc = mace_mp(model=model_path, device=device,\n"
    "                                 default_dtype=self._dtype)\n"
    "\n"
    "        # Apply torch.compile post-hoc for registry models\n"
    "        if self._compile_mode and not is_local:\n"
    "            try:\n"
    "                import torch\n"
    "                for i, m in enumerate(self._calc.models):\n"
    "                    self._calc.models[i] = torch.compile(m, mode=self._compile_mode)\n"
    "                self._calc.use_compile = True\n"
    "                print(f'[pymlip] MACE torch.compile({self._compile_mode}) enabled')\n"
    "            except Exception as e:\n"
    "                print(f'[pymlip] WARNING: torch.compile failed: {e}, using eager mode')\n"
    "\n"
    "        self._atom_refs = None\n"
    "\n"
    "    def get_energy_and_forces(self, positions_bohr, atomic_numbers):\n"
    "        \"\"\"Compute energy (Hartree) and gradient (Hartree/Bohr).\n"
    "\n"
    "        Parameters:\n"
    "            positions_bohr: numpy array [nat, 3] in Bohr\n"
    "            atomic_numbers: numpy array [nat] of atomic numbers\n"
    "\n"
    "        Returns:\n"
    "            (energy_hartree, gradient_hartree_bohr) as numpy arrays\n"
    "        \"\"\"\n"
    "        from ase import Atoms\n"
    "        import numpy as np\n"
    "\n"
    "        nat = len(atomic_numbers)\n"
    "        pos_ang = positions_bohr * BOHR_TO_ANG\n"
    "\n"
    "        # Create or update ASE Atoms object\n"
    "        if self._atoms is None or len(self._atoms) != nat:\n"
    "            self._atoms = Atoms(\n"
    "                numbers=atomic_numbers,\n"
    "                positions=pos_ang,\n"
    "                pbc=False,\n"
    "            )\n"
    "            self._atoms.info['charge'] = self.charge\n"
    "            self._atoms.info['spin'] = self.spin\n"
    "            self._atoms.calc = self._calc\n"
    "        else:\n"
    "            self._atoms.set_positions(pos_ang)\n"
    "            if not np.array_equal(self._atoms.get_atomic_numbers(), atomic_numbers):\n"
    "                self._atoms.set_atomic_numbers(atomic_numbers)\n"
    "\n"
    "        # Compute\n"
    "        energy_ev = self._atoms.get_potential_energy()\n"
    "        forces_ev_ang = self._atoms.get_forces()\n"
    "\n"
    "        # Apply atom references if available\n"
    "        if self._atom_refs is not None:\n"
    "            for i, z in enumerate(atomic_numbers):\n"
    "                ref = self._atom_refs.get(int(z), 0.0)\n"
    "                energy_ev -= ref\n"
    "\n"
    "        # Convert units: eV -> Hartree, forces -> gradient\n"
    "        energy_hartree = energy_ev * EV_TO_HARTREE\n"
    "        # gradient = -forces, and convert eV/Ang -> Hartree/Bohr\n"
    "        gradient_hartree_bohr = -forces_ev_ang * EV_TO_HARTREE / BOHR_TO_ANG\n"
    "\n"
    "        return energy_hartree, gradient_hartree_bohr.ravel()\n"
    "\n"
    "\n"
    "def create_calculator(model_type, model_path, device, task, atom_refs, charge, spin,\n"
    "                       compile_mode='', dtype='float64', turbo=0):\n"
    "    return MLIPWrapper(model_type, model_path, device, task, atom_refs, charge, spin,\n"
    "                       compile_mode=compile_mode, dtype=dtype, turbo=turbo)\n";


/* Helper: write error message to buffer safely */
static void set_error(char* err_msg, int err_len, const char* msg) {
    if (err_msg && err_len > 0) {
        int n = (int)strlen(msg);
        if (n >= err_len) n = err_len - 1;
        memcpy(err_msg, msg, n);
        err_msg[n] = '\0';
    }
}

/* Helper: format Python exception into error buffer */
static void set_python_error(char* err_msg, int err_len) {
    if (!err_msg || err_len <= 0) return;

    PyObject *ptype, *pvalue, *ptraceback;
    PyErr_Fetch(&ptype, &pvalue, &ptraceback);

    if (pvalue) {
        PyObject* str = PyObject_Str(pvalue);
        if (str) {
            const char* s = PyUnicode_AsUTF8(str);
            if (s) {
                set_error(err_msg, err_len, s);
            } else {
                set_error(err_msg, err_len, "Python error (could not decode)");
            }
            Py_DECREF(str);
        } else {
            set_error(err_msg, err_len, "Python error (no string repr)");
        }
    } else {
        set_error(err_msg, err_len, "Unknown Python error");
    }

    Py_XDECREF(ptype);
    Py_XDECREF(pvalue);
    Py_XDECREF(ptraceback);
}


int pymlip_init_python(char* err_msg, int err_len) {
    if (python_initialized) return 0;

    if (!Py_IsInitialized()) {
        Py_Initialize();
        if (!Py_IsInitialized()) {
            set_error(err_msg, err_len, "Failed to initialize Python interpreter");
            return 1;
        }
    }

    /* Import numpy early to ensure it's available */
    PyObject* np = PyImport_ImportModule("numpy");
    if (!np) {
        set_python_error(err_msg, err_len);
        return 2;
    }
    Py_DECREF(np);

    /* Compile and execute the setup code as a module */
    PyObject* code = Py_CompileString(SETUP_CODE, "pymlip_setup", Py_file_input);
    if (!code) {
        set_python_error(err_msg, err_len);
        return 3;
    }

    setup_module = PyImport_ExecCodeModule("pymlip_setup", code);
    Py_DECREF(code);
    if (!setup_module) {
        set_python_error(err_msg, err_len);
        return 4;
    }

    /* Release the GIL so other threads can acquire it */
    PyEval_SaveThread();

    python_initialized = 1;
    return 0;
}


pymlip_handle_t pymlip_create(const char* model_type,
                               const char* model_path,
                               const char* device,
                               const char* task,
                               const char* atom_refs,
                               int charge,
                               int spin,
                               const char* compile_mode,
                               const char* dtype,
                               int turbo,
                               char* err_msg, int err_len)
{
    if (!python_initialized) {
        int rc = pymlip_init_python(err_msg, err_len);
        if (rc != 0) return NULL;
    }

    /* Acquire the GIL */
    PyGILState_STATE gstate = PyGILState_Ensure();

    PyMLIPContext* ctx = (PyMLIPContext*)calloc(1, sizeof(PyMLIPContext));
    if (!ctx) {
        set_error(err_msg, err_len, "Memory allocation failed");
        PyGILState_Release(gstate);
        return NULL;
    }

    /* Store charge and spin in context */
    ctx->charge = charge;
    ctx->spin = spin;

    /* Call create_calculator(model_type, model_path, device, task, atom_refs,
     *                        charge, spin, compile_mode, dtype, turbo) */
    PyObject* func = PyObject_GetAttrString(setup_module, "create_calculator");
    if (!func) {
        set_python_error(err_msg, err_len);
        free(ctx);
        PyGILState_Release(gstate);
        return NULL;
    }

    PyObject* args = Py_BuildValue("(sssssiiisi)",
        model_type, model_path, device, task,
        atom_refs ? atom_refs : "", charge, spin,
        compile_mode ? compile_mode : "",
        dtype ? dtype : "float64",
        turbo);
    if (!args) {
        set_python_error(err_msg, err_len);
        Py_DECREF(func);
        free(ctx);
        PyGILState_Release(gstate);
        return NULL;
    }

    ctx->calculator = PyObject_CallObject(func, args);
    Py_DECREF(args);
    Py_DECREF(func);

    if (!ctx->calculator) {
        set_python_error(err_msg, err_len);
        free(ctx);
        PyGILState_Release(gstate);
        return NULL;
    }

    /* Cache the bound method */
    ctx->calc_func = PyObject_GetAttrString(ctx->calculator,
                                             "get_energy_and_forces");
    if (!ctx->calc_func) {
        set_python_error(err_msg, err_len);
        Py_DECREF(ctx->calculator);
        free(ctx);
        PyGILState_Release(gstate);
        return NULL;
    }

    /* Cache numpy module and frequently used functions */
    ctx->numpy_mod = PyImport_ImportModule("numpy");
    if (!ctx->numpy_mod) {
        set_python_error(err_msg, err_len);
        Py_DECREF(ctx->calc_func);
        Py_DECREF(ctx->calculator);
        free(ctx);
        PyGILState_Release(gstate);
        return NULL;
    }

    ctx->np_frombuffer = PyObject_GetAttrString(ctx->numpy_mod, "frombuffer");
    if (!ctx->np_frombuffer) {
        set_python_error(err_msg, err_len);
        Py_DECREF(ctx->numpy_mod);
        Py_DECREF(ctx->calc_func);
        Py_DECREF(ctx->calculator);
        free(ctx);
        PyGILState_Release(gstate);
        return NULL;
    }

    ctx->initialized = 1;

    PyGILState_Release(gstate);
    return (pymlip_handle_t)ctx;
}


int pymlip_engrad(pymlip_handle_t handle,
                  int nat,
                  const double* positions_bohr,
                  const int* atomic_numbers,
                  double* energy_out,
                  double* gradient_out,
                  char* err_msg, int err_len)
{
    if (!handle) {
        set_error(err_msg, err_len, "pymlip_engrad: null handle");
        return 1;
    }

    PyMLIPContext* ctx = (PyMLIPContext*)handle;
    if (!ctx->initialized) {
        set_error(err_msg, err_len, "pymlip_engrad: not initialized");
        return 2;
    }

    if (nat <= 0) {
        set_error(err_msg, err_len, "pymlip_engrad: invalid nat (must be > 0)");
        *energy_out = 0.0;
        return 10;
    }

    /* Defensively zero outputs so error paths never return garbage */
    *energy_out = 0.0;
    memset(gradient_out, 0, (size_t)nat * 3 * sizeof(double));

    /* Acquire the GIL */
    PyGILState_STATE gstate = PyGILState_Ensure();

    /* Create positions array via bulk copy: bytes -> np.frombuffer -> reshape
     * This replaces element-by-element PyFloat/PyList construction */
    PyObject* pos_bytes = PyBytes_FromStringAndSize(
        (const char*)positions_bohr, nat * 3 * (Py_ssize_t)sizeof(double));
    if (!pos_bytes) {
        set_python_error(err_msg, err_len);
        PyGILState_Release(gstate);
        return 3;
    }

    PyObject* pos_flat = PyObject_CallFunction(ctx->np_frombuffer,
        "Os", pos_bytes, "float64");
    Py_DECREF(pos_bytes);
    if (!pos_flat) {
        set_python_error(err_msg, err_len);
        PyGILState_Release(gstate);
        return 3;
    }

    PyObject* pos_array = PyObject_CallMethod(pos_flat, "reshape", "(ii)", nat, 3);
    Py_DECREF(pos_flat);
    if (!pos_array) {
        set_python_error(err_msg, err_len);
        PyGILState_Release(gstate);
        return 4;
    }

    /* Create atomic numbers array via bulk copy */
    PyObject* z_bytes = PyBytes_FromStringAndSize(
        (const char*)atomic_numbers, nat * (Py_ssize_t)sizeof(int));
    if (!z_bytes) {
        set_python_error(err_msg, err_len);
        Py_DECREF(pos_array);
        PyGILState_Release(gstate);
        return 5;
    }

    /* frombuffer as int32 (C int), then astype int64 for Python/ASE */
    PyObject* z_i32 = PyObject_CallFunction(ctx->np_frombuffer,
        "Os", z_bytes, "int32");
    Py_DECREF(z_bytes);
    if (!z_i32) {
        set_python_error(err_msg, err_len);
        Py_DECREF(pos_array);
        PyGILState_Release(gstate);
        return 5;
    }

    PyObject* z_array = PyObject_CallMethod(z_i32, "astype", "s", "int64");
    Py_DECREF(z_i32);
    if (!z_array) {
        set_python_error(err_msg, err_len);
        Py_DECREF(pos_array);
        PyGILState_Release(gstate);
        return 5;
    }

    /* Call get_energy_and_forces(positions_bohr, atomic_numbers) */
    PyObject* result = PyObject_CallFunction(ctx->calc_func,
        "(OO)", pos_array, z_array);
    Py_DECREF(pos_array);
    Py_DECREF(z_array);

    if (!result) {
        set_python_error(err_msg, err_len);
        PyGILState_Release(gstate);
        return 6;
    }

    /* Unpack tuple (energy, gradient) */
    if (!PyTuple_Check(result) || PyTuple_Size(result) != 2) {
        set_error(err_msg, err_len,
                  "pymlip_engrad: expected (energy, gradient) tuple");
        Py_DECREF(result);
        PyGILState_Release(gstate);
        return 7;
    }

    /* Extract energy — .item() handles both numpy scalars and plain floats */
    PyObject* energy_obj = PyTuple_GET_ITEM(result, 0);
    PyObject* energy_item = PyObject_CallMethod(energy_obj, "item", NULL);
    if (energy_item) {
        *energy_out = PyFloat_AsDouble(energy_item);
        Py_DECREF(energy_item);
    } else {
        PyErr_Clear();
        *energy_out = PyFloat_AsDouble(energy_obj);
    }
    if (PyErr_Occurred()) {
        set_python_error(err_msg, err_len);
        Py_DECREF(result);
        PyGILState_Release(gstate);
        return 8;
    }

    /* Extract gradient via buffer protocol (bulk memcpy instead of element-by-element) */
    PyObject* grad_obj = PyTuple_GET_ITEM(result, 1);
    PyObject* flat_grad = PyObject_CallMethod(grad_obj, "ravel", NULL);
    if (!flat_grad) {
        set_python_error(err_msg, err_len);
        Py_DECREF(result);
        PyGILState_Release(gstate);
        return 9;
    }

    Py_buffer grad_buf;
    int buf_ok = PyObject_GetBuffer(flat_grad, &grad_buf, PyBUF_SIMPLE);
    if (buf_ok == 0 && grad_buf.len >= (Py_ssize_t)(3 * nat * sizeof(double))) {
        memcpy(gradient_out, grad_buf.buf, 3 * nat * sizeof(double));
        PyBuffer_Release(&grad_buf);
    } else {
        /* Fallback: element-by-element extraction */
        if (buf_ok == 0) PyBuffer_Release(&grad_buf);
        PyErr_Clear();
        for (int i = 0; i < 3 * nat; i++) {
            PyObject* val = PySequence_GetItem(flat_grad, i);
            if (!val) break;
            PyObject* vi = PyObject_CallMethod(val, "item", NULL);
            if (vi) {
                gradient_out[i] = PyFloat_AsDouble(vi);
                Py_DECREF(vi);
            } else {
                PyErr_Clear();
                gradient_out[i] = PyFloat_AsDouble(val);
            }
            Py_DECREF(val);
        }
    }

    Py_DECREF(flat_grad);
    Py_DECREF(result);

    PyGILState_Release(gstate);
    return 0;
}


int pymlip_engrad_batch(pymlip_handle_t handle, int batch_size, int nat,
                        const double* positions_batch,
                        const int* atomic_numbers,
                        double* energies_out, double* gradients_out,
                        char* err_msg, int err_len)
{
    if (!handle) {
        set_error(err_msg, err_len, "pymlip_engrad_batch: NULL handle");
        return 1;
    }

    PyMLIPContext* ctx = (PyMLIPContext*)handle;
    if (!ctx->initialized) {
        set_error(err_msg, err_len, "pymlip_engrad_batch: not initialized");
        return 2;
    }

    if (batch_size <= 0 || nat <= 0) {
        set_error(err_msg, err_len, "pymlip_engrad_batch: invalid batch_size or nat");
        return 10;
    }

    /* Zero all outputs */
    memset(energies_out, 0, (size_t)batch_size * sizeof(double));
    memset(gradients_out, 0, (size_t)batch_size * nat * 3 * sizeof(double));

    /* Acquire GIL ONCE for the entire batch */
    PyGILState_STATE gstate = PyGILState_Ensure();

    /* Build atomic numbers array once (shared across batch) */
    PyObject* z_bytes = PyBytes_FromStringAndSize(
        (const char*)atomic_numbers, nat * (Py_ssize_t)sizeof(int));
    if (!z_bytes) {
        set_python_error(err_msg, err_len);
        PyGILState_Release(gstate);
        return 3;
    }

    PyObject* z_i32 = PyObject_CallFunction(ctx->np_frombuffer, "Os", z_bytes, "int32");
    Py_DECREF(z_bytes);
    if (!z_i32) {
        set_python_error(err_msg, err_len);
        PyGILState_Release(gstate);
        return 3;
    }

    PyObject* z_array = PyObject_CallMethod(z_i32, "astype", "s", "int64");
    Py_DECREF(z_i32);
    if (!z_array) {
        set_python_error(err_msg, err_len);
        PyGILState_Release(gstate);
        return 3;
    }

    /* Process each structure in the batch */
    int status = 0;
    for (int b = 0; b < batch_size; b++) {
        const double* pos_b = positions_batch + (size_t)b * nat * 3;
        double* grad_b = gradients_out + (size_t)b * nat * 3;

        /* Create positions array for this structure */
        PyObject* pos_bytes = PyBytes_FromStringAndSize(
            (const char*)pos_b, nat * 3 * (Py_ssize_t)sizeof(double));
        if (!pos_bytes) {
            set_python_error(err_msg, err_len);
            status = 4;
            break;
        }

        PyObject* pos_flat = PyObject_CallFunction(ctx->np_frombuffer,
            "Os", pos_bytes, "float64");
        Py_DECREF(pos_bytes);
        if (!pos_flat) {
            set_python_error(err_msg, err_len);
            status = 4;
            break;
        }

        PyObject* pos_array = PyObject_CallMethod(pos_flat, "reshape", "(ii)", nat, 3);
        Py_DECREF(pos_flat);
        if (!pos_array) {
            set_python_error(err_msg, err_len);
            status = 5;
            break;
        }

        /* Call get_energy_and_forces(positions_bohr, atomic_numbers) */
        PyObject* result = PyObject_CallFunction(ctx->calc_func,
            "(OO)", pos_array, z_array);
        Py_DECREF(pos_array);

        if (!result) {
            set_python_error(err_msg, err_len);
            status = 6;
            break;
        }

        if (!PyTuple_Check(result) || PyTuple_Size(result) != 2) {
            set_error(err_msg, err_len,
                      "pymlip_engrad_batch: expected (energy, gradient) tuple");
            Py_DECREF(result);
            status = 7;
            break;
        }

        /* Extract energy */
        PyObject* energy_obj = PyTuple_GET_ITEM(result, 0);
        PyObject* energy_item = PyObject_CallMethod(energy_obj, "item", NULL);
        if (energy_item) {
            energies_out[b] = PyFloat_AsDouble(energy_item);
            Py_DECREF(energy_item);
        } else {
            PyErr_Clear();
            energies_out[b] = PyFloat_AsDouble(energy_obj);
        }
        if (PyErr_Occurred()) {
            set_python_error(err_msg, err_len);
            Py_DECREF(result);
            status = 8;
            break;
        }

        /* Extract gradient via buffer protocol */
        PyObject* grad_obj = PyTuple_GET_ITEM(result, 1);
        PyObject* flat_grad = PyObject_CallMethod(grad_obj, "ravel", NULL);
        if (!flat_grad) {
            set_python_error(err_msg, err_len);
            Py_DECREF(result);
            status = 9;
            break;
        }

        Py_buffer grad_buf;
        int buf_ok = PyObject_GetBuffer(flat_grad, &grad_buf, PyBUF_SIMPLE);
        if (buf_ok == 0 && grad_buf.len >= (Py_ssize_t)(3 * nat * sizeof(double))) {
            memcpy(grad_b, grad_buf.buf, 3 * nat * sizeof(double));
            PyBuffer_Release(&grad_buf);
        } else {
            /* Fallback: element-by-element */
            if (buf_ok == 0) PyBuffer_Release(&grad_buf);
            PyErr_Clear();
            for (int i = 0; i < 3 * nat; i++) {
                PyObject* val = PySequence_GetItem(flat_grad, i);
                if (!val) break;
                PyObject* vi = PyObject_CallMethod(val, "item", NULL);
                if (vi) {
                    grad_b[i] = PyFloat_AsDouble(vi);
                    Py_DECREF(vi);
                } else {
                    PyErr_Clear();
                    grad_b[i] = PyFloat_AsDouble(val);
                }
                Py_DECREF(val);
            }
        }

        Py_DECREF(flat_grad);
        Py_DECREF(result);
    }

    Py_DECREF(z_array);
    PyGILState_Release(gstate);
    return status;
}


void pymlip_free(pymlip_handle_t handle) {
    if (!handle) return;

    PyMLIPContext* ctx = (PyMLIPContext*)handle;

    if (python_initialized && Py_IsInitialized()) {
        PyGILState_STATE gstate = PyGILState_Ensure();

        Py_XDECREF(ctx->np_frombuffer);
        Py_XDECREF(ctx->calc_func);
        Py_XDECREF(ctx->calculator);
        Py_XDECREF(ctx->numpy_mod);

        PyGILState_Release(gstate);
    }

    free(ctx);
}


void pymlip_finalize_python(void) {
    if (!python_initialized) return;

    PyGILState_STATE gstate = PyGILState_Ensure();
    Py_XDECREF(setup_module);
    setup_module = NULL;
    /* Note: we don't call Py_Finalize() because it can cause
     * issues with numpy and other C-extension modules. The OS
     * will clean up when the process exits. */
    PyGILState_Release(gstate);

    python_initialized = 0;
}

#else /* !WITH_PYMLIP */

/* Stub implementations when compiled without Python support */

int pymlip_init_python(char* err_msg, int err_len) {
    if (err_msg && err_len > 0) {
        const char* msg = "pymlip support not compiled (need -DWITH_PYMLIP)";
        int n = (int)strlen(msg);
        if (n >= err_len) n = err_len - 1;
        memcpy(err_msg, msg, n);
        err_msg[n] = '\0';
    }
    return 1;
}

pymlip_handle_t pymlip_create(const char* model_type,
                               const char* model_path,
                               const char* device,
                               const char* task,
                               const char* atom_refs,
                               int charge,
                               int spin,
                               const char* compile_mode,
                               const char* dtype,
                               int turbo,
                               char* err_msg, int err_len) {
    (void)charge; (void)spin; (void)compile_mode; (void)dtype; (void)turbo;
    pymlip_init_python(err_msg, err_len);
    return NULL;
}

int pymlip_engrad(pymlip_handle_t handle, int nat,
                  const double* positions_bohr,
                  const int* atomic_numbers,
                  double* energy_out, double* gradient_out,
                  char* err_msg, int err_len) {
    if (energy_out) *energy_out = 0.0;
    if (gradient_out && nat > 0) memset(gradient_out, 0, (size_t)nat * 3 * sizeof(double));
    pymlip_init_python(err_msg, err_len);
    return 1;
}

int pymlip_engrad_batch(pymlip_handle_t handle, int batch_size, int nat,
                        const double* positions_batch,
                        const int* atomic_numbers,
                        double* energies_out, double* gradients_out,
                        char* err_msg, int err_len) {
    if (energies_out && batch_size > 0) memset(energies_out, 0, (size_t)batch_size * sizeof(double));
    if (gradients_out && nat > 0 && batch_size > 0)
        memset(gradients_out, 0, (size_t)batch_size * nat * 3 * sizeof(double));
    pymlip_init_python(err_msg, err_len);
    return 1;
}

void pymlip_free(pymlip_handle_t handle) { (void)handle; }
void pymlip_finalize_python(void) {}

#endif /* WITH_PYMLIP */
