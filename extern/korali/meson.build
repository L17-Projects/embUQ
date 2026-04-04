project('korali', ['c', 'cpp'],
  version: '3.0.0',
  license: 'MIT',
  default_options: [
    'cpp_std=c++17',
    'b_ndebug=if-release',
    'b_asneeded=false',
    'default_library=shared'
    ]
)
summary({
  'buildtype': get_option('buildtype'),
  'default_library': get_option('default_library'),
  }, section: 'Build')

cpp = meson.get_compiler('cpp')
korali_conf = configuration_data()
pymodule = import('python')
python = pymodule.find_installation('python3', required: true)

korali = 'korali'
korali_install_base = python.get_install_dir() / korali
korali_install_headers = korali_install_base / 'include'
summary({
  'prefix': korali_install_base,
  'python': python.language_version()
  }, section: 'Python')

if get_option('b_coverage')
  gcov_args = ['-fno-inline', '-Wno-error=cpp']
  if cpp.get_id().to_lower() == 'gcc'
    gcov_args += ['-fno-default-inline', '-fno-inline-small-functions']
  endif
  add_project_arguments(gcov_args, language: ['cpp'])
endif

null_dep = dependency('', required : false)
korali_include = []
korali_deps = []
korali_deps += dependency('gsl', fallback: ['gsl', 'gsl_dep'], version : '>=2.5', required: true)
korali_deps += dependency('eigen3', fallback: ['eigen', 'eigen_dep'], required: true)
pybind11_dep = dependency('pybind11', fallback: ['pybind11', 'pybind11_dep'], required: true)
korali_deps += [ pybind11_dep ]

mpi_dep = null_dep
openmp_dep = null_dep
cuda_batch_driver_dep = null_dep
cuda_batch_nvrtc_dep = null_dep

if get_option('mpi')
  mpi_dep = dependency('mpi', language: 'cpp', required: false)
  if mpi_dep.found() == false
   mpi_dep = dependency('mpich', required: false)
  endif
endif

mpi4py_found = false
if get_option('mpi4py')
 assert(get_option('mpi'), 'MPI4Py can only be requested if MPI is also requested.')
 mpi4py_include = run_command(python, '-c', 'import mpi4py; print(mpi4py.get_include())').stdout().strip()
 assert(mpi4py_include.contains('include'), 'Could not find a correct MPI4Py Python module support.')
 message('Using mpi4py from ' + mpi4py_include)
 mpi4py_found = true
 korali_include += [ mpi4py_include ]
endif

if get_option('openmp')
  openmp_dep = dependency('openmp', required: false)
endif
korali_deps += [mpi_dep, openmp_dep]

if get_option('native_cuda_batch')
  cuda_home_env = run_command('sh', '-c', 'echo "${CUDA_HOME:-/opt/cuda}"', check: true).stdout().strip()
  cuda_include = cuda_home_env / 'include'
  cuda_lib64   = cuda_home_env / 'lib64'
  cuda_stubs   = cuda_home_env / 'stubs' / 'lib64'
  cuda_targets_stubs = cuda_home_env / 'targets' / 'x86_64-linux' / 'lib' / 'stubs'
  cuda_targets_lib   = cuda_home_env / 'targets' / 'x86_64-linux' / 'lib'
  korali_include += include_directories(cuda_include)
  assert(cpp.has_header('cuda.h', args: '-I' + cuda_include), 'Could not find cuda.h in ' + cuda_include)
  assert(cpp.has_header('nvrtc.h', args: '-I' + cuda_include), 'Could not find nvrtc.h in ' + cuda_include)
  cuda_batch_driver_dep = cpp.find_library('cuda', dirs: [cuda_stubs, cuda_targets_stubs, '/usr/lib64', '/usr/lib'], required: true)
  cuda_batch_nvrtc_dep = cpp.find_library('nvrtc', dirs: [cuda_lib64, cuda_targets_lib, '/opt/cuda/lib64'], required: true)
endif
korali_deps += [cuda_batch_driver_dep, cuda_batch_nvrtc_dep]

onednn_dep = null_dep
if get_option('onednn') or get_option('onednn_path') != ''
  if get_option('onednn_path') != ''
    onednn_dep = cpp.find_library('dnnl', dirs: [get_option('onednn_path') / 'lib', get_option('onednn_path') / 'lib64'], required: true)
    inc = get_option('onednn_path') / 'include'
    korali_include += include_directories(inc)
    assert(cpp.has_header('dnnl.h', args: '-I' + inc))
  elif get_option('onednn')
    onednn_dep = dependency('dnnl', required: true)
  endif
endif
korali_deps += onednn_dep

cudnn_dep = null_dep
if get_option('cudnn') or get_option('cudnn_path') != ''
  korali_deps += dependency('cuda', required: true)
  if get_option('cudnn_path') != ''
    cudnn_dep = cpp.find_library('cudnn', dirs: [get_option('cudnn_path') / 'lib', get_option('cudnn_path') / 'lib64'], required: true)
    inc = get_option('cudnn_path') / 'include'
    korali_include += include_directories(inc)
    assert(cpp.has_header('cudnn.h', args: '-I' + inc))
  elif get_option('cudnn')
    cudnn_dep = cpp.find_library('cudnn', required: true)
    assert(cpp.has_header('cudnn.h'))
  endif
endif
korali_deps += cudnn_dep

korali_conf.set('_KORALI_USE_MPI', get_option('mpi'))
korali_conf.set('_KORALI_USE_MPI4PY', mpi4py_found)
korali_conf.set('_KORALI_USE_ONEDNN', onednn_dep.found())
korali_conf.set('_KORALI_USE_CUDNN', cudnn_dep.found())
korali_conf.set('_KORALI_USE_CUDA_BATCH', get_option('native_cuda_batch'))
summary({
  'MPI': get_option('mpi'),
  'MPI4Py': mpi4py_found,
  'OpenMP': openmp_dep.found(),
  'NativeCudaBatch': get_option('native_cuda_batch'),
  'oneDNN': onednn_dep.found(),
  'cuDNN': cudnn_dep.found(),
  }, section: 'Dependencies')

subdir('source')
subdir('python/korali')

if get_option('test')
  subdir('tests')
  subdir('examples')
  subdir('study.cases')
endif
