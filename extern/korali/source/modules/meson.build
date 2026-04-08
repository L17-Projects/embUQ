modules_header = files([
  'module.hpp',
])
install_headers(modules_header,
  install_dir: run_command(header_path, [korali_install_headers, meson.current_source_dir()]).stdout().strip()
)

modules_source = files([
  'module.cpp',
])
korali_include += include_directories('.')
korali_source += modules_header
korali_source += modules_source
korali_config = files([])

subdir('conduit')
subdir('distribution')
subdir('experiment')
subdir('neuralNetwork')
subdir('problem')
subdir('solver')
