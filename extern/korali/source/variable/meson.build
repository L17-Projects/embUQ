korali_gen_variable = find_program('../../tools/build/build_variables_header.py')

run_command(korali_gen_variable, [ '--input',  'variable.hpp.base', korali_config, '--output', 'variable.hpp' ])
if r.returncode() != 0
 output = r.stdout().strip()
 errortxt = r.stderr().strip()
 error('Failed to run variable module generation command. Details: \n' + output + errortxt)
endif

module_header = files([ 'variable.hpp'])
install_headers(module_header,
  install_dir: run_command(header_path, [korali_install_headers, meson.current_source_dir()]).stdout().strip()
)

korali_include += include_directories('.')
korali_source += module_header

