module_name = 'convolution'

r = run_command(korali_gen, [ '--input', module_name + '.hpp.base', module_name + '.cpp.base', '--config', module_name + '.config', '--output', module_name + '.hpp', module_name + '.cpp' ])
if r.returncode() != 0
 output = r.stdout().strip()
 errortxt = r.stderr().strip()
 error('Failed to run module generation command. Details: \n' + output + errortxt)
endif

module_header = files([ module_name + '.hpp'])
module_source = files([ module_name + '.cpp'])
module_config = files([ module_name + '.config'])

install_headers(module_header,
  install_dir: run_command(header_path, [korali_install_headers, meson.current_source_dir()]).stdout().strip()
)

korali_include += include_directories('.')
korali_source += module_header
korali_source += module_source
korali_config += module_config