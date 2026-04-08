#! /usr/bin/env python3
import os
import sys
import sysconfig
import subprocess
import korali
import pybind11
import libkorali as lk

def printHelp():
 print('[Korali] Usage: python korali.cxx [--help|--cflags|--libs]')
 
def main():

  if (len(sys.argv) != 2):
    print('[Korali] Syntax error on call to korali.cxx module.')
    printHelp()
    exit(-1)
  correctSyntax = False
  
  # Looking for Korali library
  koraliLib = os.path.realpath(lk.__file__)

  if (sys.argv[1] == '--cflags'):
    correctSyntax = True
    pythonCFlagsCommand = subprocess.Popen("python3-config --includes", shell=True, stdout=subprocess.PIPE)
    pythonCFlags = pythonCFlagsCommand.stdout.read().decode().rstrip("\n")
    pybind11Includes=pybind11.get_include()
    koraliIncludes=' -I' + os.path.dirname(korali.__file__) + '/include' + ' -I' +  os.path.dirname(korali.__file__) + '/../../../../include/' + ' -I' + os.path.dirname(korali.__file__) + '/../../source/' + ' -I' + os.path.dirname(korali.__file__) + '/../../build/source/'
    flags = '-D_KORALI_NO_MPI4PY -std=c++17' + koraliIncludes + ' -I' + sysconfig.get_path("include") + ' -I' + sysconfig.get_path("platinclude") + ' -I' + pybind11Includes + ' ' + pythonCFlags
    print(flags + ' ')
    exit(0)
    
  if (sys.argv[1] == '--libs'):
    correctSyntax = True
    
    pythonPrefixCommand = subprocess.Popen("python3-config --prefix", shell=True, stdout=subprocess.PIPE)
    pythonPrefix = pythonPrefixCommand.stdout.read().decode().rstrip("\n")

    try:
     pythonLibsCommand = subprocess.Popen("python3-config --ldflags --embed", shell=True, stdout=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
     pythonLibsCommand = subprocess.Popen("python3-config --ldflags", shell=True, stdout=subprocess.PIPE)
    pythonLibs = pythonLibsCommand.stdout.read().decode().rstrip("\n")
    try:
     pythonLibsCommand = subprocess.Popen("python3-config --ldflags --embed", shell=True, stdout=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
     pythonLibsCommand = subprocess.Popen("python3-config --ldflags", shell=True, stdout=subprocess.PIPE)
    pythonLibs = pythonLibsCommand.stdout.read().decode().rstrip("\n")
    
    flags = koraliLib + ' -Wl,-rpath,' + os.path.dirname(korali.__file__) + ' ' + pythonLibs + ' -Wl,-rpath,' + pythonPrefix + '/lib'
    print(flags + ' ')
    exit(0)

  if (sys.argv[1] == '--help'):
   printHelp()
   exit(0)

  if (correctSyntax == False):
   print('[Korali] Syntax error on call to korali.cxx module: Argument \'{0}\' not recognized.')
   printHelp()
   exit(-1)

if __name__ == '__main__':
  main()
