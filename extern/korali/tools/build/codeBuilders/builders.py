import sys
import json
import os
from pathlib import Path

from . import header_builders as hb
from . import source_builders as sb
from . import auxiliar as aux
from . import variables as vr


def buildHeaderString( configFilePath, templateFilePath ):

  moduleTemplate = templateFilePath.read_text()

  moduleConfig = aux.loadModuleConfiguration( configFilePath, moduleTemplate )

  hb.checkHeaderTemplateString( moduleConfig, templateFilePath, moduleTemplate )

  headerString = hb.createHeaderDoxygenString(moduleConfig)

  headerString += moduleTemplate

  overrideFunctionString = hb.createOverrideFunctionString(moduleConfig)
  headerString = headerString.replace('public:', 'public: \n' + overrideFunctionString + '\n')

  calssDoxygenString = hb.createrClassDoxygenString( moduleConfig )
  headerString = headerString.replace('class ', calssDoxygenString + 'class ')

  declarationsString = hb.createHeaderDeclarations(moduleConfig)
  headerString = headerString.replace('public:', 'public: \n' + declarationsString + '\n')

  headerString = aux.replaceKeys( moduleConfig, headerString );

  return headerString



def buildCodeString( configFilePath, templateFilePath ):

  moduleTemplate = templateFilePath.read_text()

  moduleConfig = aux.loadModuleConfiguration( configFilePath, moduleTemplate )

  sb.checkSourceTemplateString( moduleConfig, templateFilePath, moduleTemplate )

  sourceString = sb.createSetConfiguration(moduleConfig)
  sourceString += sb.createGetConfiguration(moduleConfig)
  sourceString += sb.createApplyModuleDefaults(moduleConfig)
  sourceString += sb.createApplyVariableDefaults(moduleConfig)

  if 'Termination Criteria' in moduleConfig:
    sourceString += sb.createCheckTermination(moduleConfig)

  if 'Available Operations' in moduleConfig:
    sourceString += sb.createRunOperation(moduleConfig)

  if 'Conditional Variables' in moduleConfig:
    sourceString += sb.createGetPropertyPointer(moduleConfig)

  sourceString = moduleTemplate.replace( '__moduleAutoCode__',  sourceString )

  sourceString = aux.replaceKeys( moduleConfig, sourceString );

  return sourceString




def buildCodeFromTemplate( configFile, templateFile, outputFile=None ):
  configFilePath = Path(configFile)
  configFilePath = configFilePath.resolve()

  templateFilePath = Path(templateFile)
  templateFilePath = templateFilePath.resolve()

  if configFilePath.suffix != '.config' :
    sys.exit(f'[Korali] Error: {configFilePath} is not a .config file.\n')

  if configFilePath.parent != templateFilePath.parent:
    sys.exit(f'[Korali] Error: configuration file and template file are not in the same directory.\n{configFilePath.parent}\n{templateFilePath.parent}\n')

  if '.hpp.base' in str(templateFilePath):
    codeString = buildHeaderString( configFilePath, templateFilePath )
  elif '.cpp.base' in str(templateFilePath):
    codeString = buildCodeString( configFilePath, templateFilePath )
  else:
    sys.exit('[Korali] Error: Unknown extension in template file.\n')

  if outputFile == None:
    suffix = templateFilePath.suffix.replace('.base','')
    filePath = templateFilePath.with_suffix(suffix)
  else:
    filePath = Path(outputFile)

  filePath = filePath.resolve()
  # if does not exist create the path of filePath
  try:
      filePath.parents[0].mkdir(parents=True, exist_ok=False)
  except FileExistsError:
      pass

  # Check if the new version will produce a different result 
  doSave = True
  if (os.path.isfile(filePath)):
   with open(filePath, 'r') as file:
    prevString = file.read()
   if (prevString == codeString):
    doSave = False
  
  if (doSave): filePath.write_text(codeString)


def buildVariablesHeader(configFileList, templateFile, outputFile=None ):
  variableDeclarationList = ''
  variableDeclarationSet = set()

  for _file in configFileList:
    p = Path(_file)
    p = p.resolve()

    moduleConfig = json.loads( p.read_text() )
    string = ''

    if 'Variables Configuration' in moduleConfig: 
      for v in moduleConfig["Variables Configuration"]:
        varName = vr.getCXXVariableName(v["Name"])
        if (not varName in variableDeclarationSet):
          string += '/**\n'
          string += '* @brief [Module: ' + moduleConfig["Module Data"]["Class Name"] + '] ' + v["Description"] + '\n'
          string += '*/\n'
          string += '  ' + vr.getVariableType(v) + ' ' + varName + ';\n'
          variableDeclarationSet.add(varName)

    variableDeclarationList += string

  variableHeaderTemplatePath = Path(templateFile)
  variableHeaderTemplateString = variableHeaderTemplatePath.read_text()
  variableHeaderString = variableHeaderTemplateString.replace('// Variable Declaration List', variableDeclarationList)

  if outputFile == None:
    filePath = variableHeaderTemplatePath.with_suffix('.hpp')
  else:
    filePath = Path(outputFile)

  filePath = filePath.resolve()
  # if does not exist create  the path of filePath
  try:
      filePath.parents[0].mkdir(parents=True, exist_ok=False)
  except FileExistsError:
      pass

  # Check if the new version will produce a different result 
  doSave = True
  if (os.path.isfile(filePath)):
   with open(filePath, 'r') as file:
    prevString = file.read()
   if (prevString == variableHeaderString):
    doSave = False
  
  if (doSave): filePath.write_text(variableHeaderString)

  
  